# ═══════════════════════════════════════════════════════════════════
#  E6 노트북에 이어 붙이는 RM + PPO 추가 셀 (총 3개, 약 20분)
#
#  사용법: E6가 전부 끝난 뒤, 노트북 맨 아래에 코드 셀을 3개 만들고
#          아래 "셀 1/3", "셀 2/3", "셀 3/3" 블록을 하나씩 붙여넣어
#          순서대로 실행하세요. (E6와 같은 런타임이어야 합니다!)
#
#  하는 일: ① 보상 모델(RM)을 랭킹 데이터로 학습
#           ② E0(SFT) 모델에 PPO 강화학습 적용
#           ③ 동일 테스트셋 60개로 SFT vs PPO 정량/정성 비교 + CSV 기록
#  → 평가기준 "SFT 모델과 RM 적용 모델 결과 비교/분석" 충족용
# ═══════════════════════════════════════════════════════════════════


# ════════════════ 셀 1/3: 메모리 정리 + 보상 모델(RM) 학습 (~5분) ════════════════
import gc
import torch.nn as nn
for _v in ['model', 'base_model', 'trainer']:          # E6의 1.2B 모델 메모리 해제
    if _v in globals(): del globals()[_v]
gc.collect(); torch.cuda.empty_cache()
!pip install -q trl==0.11.4

from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
ktok = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2',
    bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

# 랭킹 데이터(chatGPT>davinci>ada 답변 순위) → (좋은 답, 나쁜 답) 쌍
rm_raw = load_jsonl('KoChatGPT/data_kochatgpt/kochatgpt_2_RM.jsonl')
pairs = []
for d in rm_raw:
    ranked = [d['completion_0'], d['completion_1'], d['completion_2']]
    best, mid, worst = (ranked[i] for i in d['ranking'])   # ranking = 품질 순 인덱스
    for c, r in [(best, mid), (best, worst), (mid, worst)]:
        if c.strip() and r.strip():
            pairs.append({'prompt': d['prompt'], 'chosen': c, 'rejected': r})
random.shuffle(pairs)
pairs = pairs[:2000]
print(f'랭킹 쌍 {len(pairs)}개로 RM 학습')

class RewardModel(nn.Module):
    """KoGPT-2 몸통 + 점수 1개를 내는 머리(nn.Linear) — 수업 노트북 Q15의 구조 그대로"""
    def __init__(self):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2').transformer
        self.head = nn.Linear(self.backbone.config.n_embd, 1)
    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        idx = attention_mask.sum(1) - 1                     # 마지막 실제 토큰 위치
        return self.head(h[torch.arange(h.size(0)), idx]).squeeze(-1)

rm = RewardModel().cuda()
opt = torch.optim.AdamW(rm.parameters(), lr=1e-5)

def rm_encode(texts):
    return ktok(texts, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to('cuda')

rm.train(); t0 = time.time(); B = 8
for i in range(0, len(pairs), B):
    batch = pairs[i:i+B]
    ch = rm_encode([f"{b['prompt']} {b['chosen']}"   for b in batch])
    rj = rm_encode([f"{b['prompt']} {b['rejected']}" for b in batch])
    # 좋은 답 점수 > 나쁜 답 점수가 되도록 학습 (수업 노트북 Q16의 sigmoid 랭킹 loss)
    loss = -torch.nn.functional.logsigmoid(rm(**ch) - rm(**rj)).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (i // B) % 50 == 0: print(f'  step {i//B:>3} | loss {loss.item():.4f}')
print(f'✅ RM 학습 완료 ({(time.time()-t0)/60:.1f}분)')

@torch.no_grad()
def rm_score(texts, bs=16):
    rm.eval(); out = []
    for i in range(0, len(texts), bs):
        out += rm(**rm_encode(texts[i:i+bs])).tolist()
    return out

# 품질이 점점 좋아지는 세 문장 → 점수도 점점 올라가면 정상 (수업 노트북과 같은 검증)
tests = ['인공지능은 그냥 나쁜 것입니다.',
         '인공지능은 컴퓨터가 학습하는 기술입니다.',
         '인공지능은 데이터로부터 패턴을 학습해 추론하는 기술로, 의료·금융 등 다양한 산업에서 활용됩니다.']
for t, s in zip(tests, rm_score(tests)):
    print(f'{s:+.3f} | {t}')


# ════════════════ 셀 2/3: E0(SFT) 모델에 PPO 적용 (~10분) ════════════════
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead

kbase = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')

# 백업 미복원 등으로 E0 어댑터가 없으면 3분짜리 미니 SFT로 만들어 사용
if not os.path.exists('adapters/kogpt2_clean'):
    print('E0 어댑터 없음 → 미니 SFT 학습으로 생성 (~3분)')
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    mini = get_peft_model(kbase, LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32,
        lora_dropout=0.05, target_modules=['c_attn'], fan_in_fan_out=True)).cuda()
    class _DS(torch.utils.data.Dataset):
        def __init__(self, data):
            self.ex = [ktok(PROMPT_TMPL.format(q=d['prompt']) + ' ' + d['completion'] + ktok.eos_token,
                            truncation=True, max_length=MAX_LEN)['input_ids'] for d in data]
        def __len__(self): return len(self.ex)
        def __getitem__(self, i): return {'input_ids': self.ex[i]}
    Trainer(model=mini,
            args=TrainingArguments(output_dir='tmp_mini', num_train_epochs=2,
                per_device_train_batch_size=16, gradient_accumulation_steps=2, learning_rate=2e-4,
                fp16=True, logging_steps=50, save_strategy='no', report_to='none', seed=SEED),
            train_dataset=_DS(dataset_all[80:2080]),
            data_collator=DataCollatorForLanguageModeling(tokenizer=ktok, mlm=False)).train()
    mini.save_pretrained('adapters/kogpt2_clean')
    del mini; gc.collect(); torch.cuda.empty_cache()
    kbase = AutoModelForCausalLM.from_pretrained('skt/kogpt2-base-v2')   # 원본 새로 로드

sft_peft = PeftModel.from_pretrained(kbase, 'adapters/kogpt2_clean', is_trainable=True)
actor = AutoModelForCausalLMWithValueHead.from_pretrained(sft_peft).cuda()

@torch.no_grad()
def kgen(prompts, mdl, bs=8):
    ktok.padding_side = 'left'; outs = []
    mdl.eval()
    for i in range(0, len(prompts), bs):
        batch = [PROMPT_TMPL.format(q=p) for p in prompts[i:i+bs]]
        enc = ktok(batch, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LEN).to('cuda')
        g = mdl.generate(**enc, max_new_tokens=64, do_sample=True, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.2, eos_token_id=ktok.eos_token_id, pad_token_id=ktok.pad_token_id)
        outs += [ktok.decode(g[j][enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                 for j in range(len(batch))]
    ktok.padding_side = 'right'; return outs

# PPO 학습 "전" SFT 상태의 답변을 먼저 저장 (비교 기준)
hyps_sft60 = kgen(test_prompts, actor)
print('SFT(PPO 전) 테스트 답변 60개 확보')

# ── PPO 루프: actor가 답변 생성 → RM이 점수 → 점수가 오르는 방향으로 업데이트 ──
ppo_prompts_all = [d['prompt'] for d in load_jsonl('KoChatGPT/data_kochatgpt/kochatgpt_3_PPO.jsonl')]
random.shuffle(ppo_prompts_all)
PPO_STEPS, BATCH = 8, 16
cfg = PPOConfig(batch_size=BATCH, mini_batch_size=4, learning_rate=1.4e-5, seed=SEED)
ppo = PPOTrainer(cfg, actor, ref_model=None, tokenizer=ktok)   # ref=None → 어댑터 끈 상태가 기준모델

gen_kw = dict(max_new_tokens=48, do_sample=True, top_p=0.9, temperature=0.7,
              pad_token_id=ktok.pad_token_id, eos_token_id=ktok.eos_token_id)
t0 = time.time()
for step in range(PPO_STEPS):
    prompts = ppo_prompts_all[step*BATCH:(step+1)*BATCH]
    queries = [ktok(PROMPT_TMPL.format(q=p), return_tensors='pt', truncation=True,
                    max_length=128)['input_ids'][0].cuda() for p in prompts]
    responses = ppo.generate(queries, return_prompt=False, **gen_kw)
    texts = [ktok.decode(q, skip_special_tokens=True) + ' ' + ktok.decode(r, skip_special_tokens=True)
             for q, r in zip(queries, responses)]
    rewards = [torch.tensor(s) for s in rm_score(texts)]
    stats = ppo.step(queries, responses, rewards)
    print(f'step {step+1}/{PPO_STEPS} | 평균 보상 {np.mean([r.item() for r in rewards]):+.3f}'
          f' | KL {stats.get("objective/kl", float("nan")):.2f}')
print(f'✅ PPO 완료 ({(time.time()-t0)/60:.1f}분) — 평균 보상이 step을 거치며 오르면 성공')


# ════════════════ 셀 3/3: SFT vs PPO 비교 + 기록 (~3분) ════════════════
import pandas as pd

hyps_ppo60 = kgen(test_prompts, actor)          # PPO 적용 후 답변

s_sft = score_set(test_refs, hyps_sft60)
s_ppo = score_set(test_refs, hyps_ppo60)
rm_sft = np.mean(rm_score([f'{p} {h}' for p, h in zip(test_prompts, hyps_sft60)]))
rm_ppo = np.mean(rm_score([f'{p} {h}' for p, h in zip(test_prompts, hyps_ppo60)]))

print(f"{'Metric':<12}{'SFT(E0)':>10}{'PPO 후':>10}{'변화':>9}")
print('-' * 41)
for k in s_sft:
    print(f"{k:<12}{s_sft[k]:>10.4f}{s_ppo[k]:>10.4f}{s_ppo[k]-s_sft[k]:>+9.4f}")
print(f"{'RM 보상점수':<12}{rm_sft:>+10.3f}{rm_ppo:>+10.3f}{rm_ppo-rm_sft:>+9.3f}   ← RM을 평가자로 활용")

CSV = 'results/experiments.csv'
df = pd.read_csv(CSV) if os.path.exists(CSV) else pd.DataFrame()
row = {'exp': 'E-PPO', 'desc': 'RM 보상 기반 PPO (E0 SFT 모델에 적용)', 'model': 'kogpt2',
       'clean': True, 'decode': 'sampling',
       **{k: round(v, 4) for k, v in s_ppo.items()}, 'train_min': ''}
if len(df): df = df[df['exp'] != 'E-PPO']
df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.to_csv(CSV, index=False)

with open('results/gen_EPPO.json', 'w', encoding='utf-8') as f:
    pyjson.dump({'prompts': test_prompts[:10], 'refs': test_refs[:10],
                 'sft': hyps_sft60[:10], 'ppo': hyps_ppo60[:10]}, f, ensure_ascii=False, indent=1)

# 데모 4문항 정성 비교 (이전 E0 데모가 있으면 나란히 출력)
demo_questions = ["불고기용 고기 한우에요?", "리처드 닉슨이 43대 부통령직을 수행한 년도는?",
                  "시카고 오헤어 국제공항은 어디에 있어?", "오늘 미세먼지 어때?"]
demo_ppo = kgen(demo_questions, actor)
with open('results/demo_EPPO.json', 'w', encoding='utf-8') as f:
    pyjson.dump(dict(zip(demo_questions, demo_ppo)), f, ensure_ascii=False, indent=1)
old = pyjson.load(open('results/demo_E0.json', encoding='utf-8')) if os.path.exists('results/demo_E0.json') else {}
for q, a in zip(demo_questions, demo_ppo):
    print('=' * 70); print(f'❓ {q}')
    if q in old: print(f'  [SFT] {old[q]}')
    print(f'  [PPO] {a}')

print('\n💾 다 끝나면 백업 셀(BACKUP=True)로 zip 저장하는 것 잊지 마세요!')
