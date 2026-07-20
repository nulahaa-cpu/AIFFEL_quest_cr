# AIFFEL Campus Online Code Peer Review Template
- 코더 : 김한울
- 리뷰어 : (리뷰어 이름)


# PRT(Peer Review Template)

- [x] **1. 주어진 문제를 해결하는 완성된 코드가 제출되었나요?**
    - 한국어 위키 말뭉치로 vocab size 8000의 SentencePiece 토크나이저를 직접 학습했습니다. (`[PAD] [UNK] [BOS] [EOS] [SEP] [CLS] [MASK]` 특수 토큰 포함)
    - MLM(마스크 언어 모델) 데이터셋: 전체 토큰의 15%를 마스킹하고, 그중 80%는 `[MASK]`, 10%는 랜덤 토큰, 10%는 원본 유지 규칙을 word 단위로 적용했습니다. (실측 마스킹 비율 14.4%)
    - NSP(다음 문장 예측) 데이터셋: 50% 확률로 연속/랜덤 문장쌍을 구성하고 segment 0/1과 `[CLS]`/`[SEP]` 구분자를 처리했습니다. (실측 is_next 비율 47%)
    - 생성한 데이터셋을 json으로 저장한 뒤 `np.memmap`으로 변환하여 학습 시 메모리 사용을 최소화했습니다. (인스턴스 66,739개)
    - 전체 파라미터 **1,031,362개(약 1.03M)** 의 mini BERT를 PyTorch로 직접 구현했습니다. (d_model 96, encoder 2층, head 4개, MLM 출력층은 embedding과 weight tying)
    - warmup 10% + 선형 감소 학습률 스케줄로 **10 Epoch** 학습을 완료했습니다. (Colab T4 기준 전체 실행 약 15분)
    - 대표 결과:
        - MLM loss: `7.61 → 6.01`, MLM accuracy: `2.5% → 13.4%` (랜덤 예측 0.0125% 대비 1000배 이상)
        - NSP loss: `0.63 → 0.15`, NSP accuracy: `63.7% → 94.6%`
        - 두 task 모두 발산이나 loss spike 없이 단조 감소하며 안정적으로 수렴
![cap1](training_curves.png)

- [x] **2. 핵심 코드의 주석 또는 doc string을 보고 코드를 이해할 수 있었나요?**
    - `토크나이저 학습 → MASK 생성 → NSP pair 생성 → memmap 데이터셋 → 모델 구현 → pretrain → 시각화` 순서로 단계별 마크다운 설명이 있어 전체 흐름을 따라가기 쉽습니다.
    - `create_pretrain_mask()`에 마스킹을 word 단위(`▁` 시작 조각 + 뒤따르는 subword)로 묶는 이유와 80/10/10 분기가 주석으로 설명되어 있습니다.
    - `create_pretrain_instances()`에 NSP의 is_next 결정 로직(랜덤 문서 선택 시 자기 자신 제외)과 `trim_tokens()`의 A/B 자르는 방향이 명시되어 있습니다.
    - MLM loss에서 `ignore_index=0`으로 마스크되지 않은 위치를 제외하는 이유, weight tying으로 파라미터를 절약하는 부분도 주석으로 확인할 수 있습니다.

- [x] **3. 디버깅 기록 또는 새로운 시도와 추가 실험을 수행했나요?**
    - **디버깅 기록**
        - 학습 환경의 작업 디렉토리와 데이터 경로가 달라 `FileNotFoundError`가 반복 발생 → 경로 확인 셀을 추가하고 말뭉치가 없으면 HuggingFace에서 자동 생성하도록 수정했습니다.
        - 원격 서버에서 wget으로 받은 말뭉치가 실제로는 로그인 페이지 HTML(23KB)이었던 문제를 파일 크기·내용 검증으로 발견하고 데이터 준비 방식을 변경했습니다.
        - `create_pretrain_instances()` 호출부의 인자 누락(`missing 3 required positional arguments`)과 `pieces` 미정의 버그를 수정했습니다.
        - bias 없는 LayerNorm에 `nn.init.zeros_(module.bias)`를 적용하다 발생한 `AttributeError`를 `if module.bias is not None:` 가드로 해결했습니다.
    - **추가 실험: mini ModernBERT 비교**
        - 같은 토크나이저·말뭉치·파라미터 예산(1.01M)으로 2024년 ModernBERT 방식을 구현해 비교했습니다: NSP 제거 + MLM 30% 단독, RoPE(회전 위치 인코딩), GeGLU, pre-LN, bias 제거.
        - 더 어려운 30% 마스킹 조건에서도 MLM accuracy **22.5% vs 13.4%** 로 원조 방식을 크게 앞섰습니다.
        - 정성 평가(fill-mask)에서도 원조는 조사·문장부호 위주로 예측한 반면, ModernBERT 방식은 "대한민국의 수도는 [MASK] 이다"에서 `가수, 배우, 정치인` 등 위키 문형에 맞는 명사 범주를 예측하는 단계까지 도달했습니다.
![cap2](modern_comparison.png)

- [x] **4. 회고를 잘 작성했나요?**
    - NSP는 정확도 95%에 빠르게 도달했는데, 랜덤 문장쌍이 다른 문서에서 오기 때문에 주제만으로 구분 가능한 쉬운 과제라는 점(RoBERTa가 NSP를 제거한 근거)과 연결하여 해석했습니다.
    - MLM 정확도 13.4%는 낮지만 랜덤 대비 1000배 이상이며, 5 Epoch 이후 곡선이 평탄해지는 것은 1M 파라미터라는 모델 용량의 한계로 해석했습니다.
    - fill-mask 정성 평가를 통해 언어 모델이 빈도 → 문법(조사) → 의미 범주 → 세상 지식 순으로 능력을 획득한다는 것을 미니 스케일에서 직접 관찰했습니다. ('겨울→춥다' 같은 의미 추론은 실패했지만 고빈도 문법 요소는 예측)
    - 개선 방향: 모델 크기보다 데이터 양과 학습 스텝 확대가 우선이며, validation set 분리로 일반화 확인이 필요하다고 정리했습니다.

- [x] **5. 코드가 간결하고 효율적인가요?**
    - 데이터셋을 `np.memmap`으로 디스크에서 직접 읽어 대용량 데이터도 RAM을 거의 쓰지 않습니다.
    - MLM 출력층을 입력 embedding과 weight tying하여 약 77만 개 파라미터를 절약했습니다.
    - vocab 8000이 int16 범위에 들어가는 점을 이용해 memmap 저장 공간을 절반으로 줄였습니다.
    - 단계마다 누적 경과 시간을 출력하여 전체 실행이 20분 이내임을 확인할 수 있게 했습니다.


# 회고(참고 링크 및 코드 개선)

    1M 파라미터 제약에서 embedding(8000x96=77만 개)이 전체의 75%를 차지하기 때문에
    encoder를 얇게(2층) 가져갈 수밖에 없었다. 같은 예산에서 구조를 어떻게 배분하느냐가
    이 과제의 핵심 설계 결정이었다.

    원조 BERT(2018)와 ModernBERT(2024) 방식을 같은 조건에서 직접 비교하면서,
    효과 없는 과제를 버리고(NSP 제거) 위치 정보를 수학으로 대체하고(RoPE)
    학습을 안정화하는(pre-LN) 6년간의 개선이 1M 스케일에서도 유효함을 확인했다.

    참고: BERT (Devlin et al., 2018), RoBERTa (Liu et al., 2019),
    RoFormer/RoPE (Su et al., 2021), ModernBERT (Warner et al., 2024)
