import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
def md(t): cells.append(nbf.v4.new_markdown_cell(t))
def code(t): cells.append(nbf.v4.new_code_cell(t))

md("""# 🚗 자율주행 보조 시스템 (RetinaNet · PyTorch)

Object detection 모델로 **주변에 사람이나 가까운 차량이 있는지** 확인하고,
위험하면 `"Stop"`, 안전하면 `"Go"`를 반환하는 미니 자율주행 보조장치를 만듭니다.

이 노트북은 **학습 없이 바로 동작**합니다. `torchvision`에 들어있는
**COCO 데이터로 사전학습된 RetinaNet**을 불러와 쓰기 때문에 `best.pth` 체크포인트가 필요 없어요.

**판단 조건**
- 사람이 한 명 이상 있으면 → `"Stop"`
- 차량의 크기(width 또는 height)가 **300px 이상**이면(=가까이 있음) → `"Stop"`
- 그 외 → `"Go"`
""")

md("## 0단계 · 필요한 라이브러리 설치\n\n처음 한 번만 실행하면 됩니다. `!`가 아니라 `%`로 시작하는 점에 주의 (지금 커널에 바로 설치돼요).")
code("""%pip install -q torch torchvision pillow matplotlib numpy""")

md("## 1단계 · 라이브러리 불러오기")
code("""import os
import numpy as np
import torch
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("PyTorch:", torch.__version__)
print("GPU 사용 가능:", torch.cuda.is_available())""")

md("""## 2단계 · 사전학습 RetinaNet 모델 불러오기

COCO 데이터셋으로 학습된 RetinaNet(ResNet50 backbone)을 불러옵니다.
처음 실행할 때 가중치를 내려받느라 시간이 조금 걸릴 수 있어요.
`model.eval()`은 \"추론(평가) 모드\"로 바꾸는 것으로, 추론할 때는 꼭 해줘야 합니다.""")
code('''weights = RetinaNet_ResNet50_FPN_Weights.COCO_V1
model = retinanet_resnet50_fpn(weights=weights)
model.eval()   # 추론 모드

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print("모델 로드 완료! device =", device)''')

md("""## 3단계 · 클래스(라벨) 정의

torchvision의 COCO 모델은 객체마다 번호가 정해져 있습니다. 우리가 쓸 건 **사람**과 **차량**.

- `1` = person(사람)
- `3` = car, `4` = motorcycle, `6` = bus, `8` = truck""")
code("""COCO_LABELS = {1: "person", 2: "bicycle", 3: "car", 4: "motorcycle",
               6: "bus", 8: "truck"}

PERSON_CLASS    = 1
VEHICLE_CLASSES = {3, 4, 6, 8}   # 차량으로 볼 클래스""")

md("""## 4단계 · 이미지 불러오기 + 추론 함수

`run_detection`은 이미지 경로를 받아 모델로 추론한 뒤,
**박스 좌표·클래스·점수**를 정리해 돌려줍니다.

> torchvision detection 모델은 박스를 **픽셀 좌표 `[x1, y1, x2, y2]`** 로 바로 줍니다.
> (정규화 변환이 필요 없어서 300px 조건 판단이 간단해요.)""")
code('''@torch.no_grad()                      # 추론에는 기울기 계산이 필요 없음
def run_detection(img_path):
    # 1) 이미지 읽기 (RGB)
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    # 2) 텐서로 변환 ([3,H,W], 0~1) 후 모델 입력 (리스트 형태)
    img_tensor = to_tensor(img).to(device)
    output = model([img_tensor])[0]   # 이미지 1장 -> 결과 1개

    boxes  = output["boxes"].cpu().numpy()    # (N,4) 픽셀 [x1,y1,x2,y2]
    labels = output["labels"].cpu().numpy()   # (N,)
    scores = output["scores"].cpu().numpy()   # (N,)

    detections = []
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        detections.append({
            "class": int(label),
            "score": float(score),
            "box": (x1, y1, x2, y2),
            "w": x2 - x1,             # 픽셀 너비
            "h": y2 - y1,             # 픽셀 높이
        })
    return img_np, detections''')

md("## 5단계 · 결과 시각화\n\n탐지된 객체를 이미지 위에 박스로 그려 눈으로 확인합니다.")
code('''def visualize_detection(img_path, score_thresh=0.3):
    img_np, detections = run_detection(img_path)

    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img_np)

    for d in detections:
        if d["score"] < score_thresh:
            continue
        x1, y1, x2, y2 = d["box"]
        name = COCO_LABELS.get(d["class"], str(d["class"]))
        rect = patches.Rectangle((x1, y1), d["w"], d["h"],
                                 linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'{name} {d["score"]:.2f}',
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.6, pad=1))

    ax.axis("off")
    plt.tight_layout()
    plt.show()''')

md("""## 6단계 · 자율주행 보조 함수 `self_drive_assist`

핵심입니다. 추론 결과를 보고 **Stop / Go**를 결정합니다.

- `score_thresh`: 이 점수보다 낮은(=확실하지 않은) 탐지는 무시
- 사람이 한 명이라도 있으면 → `"Stop"`
- 차량이면서 너비 또는 높이가 `size_limit`(기본 300px) 이상이면 → `"Stop"`
- 둘 다 아니면 → `"Go"`""")
code('''def self_drive_assist(img_path, size_limit=300, score_thresh=0.3):
    _, detections = run_detection(img_path)

    for d in detections:
        if d["score"] < score_thresh:       # 확실하지 않은 탐지는 무시
            continue

        # 조건 1: 사람이 한 명 이상
        if d["class"] == PERSON_CLASS:
            return "Stop"

        # 조건 2: 차량 크기(너비 또는 높이)가 300px 이상
        if d["class"] in VEHICLE_CLASSES and (d["w"] >= size_limit or d["h"] >= size_limit):
            return "Stop"

    return "Go"''')

md("""## 7단계 · 실행해 보기

본인 이미지 경로로 바꿔서 실행하세요.
- 리눅스/맥: `os.getenv("HOME") + "/..."`
- 윈도우: `r"C:\\경로\\stop_1.png"` 처럼 직접 적는 게 편해요.""")
code('''# 본인 이미지 경로로 변경
img_path = os.getenv("HOME") + "/work/object_detection/data/stop_1.png"
# 윈도우 예시: img_path = r"C:\\Users\\hanwo\\Pictures\\stop_1.png"

# 1) 눈으로 확인
visualize_detection(img_path)

# 2) 판단 결과
print("판단:", self_drive_assist(img_path))''')

md("""## 8단계 · 예외 상황도 생각해 보기 (과제 힌트)

과제에 *"예외 상황을 포함하기 위해서 어떤 기준이 또 필요할까요?"* 라는 질문이 있었죠.
실제 자율주행은 **애매하면 멈추는(safe by default)** 쪽으로 설계합니다.

- **이미지를 못 읽거나 파일이 없을 때** → 판단 불가이므로 안전하게 `"Stop"`
- **추론 자체가 실패할 때** → `"Stop"`
- **자전거·오토바이 같은 취약한 도로 사용자**도 가까우면 멈추도록 확장 가능

아래는 예외 처리를 넣은 버전입니다. 위 함수 대신 써도 됩니다.""")
code('''def self_drive_assist_safe(img_path, size_limit=300, score_thresh=0.3):
    if not os.path.exists(img_path):
        print(f"[경고] 이미지를 찾을 수 없습니다: {img_path} -> 안전하게 Stop")
        return "Stop"

    try:
        _, detections = run_detection(img_path)
    except Exception as e:
        print(f"[경고] 추론 실패: {e} -> 안전하게 Stop")
        return "Stop"

    for d in detections:
        if d["score"] < score_thresh:
            continue
        if d["class"] == PERSON_CLASS:
            return "Stop"
        if d["class"] in VEHICLE_CLASSES and (d["w"] >= size_limit or d["h"] >= size_limit):
            return "Stop"
    return "Go"


print("판단(안전버전):", self_drive_assist_safe(img_path))''')

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
}

with open("/sessions/dazzling-relaxed-sagan/mnt/outputs/self_driving_retinanet_pytorch.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

import nbformat, ast
loaded = nbformat.read("/sessions/dazzling-relaxed-sagan/mnt/outputs/self_driving_retinanet_pytorch.ipynb", as_version=4)
nbformat.validate(loaded)
for i, c in enumerate(loaded.cells):
    if c.cell_type == "code":
        src = "\n".join(l for l in c.source.splitlines() if not l.strip().startswith(("!","%")))
        try: ast.parse(src)
        except SyntaxError as e: print("SYNTAX ERROR cell", i, e)
print("OK cells:", len(loaded.cells))
