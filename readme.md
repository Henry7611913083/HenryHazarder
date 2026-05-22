# HenryHazarder 설명

input 폴더의 이미지를 Transformers 기반 여러 로컬 검열 모델로 앙상블 최적화하는 연구용 프로젝트입니다

많은 분들도 저도 했다가 말았던 착각이 있는데 nsfw 손실은 소프트 라벨 때문에 0.9999 또는 1.000까지 안 올라가요

## 만든 동기

![Claude Sonnet 4.5](https://img.shields.io/badge/Claude_Sonnet_4.5-D97757?logo=anthropic&logoColor=white)
![Claude Haiku 4.5](https://img.shields.io/badge/Claude_Haiku_4.5-D97757?logo=anthropic&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/GitHub_Copilot-000000?logo=githubcopilot&logoColor=white)

먼저, 만든 이유부터 설명하자면 **Glaze**, **Nightshade** 등등같은 적대적 필터의 목적부터가 학습 데이터 포이즈닝 때문에입니다.

네, 추론 단계에선 **정상적으로 인식해서** 문제입니다,  하지만 우리 인간들이 기대한 건 **LLM도 못 알아볼 만큼 왜곡하는 것이였습니다**.

제가 거기서 더 효과적이고 빠른 방법을 생각해봤는데 **일부러 검열 모델이 검열해야 할 흉물로 인식시켜서** 학습 데이터에 못 들어가게 하는 방법입니다. 

> 그래서 더 강화시킬 수 있는 방법을 제미나이에게 여러 번 물어보며 몇 개를 뽑고 그 다음 제가 떠올린 방법을 초기 코드를 Claude Sonnet 4.5가 짜고 세션 한도 때문에 깃허브 코파일럿으로 갔는데 클로드 하이쿠 3.5가 기본값이라 그대로 썼는데 생각보다 성능이 좋더라고요. (그 후엔 소넷 4.6, 하이쿠 4.6 번갈아썼지만)

## 설치

### 필수 요구사항
- Python 3.12+
- CUDA 12.1+ (GPU 사용 시) 또는 ROCm 6.0+ (AMD GPU)

### 기본 설치

```bash
# 저장소 복제
git clone https://github.com/Henry7611913083/HenryHazarder
cd HenryHazarder

# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell:
irm https://astral.sh/uv/install.ps1 | iex

# 의존성 설치
uv sync
```

### GPU 지원 (선택사항)

```bash
# NVIDIA CUDA 12.x
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# AMD ROCm 6.0+
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0

# Apple Silicon (MPS)
# 기본 torch에서 자동 지원됨
```

## 셋팅과 실행

```bash
# 입력 폴더 생성
mkdir input
cp yourimage.png input/

# 기본 실행
uv run python nsfw_attack.py

# GPU 명시적 지정 (CUDA:0)
CUDA_VISIBLE_DEVICES=0 uv run python nsfw_attack.py

# accelerate 사용 (분산 학습/멀티 GPU)
accelerate launch nsfw_attack.py

# 고급 옵션
uv run python nsfw_attack.py \
  --input ./input \
  --output ./output \
  --steps 200 \
  --eps 0.05 \
  --robust \
  --model Falconsai/nsfw_image_detection \
  --model another-model-id
```

## 파라미터별 특성

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--input` | `./input` | 입력 이미지 디렉토리 |
| `--output` | `./output` | 출력 이미지 디렉토리 |
| `--model` | `Falconsai/nsfw_image_detection` | HuggingFace 모델 ID. 여러 번 지정하면 앙상블로 동작 |
| `--eps` | `0.03` | 픽셀당 최대 변화량 (≈ 7.6/255). 높을수록 공격력++ 시각적 변화++ |
| `--steps` | `100` | PGD 반복 횟수. 높을수록 수렴도++, 시간++ |
| `--lr` | `0.005` | Adam 학습률. 너무 크면 발산, 너무 작으면 수렴 느림 |
| `--lambda-lpips` | `2.0` | LPIPS 지각적 손실 가중치. 높을수록 원본 외관 보존++ 공격력-- |
| `--mu-l2` | `0.5` | L2 픽셀 정규화 가중치. 높을수록 perturbation 크기 억제 |
| `--label-smooth` | `0.1` | 소프트 라벨 평활화율 α. 목표 NSFW 확률이 (1-α)로 설정됨. 낮을수록 공격력++ gradient 불안정 위험++ |
| `--lambda-kl` | `0.3` | KL 발산 정규화 가중치. 높을수록 원본 분포 보존++ 공격력-- |
| `--kl-temp` | `2.0` | KL 발산 Temperature scaling. 높을수록 분포가 부드러워져 gradient 안정++ |
| `--no-lpips` | `False` | LPIPS 비활성화. 속도++ 지각적 품질-- |
| `--resize` | `None` | 처리 전 이미지를 정사각형으로 리사이즈 (예: `224`) |
| `--multi-scale` | `False` | 0.5×, 0.75×, 1.0× 스케일에서 평가해 스케일 변환 내성 부여 |
| `--use-jpeg` | `False` | quality 95/85/75로 JPEG 압축을 시뮬레이션해 압축 내성 부여 |
| `--use-transforms` | `False` | 회전·스케일·이동 변환을 적용해 기하 변환 내성 부여 |
| `--preserve-hf` | `False` | DCT 고주파 성분을 증폭해 다운샘플링 후에도 perturbation 생존율++ (scipy 필요) |
| `--robust` | `False` | 위 4가지 Robustness 옵션을 한 번에 활성화 |

(여기서부턴 클로드가 맘대로 쓴거)
## 성능 최적화

### 메모리 부족 시
```bash
# 배치 크기 축소 (단일 이미지 처리)
# --batch-size 1 (아직 미구현, 기본 1개씩)

# LPIPS 비활성화
uv run python nsfw_attack.py --no-lpips

# 리사이즈로 입력 크기 감소
uv run python nsfw_attack.py --resize 224
```

### 속도 향상
```bash
# 스텝 감소
uv run python nsfw_attack.py --steps 50

# LPIPS 제거
uv run python nsfw_attack.py --no-lpips

# 멀티 GPU (accelerate)
accelerate config  # GPU 설정
accelerate launch nsfw_attack.py
```

### KL 파라미터 <- 이건 헨리가 씀
```bash
# 소프트 라벨 + KL 발산 활성화 (기본값)
uv run python nsfw_attack.py \
  --label-smooth 0.1 \
  --lambda-kl 0.3 \
  --kl-temp 2.0

# 공격력 우선 (평활화 강화, KL 억제)
uv run python nsfw_attack.py \
  --label-smooth 0.05 \
  --lambda-kl 0.1 \
  --kl-temp 1.5

# 전이성 우선 (분포 보존 강화)
uv run python nsfw_attack.py \
  --label-smooth 0.15 \
  --lambda-kl 0.5 \
  --kl-temp 3.0
```

## 개발 및 기여

### 코드 포맷팅 및 검사
```bash
# 포맷팅
uv run ruff format .

# 린트 확인
uv run ruff check .

# 타입 체크
uv run pyright
```

### 테스트 실행
```bash
# 기본 테스트
uv run pytest

# 커버리지 포함
uv run pytest --cov=nsfw_attack
```

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 참고사항

이 프로젝트는 **연구 목적**으로만 사용되어야 합니다. 
악의적인 목적으로 사용하는 것은 법적 책임을 질 수 있습니다.