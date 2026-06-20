# HenryHazarder 설명

input 폴더의 이미지를 Transformers 기반 여러 로컬 검열 모델로 앙상블 최적화하는 연구용 프로젝트입니다

## 만든 동기

![Claude Sonnet 4.5](https://img.shields.io/badge/Claude_Sonnet_4.5-D97757?logo=anthropic&logoColor=white)
![Claude Haiku 4.5](https://img.shields.io/badge/Claude_Haiku_4.5-D97757?logo=anthropic&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/GitHub_Copilot-000000?logo=githubcopilot&logoColor=white)

먼저, 만든 이유부터 설명하자면 **Glaze**, **Nightshade** 등등같은 적대적 필터의 목적부터가 학습 데이터 포이즈닝 때문에입니다.

네, 추론 단계에선 **정상적으로 인식해서** 문제입니다,  하지만 우리 인간들이 기대한 건 **LLM도 못 알아볼 만큼 왜곡하는 것이였습니다**.

제가 거기서 더 효과적이고 빠른 방법을 생각해봤는데 **일부러 검열 모델이 검열해야 할 흉물로 인식시켜서** 학습 데이터에 못 들어가게 하는 방법입니다.
만약 수집되었다라도 노이즈가 학습 중 교란을 

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
# CPU 전용
uv sync --extra cpu

# NVIDIA CUDA 12.4
uv sync --extra cu124

# AMD ROCm 6.0+
uv sync --extra rocm6

# Apple Silicon (MPS)
# 기본 torch에서 자동 지원되므로 --extra 불필요, 그냥 uv sync
```

## 셋팅과 실행

```bash
# 입력 폴더 생성
mkdir input
cp yourimage.png input/

# 기본 실행
uv run python classifier_attack.py

# GPU 명시적 지정 (CUDA:0)
CUDA_VISIBLE_DEVICES=0 uv run python classifier_attack.py

# accelerate 사용 (분산 학습/멀티 GPU)
accelerate launch classifier_attack.py

# 고급 옵션
uv run python classifier_attack.py \
  --input ./input \
  --output ./output \
  --steps 200 \
  --eps 0.05 \
  --robust \
  --model Falconsai/nsfw_image_detection \
  --model another-model-id

  # 알고리즘/옵티마이저 지정
uv run python classifier_attack.py \
  --input ./input \
  --output ./output \
  --steps 150 \
  --eps 0.04 \
  --robust \
  -A vmifgsm \
  --model Falconsai/nsfw_image_detection \
  --model another-model-id
```

## 파라미터별 특성

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `--input` | `./input` | 입력 이미지 디렉토리 |
| `--output` | `./output` | 출력 이미지 디렉토리 |
| `-S` | `Falconsai/nsfw_image_detection:nsfw` | `MODEL_ID:TARGET_CLASS` 형식으로 모델과 타겟 클래스를 함께 지정. 여러 번 지정하면 앙상블로 동작. TARGET_CLASS는 라벨 부분 문자열(`nsfw`, `explicit` 등) 또는 정수 인덱스(`0`, `1` 등) 
| `--model` | *(deprecated)* | ~~HuggingFace 모델 ID~~ `-S` 사용 권장. 하위 호환용으로 유지되며 타겟 클래스는 `nsfw` 라벨 자동 탐색 |
| `--eps` | `0.03` | 픽셀당 최대 변화량 (≈ 7.6/255). 높을수록 공격력++ 시각적 변화++ |
| `--steps` | `100` | PGD 반복 횟수. 높을수록 수렴도++, 시간++ |
| `--lr` | `자동` | 학습률. 미지정 시 optimizer/algo 조합에 따라 자동 결정 (`adam`/`sgd-nesterov`: `0.005`, `sign`/`mifgsm`/`nifgsm`/`vmifgsm`: `eps/steps`) |
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
| `-A`, `--algo` | `vanilla` | delta 업데이트 알고리즘. `vanilla`(옵티마이저 직접 선택) / `mifgsm` / `nifgsm` / `vmifgsm`. vanilla가 아니면 `--optimizer`는 무시되고 sign 기반 업데이트로 동작 |
| `--optimizer` | `torch.Adam` | `--algo vanilla`일 때 쓸 옵티마이저. `torch.<OptimizerClassName>`(torch.optim 내 임의의 클래스, 예: `torch.Adam`, `torch.RAdam`, `torch.SGD`) 또는 `vanilla.<name>`(직접 구현: `vanilla.sign`, `vanilla.lion`) 형식. 존재하지 않는 클래스명이면 영어 에러로 즉시 중단 |
| `--momentum-mu` | `1.0` | `mifgsm`/`nifgsm`/`vmifgsm` 모멘텀 계수 |
| `--vmi-n` | `3` | `vmifgsm` 주변 샘플링 개수. 클수록 분산 보정이 정확해지지만 스텝당 forward/backward가 `vmi-n`배 늘어남 |
| `--vmi-beta` | `1.5` | `vmifgsm` 주변 샘플링 반경 = `vmi-beta × eps` |
| `--reg-weight` | `0.25` | 공격 신호(`grad_cls`) 대비 LPIPS/L2 정규화 그래디언트의 최대 비중 상한. 낮을수록 공격력++ 시각 보존--, 높을수록 반대 |
| `--optimizer-kwargs` | `""` | `torch.<...>` 옵티마이저에 전달할 추가 인자. `"key=value,key2=value2"` 형식 (예: `"momentum=0.9,nesterov=True"`). `vanilla.*`에는 무시됨 |
| `--lion-beta1` | `0.9` | `vanilla.lion`의 빠른(업데이트용) 모멘텀 계수 |
| `--lion-beta2` | `0.99` | `vanilla.lion`의 느린(누적용) 모멘텀 계수 |

## 옵티마이저 / 알고리즘 특성

### 옵티마이저 (`--optimizer`, `-A vanilla`일 때만 적용)

| 값 | 원리 | 특징 |
|---|---|---|
| `torch.Adam` | 1·2차 모멘트 추정 기반 적응적 학습률 | 수렴 빠름. 픽셀별 그래디언트 분산이 달라 L∞ eps 예산을 고르게 못 쓰는 경향 있음 |
| `torch.SGD` (+`--optimizer-kwargs "momentum=0.9,nesterov=True"`) | 모멘텀 + 룩어헤드 | Adam보다 그래디언트 노이즈에 안정적 |
| `torch.<기타>` | torch.optim 클래스를 그대로 동적 호출 | `RAdam`, `NAdam`, `Adamax` 등 자유롭게 시도 가능. `LBFGS`처럼 closure 기반인 옵티마이저는 1-step 루프와 호환되지 않아 에러 |
| `vanilla.sign` | `delta -= lr * grad.sign()` | PGD 논문 표준 업데이트. lr은 보통 `eps/steps`로 자동 설정 |
| `vanilla.lion` | 빠른/느린 두 모멘텀을 보간한 뒤 그 결과에 sign 적용 | raw sign보다 잡음에 덜 민감. 부호가 매 스텝 들쭉날쭉 뒤집히는 문제에 직접 대응 |

### 알고리즘 (`-A`)

| 값 | 원리 | 특징 |
|---|---|---|
| `vanilla` | 매 스텝 그래디언트를 그대로 사용 (위 옵티마이저 중 선택) | 화이트박스 성능은 좋지만 서로게이트 모델에 과적합되기 쉬움 |
| `mifgsm` | 그래디언트를 L1 정규화 후 모멘텀에 누적, sign으로 업데이트 | 지역 최적점에 덜 갇힘. 전이성이 vanilla sign보다 일관되게 좋음 |
| `nifgsm` | mifgsm + Nesterov 룩어헤드 (`delta + mu*lr*momentum` 위치에서 그래디언트 평가) | mifgsm보다 한 발 앞서 보고 보정. 스텝당 forward 위치가 한 번 더 옮겨짐(추가 비용 없음) |
| `vmifgsm` | mifgsm + 주변 `vmi-n`개 지점의 그래디언트 평균으로 분산 보정항 추가 | 가장 안정적인 전이성. 스텝당 forward/backward가 `vmi-n+1`배로 늘어 가장 느림 |
| `soft-mifgsm` | mifgsm과 동일하되 `sign()` 대신 `tanh(momentum/soft-temp)` 사용 | 0 근처에서는 부드럽게, 확신이 높은 곳에서는 거의 sign처럼 포화. `--soft-temp`가 작을수록 잡음에 둔감 |

> `--reg-weight`는 `grad_cls`와 `grad_reg`(lpips+l2)가 매 스텝 부호 다툼을 벌이지 않도록
> reg 그래디언트 크기에 상한을 두는 파라미터입니다. 기본값 `0.25`면 정규화 신호가
> 공격 신호의 25%를 넘지 못하도록 깎입미다다. lpips/l2 손실이 역주행하거나 발진하면
> 이 값을 낮춰보시고(예: `0.1`), 시각적 손상이 너무 크면 올려보세여(예: `0.4`~`0.5`).

**선택 가이드**: 빠른 반복/디버깅엔 `adam`, 단일 타깃 모델 최대 공격력엔 `sign`, 처음 보는 모델로의 전이성이 중요하면 `mifgsm`→`vmifgsm` 순으로 시도해보시는 걸 권장.

(여기서부턴 클로드가 맘대로 쓴거)
## 성능 최적화

### 메모리 부족 시
```bash
# 배치 크기 축소 (단일 이미지 처리)
# --batch-size 1 (아직 미구현, 기본 1개씩)

# LPIPS 비활성화
uv run python classifier_attack.py --no-lpips

# 리사이즈로 입력 크기 감소
uv run python classifier_attack.py --resize 224
```

### 속도 향상
```bash
# 스텝 감소
uv run python classifier_attack.py --steps 50

# LPIPS 제거
uv run python classifier_attack.py --no-lpips

# 멀티 GPU (accelerate)
accelerate config  # GPU 설정
accelerate launch classifier_attack.py
```

### KL 파라미터 <- 이건 헨리가 씀
```bash
# 소프트 라벨 + KL 발산 활성화 (기본값)
uv run python classifier_attack.py \
  --label-smooth 0.1 \
  --lambda-kl 0.3 \
  --kl-temp 2.0

# 공격력 우선 (평활화 강화, KL 억제)
uv run python classifier_attack.py \
  --label-smooth 0.05 \
  --lambda-kl 0.1 \
  --kl-temp 1.5

# 전이성 우선 (분포 보존 강화)
uv run python classifier_attack.py \
  --label-smooth 0.15 \
  --lambda-kl 0.5 \
  --kl-temp 3.0
```

### 옵티마이저 / 알고리즘 선택
`--optimizer`는 `torch.<클래스명>` 또는 `vanilla.<이름>` 두 네임스페이스 중 하나로 지정합니다.
`torch.*`는 torch.optim에 있는 어떤 옵티마이저든 동적으로 찾아 쓰며, 없는 클래스명을 넣으면
영어 에러 메시지와 함께 즉시 중단됩니다. `vanilla.*`는 이 프로젝트가 직접 구현한 업데이트
규칙(`sign`, `lion`)입니다.

```bash
# 기본값: vanilla + torch.Adam (수렴 빠름, 전이성은 보통)
uv run python classifier_attack.py -S Falconsai/nsfw_image_detection:nsfw

# 토치 내장 옵티마이저를 자유롭게 시도
uv run python classifier_attack.py -S ... --optimizer torch.RAdam
uv run python classifier_attack.py -S ... --optimizer torch.SGD --optimizer-kwargs "momentum=0.9,nesterov=True"

# 직접 구현한 sign 업데이트 (PGD 논문 표준, lr 자동 = eps/steps)
uv run python classifier_attack.py -S ... --optimizer vanilla.sign

# 직접 구현한 lion (모멘텀 보간 후 sign — 잡음에 raw sign보다 둔감)
uv run python classifier_attack.py -S ... --optimizer vanilla.lion --lion-beta1 0.9 --lion-beta2 0.99

# 전이성 우선 (모멘텀+sign 알고리즘, --optimizer는 자동 무시되고 vanilla.sign 기반으로 강제 동작)
uv run python classifier_attack.py -S ... -A mifgsm
uv run python classifier_attack.py -S ... -A nifgsm

# 잡음에 더 둔감한 부드러운 버전 (tanh 기반)
uv run python classifier_attack.py -S ... -A soft-mifgsm --soft-temp 0.3

# 전이성 최우선 (분산 보정 포함, 비용 큼 — vmi-n배 forward/backward 추가)
uv run python classifier_attack.py -S ... -A vmifgsm --vmi-n 3 --vmi-beta 1.5

# 존재하지 않는 클래스명 → 영어 에러로 즉시 중단
uv run python classifier_attack.py -S ... --optimizer torch.Adamm
# → ValueError: 'torch.Adamm' is not a valid PyTorch optimizer — no class named 'Adamm' found in torch.optim ...
```

> `-A`가 `vanilla`가 아니면(`mifgsm`/`nifgsm`/`vmifgsm`/`soft-mifgsm`) `--optimizer`는 무시되고
> 내부적으로 `vanilla.sign` 기반 모멘텀-부호 업데이트로 강제 동작합니다.

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
uv run pytest --cov=classifier_attack
```

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 참고사항

이 프로젝트는 **연구 목적**으로만 사용되어야 합니다. 
악의적인 목적으로 사용하는 것은 법적 책임을 질 수 있습니다.