# HenryHazarder 설명

input 폴더의 이미지를 Transformers 기반 여러 로컬 검열 모델로 앙상블 최적화하는 연구용 프로젝트입니다

## 만든 동기

![Claude Sonnet 4.5](https://img.shields.io/badge/Claude_Sonnet_4.5-D97757?logo=anthropic&logoColor=white)
![Claude Haiku 4.5](https://img.shields.io/badge/Claude_Haiku_4.5-D97757?logo=anthropic&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-4285F4?logo=google&logoColor=white)
![GitHub Copilot](https://img.shields.io/badge/GitHub_Copilot-000000?logo=githubcopilot&logoColor=white)

먼저, 만든 이유부터 설명하자면 **Glaze**, **Nightshade** 등등같은 적대적 필터의 목적부터가 학습 데이터 포이즈닝입니다.

네, 추론 단계에선 **정상적으로 인식해서** 문제입니다,  하지만 우리 인간들이 기대한 건 **LLM도 못 알아볼 만큼 왜곡하는 것이였습니다**.

제가 거기서 더 효과적이고 빠른 방법을 생각해봤는데 **일부러 검열 모델이 검열해야 할 흉물로 인식시켜서** 학습 데이터에 못 들어가게 하는 방법입니다. 

> 그래서 더 강화시킬 수 있는 방법을 제미나이에게 여러 번 물어보며 몇 개를 뽑고 그 다음 제가 떠올린 방법을 초기 코드를 Claude Sonnet 4.5가 짜고 세션 한도 때문에 깃허브 코파일럿으로 갔는데 클로드 하이쿠 4.5가 기본값이라 그대로 썼는데 생각보다 성능이 좋더라고요. (그 후엔 소넷 4.5로 갔지만)

## 셋팅과 실행

```
# 저장소 복제
https://github.com/Henry7611913083/HenryHazarder
cd HenryHazarder

# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 만약 윈도우즈 파워셸 쓰신다면:
irm https://astral.sh/uv/install.ps1 | iex

# venv 세팅
uv sync

# 만약 cuda나 rocm 있으시면:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/<cu126 or rocm7.2>

# 입력 폴더 만들고 여기에 이미지를 넣으세요
mkdir input
cp yourimage.png input/

# 실행
uv run python nsfw_attack.py
uv run python nsfw_attack.py --steps 200 --eps 0.05 --robust
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
| `--no-lpips` | `False` | LPIPS 비활성화. 속도++ 지각적 품질-- |
| `--resize` | `None` | 처리 전 이미지를 정사각형으로 리사이즈 (예: `224`) |
| `--multi-scale` | `False` | 0.5×, 0.75×, 1.0× 스케일에서 평가해 스케일 변환 내성 부여 |
| `--use-jpeg` | `False` | quality 95/85/75로 JPEG 압축을 시뮬레이션해 압축 내성 부여 |
| `--use-transforms` | `False` | 회전·스케일·이동 변환을 적용해 기하 변환 내성 부여 |
| `--preserve-hf` | `False` | DCT 고주파 성분을 증폭해 다운샘플링 후에도 perturbation 생존율++ (scipy 필요) |
| `--robust` | `False` | 위 4가지 Robustness 옵션을 한 번에 활성화 |