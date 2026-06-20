"""
classifier_attack.py
--------------
Gradient-based adversarial perturbation that maximizes a user-specified target
class score of any HuggingFace image classification model, while minimizing
perceptual distortion.

Method: PGD (Projected Gradient Descent) + LPIPS perceptual regularization
  L_total = L_cls(soft) + lambda_kl * L_KL + lambda_lpips * L_LPIPS + mu_l2 * L_L2
  Perturbation is projected onto the Linf ball [-eps, +eps] after each step.

Model/target specification:
  -S MODEL_ID:TARGET_CLASS   (can be repeated for ensemble)
  TARGET_CLASS can be:
    - a label substring  (e.g. "nsfw", "explicit", "safe")
    - an integer index   (e.g. 0, 1, 2)
    - omitted            (falls back to searching for "nsfw" label)

Robustness enhancements:
  - Multi-scale evaluation (--multi-scale)
  - JPEG compression robustness (--use-jpeg)
  - Geometric transformations (--use-transforms)
  - DCT high-frequency preservation (--preserve-hf)
  - High-quality resampling (Lanczos)

Requirements:
  uv pip install torch torchvision transformers pillow lpips tqdm scipy numpy accelerate

Usage:
  python classifier_attack.py \\
      -S Falconsai/nsfw_image_detection:nsfw \\
      -S AdamCodd/vit-base-nsfw-detector:nsfw \\
      [--eps 0.03] [--steps 100] [--lr 0.005] \\
      [--lambda-lpips 2.0] [--mu-l2 0.5] \\
      [--multi-scale] [--use-jpeg] [--use-transforms] \\
      [--preserve-hf] [--input ./input] [--output ./output]

  # Legacy --model flag is still accepted (target defaults to "nsfw" search):
  python classifier_attack.py --model Falconsai/nsfw_image_detection
"""

import ast
import argparse
import logging
import sys
import time
from pathlib import Path

import random
from enum import Enum
from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from vision_classifier_wrapper import VisionClassifierWrapper, build_wrapper

# ── Optional dependencies ──────────────────────────────────────────────────────

try:
    import lpips as _lpips_mod
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False

try:
    from scipy.fftpack import dct, idct
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("classifier_attack")

if not HAS_LPIPS:
    log.warning("lpips not found — falling back to L2 regularization only.")
    log.warning("  install with: uv pip install lpips")

if not HAS_SCIPY:
    log.warning("scipy not found — DCT high-frequency preservation disabled.")
    log.warning("  install with: uv pip install scipy")


class Optimizer(str, Enum):
    ADAM    = "adam"
    SGD_NAG = "sgd-nesterov"
    SIGN    = "sign"

class Algo(str, Enum):
    VANILLA      = "vanilla"
    MIFGSM       = "mifgsm"
    NIFGSM       = "nifgsm"
    VMIFGSM      = "vmifgsm"
    SOFT_MIFGSM  = "soft-mifgsm"

class DeltaUpdater:
    """
    delta 업데이트 규칙.

    --optimizer는 두 네임스페이스 중 하나로 지정한다.
      torch.<OptimizerClassName>  : torch.optim에 있는 임의의 옵티마이저를 동적으로 resolve
                                     (예: torch.Adam, torch.SGD, torch.RAdam, torch.NAdam ...)
      vanilla.<name>              : 이 프로젝트가 직접 구현한 업데이트 규칙 ('sign', 'lion')

    --algo가 vanilla가 아니면(mifgsm/nifgsm/vmifgsm/soft-mifgsm) --optimizer는 무시되고
    내부적으로 vanilla.sign 기반의 모멘텀-부호 업데이트로 강제 동작한다.
    """
    MOMENTUM_ALGOS = {Algo.MIFGSM, Algo.NIFGSM, Algo.VMIFGSM, Algo.SOFT_MIFGSM}
    VANILLA_NAMES  = {"sign", "lion"}
    UNSUPPORTED_TORCH = {"LBFGS"}  # closure 기반이라 1-step 루프와 호환 불가

    def __init__(
        self,
        optimizer: str,
        algo: str,
        delta: torch.Tensor,
        lr: float,
        mu: float = 1.0,
        soft_temp: float = 0.5,
        lion_beta1: float = 0.9,
        lion_beta2: float = 0.99,
        optimizer_kwargs: dict | None = None,
    ):
        self.algo = algo
        self.delta = delta
        self.lr = lr
        self.mu = mu
        self.soft_temp = soft_temp
        self.lion_beta1 = lion_beta1
        self.lion_beta2 = lion_beta2
        self.momentum  = torch.zeros_like(delta)   # mifgsm 계열 공용 모멘텀
        self.lion_slow = torch.zeros_like(delta)   # lion 전용 느린(2차) 모멘텀

        # algo가 vanilla가 아니면 optimizer 선택 자체가 무의미하므로 강제 통일
        resolved = optimizer if algo == Algo.VANILLA else "vanilla.sign"
        self.namespace, self.opt_name = self._parse_namespace(resolved)

        self.torch_opt = None
        if self.namespace == "torch":
            self.torch_opt = self._build_torch_optimizer(
                self.opt_name, delta, lr, optimizer_kwargs or {}
            )
        elif self.opt_name not in self.VANILLA_NAMES:
            raise ValueError(
                f"Unknown vanilla optimizer 'vanilla.{self.opt_name}'. "
                f"Available: {sorted(self.VANILLA_NAMES)}."
            )

    @staticmethod
    def _parse_namespace(optimizer: str) -> tuple[str, str]:
        if "." not in optimizer:
            raise ValueError(
                f"Invalid --optimizer value '{optimizer}'. "
                f"Must be namespaced as 'torch.<OptimizerClassName>' (e.g. 'torch.Adam') "
                f"or 'vanilla.<name>' (e.g. 'vanilla.sign', 'vanilla.lion')."
            )
        namespace, name = optimizer.split(".", 1)
        if namespace not in ("torch", "vanilla"):
            raise ValueError(
                f"Invalid --optimizer namespace '{namespace}' in '{optimizer}'. Must be 'torch' or 'vanilla'."
            )
        return namespace, name

    @classmethod
    def _build_torch_optimizer(cls, cls_name: str, delta: torch.Tensor, lr: float, extra_kwargs: dict):
        if cls_name in cls.UNSUPPORTED_TORCH:
            raise ValueError(
                f"'torch.{cls_name}' requires a closure-based step() call and is not supported "
                f"by this single-step PGD loop. Choose a first-order optimizer (e.g. 'torch.Adam', 'torch.SGD')."
            )
        opt_cls = getattr(torch.optim, cls_name, None)
        if opt_cls is None or not (isinstance(opt_cls, type) and issubclass(opt_cls, torch.optim.Optimizer)):
            raise ValueError(
                f"'torch.{cls_name}' is not a valid PyTorch optimizer — no class named '{cls_name}' "
                f"found in torch.optim (installed PyTorch version: {torch.__version__}). "
                f"Class names are case-sensitive, e.g. 'torch.Adam', not 'torch.adam'."
            )
        try:
            return opt_cls([delta], lr=lr, **extra_kwargs)
        except TypeError as e:
            raise ValueError(
                f"Failed to construct torch.optim.{cls_name}(lr={lr}, **{extra_kwargs}): {e}. "
                f"This optimizer likely needs extra arguments — pass them via "
                f"--optimizer-kwargs \"key=value,key2=value2\" (e.g. \"momentum=0.9,nesterov=True\")."
            ) from e

    def lookahead_offset(self) -> torch.Tensor:
        if self.algo == Algo.NIFGSM:
            return self.mu * self.lr * self.momentum
        return torch.zeros_like(self.delta)

    def step(self, grad: torch.Tensor) -> None:
        if self.torch_opt is not None:
            self.delta.grad = grad
            self.torch_opt.step()
            self.torch_opt.zero_grad()
            return

        with torch.no_grad():
            if self.algo in self.MOMENTUM_ALGOS:
                g = grad / (grad.abs().mean() + 1e-12)
                self.momentum = self.mu * self.momentum + g
                if self.algo == Algo.SOFT_MIFGSM:
                    self.delta -= self.lr * torch.tanh(self.momentum / self.soft_temp)
                else:
                    self.delta -= self.lr * self.momentum.sign()

            elif self.opt_name == "lion":
                update = torch.sign(
                    self.lion_beta1 * self.lion_slow + (1 - self.lion_beta1) * grad
                )
                self.delta -= self.lr * update
                self.lion_slow = self.lion_beta2 * self.lion_slow + (1 - self.lion_beta2) * grad

            else:  # vanilla.sign
                self.delta -= self.lr * grad.sign()

def parse_kv_string(s: str) -> dict:
    """'key=value,key2=value2' 형태 문자열을 dict로 파싱. 값은 가능하면 Python 리터럴로 평가."""
    if not s:
        return {}
    out: dict = {}
    for pair in s.split(","):
        if "=" not in pair:
            raise ValueError(f"Invalid kwargs segment '{pair}' — expected key=value.")
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v  # 리터럴로 안 풀리면 문자열 그대로 사용
    return out

# ── Constants ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ── Basic image utilities ──────────────────────────────────────────────────────

def load_image(path: Path, size: tuple[int, int] | None = None) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return img


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a CHW float [0, 1] tensor as PNG."""
    arr = tensor.detach().cpu().clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(path)


def tensor_from_pil(img: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a CHW float [0, 1] tensor."""
    return transforms.ToTensor()(img)


# ── High-quality resampling ────────────────────────────────────────────────────

def high_quality_resize(tensor: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """
    Lanczos-based high-quality resampling to minimise information loss.
    tensor: CHW [0, 1]
    """
    pil_img = transforms.ToPILImage()(tensor.cpu())
    resized_pil = pil_img.resize(target_size, Image.LANCZOS)
    return transforms.ToTensor()(resized_pil).to(tensor.device)


def adaptive_downsample(
    tensor: torch.Tensor,
    target_size: tuple[int, int],
    num_steps: int = 3,
) -> torch.Tensor:
    """
    Gradual multi-step Lanczos downsampling to reduce cumulative aliasing error.
    tensor: CHW [0, 1]
    """
    current = tensor.clone()
    h_start, w_start = tensor.shape[1], tensor.shape[2]
    h_target, w_target = target_size

    for step in range(1, num_steps + 1):
        progress = step / num_steps
        h_inter = int(h_start + (h_target - h_start) * progress)
        w_inter = int(w_start + (w_target - w_start) * progress)

        pil_img = transforms.ToPILImage()(current.cpu())
        current = transforms.ToTensor()(
            pil_img.resize((w_inter, h_inter), Image.LANCZOS)
        ).to(tensor.device)

    return current


# ── DCT high-frequency preservation ───────────────────────────────────────────

def preserve_high_frequency_dct(
    delta: torch.Tensor,
    alpha: float = 0.5,
) -> torch.Tensor:
    """
    Amplify high-frequency DCT components in the perturbation delta so they
    survive subsequent downsampling / upsampling operations.

    delta: CHW tensor
    alpha: amplification factor (0.5 → 50 % boost to high-freq quadrant)
    """
    if not HAS_SCIPY:
        log.warning("scipy unavailable — skipping DCT high-frequency preservation")
        return delta

    delta_np = delta.cpu().detach().numpy()
    delta_preserved = np.zeros_like(delta_np)

    for c in range(delta_np.shape[0]):
        # 2-D DCT transform
        delta_freq = dct(dct(delta_np[c], axis=0, norm="ortho"), axis=1, norm="ortho")

        # Amplify the upper-right (high-frequency) quadrant
        h, w = delta_freq.shape
        delta_freq[h // 4 :, w // 4 :] *= (1 + alpha)

        # Inverse DCT back to spatial domain
        delta_preserved[c] = idct(
            idct(delta_freq, axis=0, norm="ortho"), axis=1, norm="ortho"
        )

    return torch.from_numpy(delta_preserved).float().to(delta.device)


# ── JPEG compression robustness ────────────────────────────────────────────────

def apply_jpeg_compression(
    tensor: torch.Tensor,
    quality: int = 85,
) -> torch.Tensor:
    """
    Simulate JPEG compression in memory to test perturbation robustness.
    tensor: CHW [0, 1]
    """
    import io

    pil_img = transforms.ToPILImage()(tensor.cpu())
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return transforms.ToTensor()(Image.open(buffer).convert("RGB")).to(tensor.device)


# ── Geometric transformation robustness ───────────────────────────────────────

def apply_geometric_transform(
    tensor: torch.Tensor,
    angle: float = 0,
    scale: float = 1.0,
    translate: tuple[float, float] = (0, 0),
) -> torch.Tensor:
    """
    Apply an affine transformation (rotation, scaling, translation).
    tensor: CHW [0, 1]
    """
    return transforms.functional.affine(
        tensor,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        interpolation=transforms.InterpolationMode.BILINEAR,  # LANCZOS는 스텝마다 쓰기엔 비용이 큼
    )

# ── EOT-style robustness sampling ──────────────────────────────────────────────

def eot_view(
    adv_native: torch.Tensor,
    multi_scale: bool,
    use_jpeg: bool,
    use_transforms: bool,
) -> torch.Tensor:
    """매 스텝마다 무작위 견고성 변환 하나를 적용한 뷰를 반환한다."""
    choices = ["identity"]
    if multi_scale:    choices.append("scale")
    if use_transforms: choices.append("affine")
    if use_jpeg:       choices.append("jpeg")
    choice = random.choice(choices)

    if choice == "scale":
        factor = random.choice([0.5, 0.75, 1.0])
        h, w = adv_native.shape[-2:]
        small = F.interpolate(
            adv_native.unsqueeze(0), scale_factor=factor,
            mode="bilinear", align_corners=False,
        )
        return F.interpolate(
            small, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)

    if choice == "affine":
        angle = random.uniform(-8, 8)
        return apply_geometric_transform(adv_native, angle=angle)

    if choice == "jpeg":
        with torch.no_grad():
            compressed = apply_jpeg_compression(adv_native, quality=random.choice([95, 85, 75]))
        return adv_native + (compressed - adv_native).detach()  # straight-through

    return adv_native


# ── Translation-invariant gradient smoothing (TIM) ─────────────────────────────

def ti_smooth(grad: torch.Tensor, kernel_size: int = 7, sigma: float = 3.0) -> torch.Tensor:
    """평행이동 불변성을 흉내내기 위해 그래디언트를 가우시안 커널로 컨볼브."""
    coords = torch.arange(kernel_size, dtype=torch.float32, device=grad.device) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k1d = g / g.sum()
    kernel = (k1d[:, None] @ k1d[None, :]).expand(grad.shape[0], 1, kernel_size, kernel_size)
    return F.conv2d(
        grad.unsqueeze(0), kernel, padding=kernel_size // 2, groups=grad.shape[0]
    ).squeeze(0)


# ── 모델별 정규화 앙상블 그래디언트 ──────────────────────────────────────────────

def ensemble_grad(
    wrappers: list[VisionClassifierWrapper],
    view: torch.Tensor,
    orig: torch.Tensor,
    delta: torch.Tensor,
    label_smooth: float,
    lambda_kl: float,
    kl_temp: float,
    converged_thresh: float = 0.01,
) -> tuple[torch.Tensor, float, float, list[float]]:
    """
    모델별 그래디언트를 norm이 아니라 '얼마나 안 풀렸는지'로 가중해 합산한다.
    이미 converged_thresh 아래로 떨어진 모델은 잡음 증폭을 막기 위해
    그래디언트를 거의 무시(낮은 가중치)한다.
    """
    raw_grads, weights, target_probs = [], [], []
    loss_cls_sum, loss_kl_sum = 0.0, 0.0

    for w in wrappers:
        delta.grad = None
        x_in   = w.preprocess(view)
        logits = w.logits(x_in)
        p = F.softmax(logits, dim=-1)[0, w.target_idx].item()
        target_probs.append(p)

        l_cls = w.cls_loss(logits, label_smooth)
        l_kl  = w.kl_loss(logits, orig, kl_temp)
        (l_cls + lambda_kl * l_kl).backward(retain_graph=True)

        raw_grads.append(delta.grad.clone())
        # 이미 충분히 떨어진(converged) 모델은 가중치를 낮춰 잡음 증폭을 방지
        remaining = 1.0 - p
        weights.append(remaining if remaining > converged_thresh else converged_thresh)
        loss_cls_sum += l_cls.item()
        loss_kl_sum  += l_kl.item()

    weights_t = torch.tensor(weights, device=delta.device)
    weights_t = weights_t / (weights_t.sum() + 1e-8)

    # 모델별로 unit-norm 방향만 취하고, 가중치로 크기를 결정
    directions = [g / (g.norm() + 1e-8) for g in raw_grads]
    grad = sum(w_i * d for w_i, d in zip(weights_t, directions))

    n = len(wrappers)
    return grad, loss_cls_sum / n, loss_kl_sum / n, target_probs

# ── Spec parsing ───────────────────────────────────────────────────────────────

def parse_spec(spec: str) -> tuple[str, str | int | None]:
    """
    Parse a '-S MODEL_ID:TARGET_CLASS' token into (model_id, target).

    TARGET_CLASS rules:
      - integer string  → converted to int (used as class index directly)
      - other string    → used as a label substring for fuzzy matching
      - absent          → None  (falls back to searching for "nsfw" in id2label)

    Uses rfind(':') so that HuggingFace org/model paths with '/' are safe.

    Examples:
      "Falconsai/nsfw_image_detection:nsfw"  → ("Falconsai/nsfw_image_detection", "nsfw")
      "myorg/model:1"                        → ("myorg/model", 1)
      "myorg/model"                          → ("myorg/model", None)
    """
    colon = spec.rfind(":")
    if colon == -1:
        return spec, None

    model_id = spec[:colon]
    target_raw = spec[colon + 1:]

    # Empty string after colon → treat as omitted
    if not target_raw:
        return model_id, None

    try:
        return model_id, int(target_raw)
    except ValueError:
        return model_id, target_raw


def resolve_target_idx(m, target: str | int | None) -> int:
    """
    Resolve the target class index for a given model.

    - int   → validated against id2label and returned as-is
    - str   → case-insensitive substring search in id2label values
    - None  → falls back to searching for the substring "nsfw"

    Raises ValueError for an out-of-range integer index.
    Warns and returns 1 when no matching label is found.
    """
    id2label: dict = m.config.id2label

    if isinstance(target, int):
        if target not in id2label:
            raise ValueError(
                f"Target index {target} is out of range for id2label: {id2label}"
            )
        return target

    search = (target or "nsfw").lower()
    for idx, label in id2label.items():
        if search in label.lower():
            return int(idx)

    log.warning(
        f"Label '{search}' not found in {id2label} — falling back to index 1"
    )
    return 1


# ── Score computation ──────────────────────────────────────────────────────────

def compute_scores(
    model,
    feature_extractor,
    tensor: torch.Tensor,
    id2label: dict,
    device: torch.device,
) -> dict[str, float]:
    """Compute per-class probability scores for a single image tensor."""
    pil = transforms.ToPILImage()(tensor.cpu())
    inputs = feature_extractor(images=pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0]
    return {id2label[i]: probs[i].item() for i in range(len(probs))}


def compute_scores_single_class(
    model,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    input_size: int,
    tensor: torch.Tensor,
    target_class_idx: int,
    device: torch.device,
) -> float:
    """
    Compute a single target-class probability using the same preprocessing
    pipeline as the PGD optimisation loop (bilinear resize → normalise).
    """
    x = tensor.to(device)
    with torch.no_grad():
        x_resized = F.interpolate(
            x.unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        )
        pixel_values = (x_resized - mean_t) / std_t
        logits = model(pixel_values=pixel_values).logits
        probs = F.softmax(logits, dim=-1)[0]
    return probs[target_class_idx].item()


# ── PGD attack ─────────────────────────────────────────────────────────────────

def pgd_attack(
    wrappers: list[VisionClassifierWrapper],
    orig_tensor: torch.Tensor,
    device: torch.device,
    image_name: str,
    eps: float = 0.03,
    steps: int = 100,
    optimizer: str = "torch.Adam",
    algo: str = "vanilla",
    lr: float = 0.005,
    momentum_mu: float = 1.0,
    soft_temp: float = 0.5,
    lion_beta1: float = 0.9,
    lion_beta2: float = 0.99,
    optimizer_kwargs: dict | None = None,
    vmi_n: int = 3,
    vmi_beta: float = 1.5,
    lambda_lpips: float = 2.0,
    mu_l2: float = 0.5,
    lpips_net=None,
    accelerator=None,
    label_smooth: float = 0.1,
    lambda_kl: float = 0.3,
    kl_temp: float = 2.0,
    multi_scale: bool = False,
    use_jpeg: bool = False,
    use_transforms: bool = False,
    reg_weight: float = 0.25,
) -> torch.Tensor:
    """
    PGD 계열 공격. 모델별 정규화 그래디언트 합산 + EOT 견고성 샘플링 +
    TIM 그래디언트 스무딩 + 선택 가능한 옵티마이저/알고리즘(adam, sgd-nesterov,
    sign-pgd, mifgsm, nifgsm, vmifgsm)을 지원한다.
    """
    orig = orig_tensor.to(device)
    delta = ((torch.rand_like(orig) * 2 - 1) * eps).clamp(-eps, eps).to(device).requires_grad_(True)
    updater = DeltaUpdater(
        optimizer=optimizer, algo=algo, delta=delta, lr=lr, mu=momentum_mu,
        soft_temp=soft_temp, lion_beta1=lion_beta1, lion_beta2=lion_beta2,
        optimizer_kwargs=optimizer_kwargs,
    )

    step_w = len(str(steps))
    log_fn = accelerator.print if accelerator else log.info
    log_fn(
        f"[{image_name}] starting PGD  steps={steps}  eps={eps}  "
        f"optimizer={optimizer}  algo={algo}  lr={lr}"
    )

    t0 = time.time()
    for step in range(1, steps + 1):
        delta.grad = None

        lookahead   = updater.lookahead_offset()
        adv_native  = orig + delta + lookahead
        adv_view    = eot_view(adv_native, multi_scale, use_jpeg, use_transforms)

        grad_cls, loss_cls_avg, loss_kl_avg, target_probs = ensemble_grad(
            wrappers, adv_view, orig, delta, label_smooth, lambda_kl, kl_temp
        )

        # VMI-FGSM: 주변 점들의 그래디언트 평균 - 현재 그래디언트 = 분산 보정항
        # (vmi_n번의 추가 앙상블 forward/backward가 필요해 비용이 큽니다)
        if algo == Algo.VMIFGSM and vmi_n > 0:
            neighbor_grads = []
            for _ in range(vmi_n):
                r = (torch.rand_like(delta) * 2 - 1) * (vmi_beta * eps)
                neighbor_view = eot_view(orig + delta + lookahead + r, multi_scale, use_jpeg, use_transforms)
                g_n, _, _, _ = ensemble_grad(
                    wrappers, neighbor_view, orig, delta, label_smooth, lambda_kl, kl_temp
                )
                neighbor_grads.append(g_n)
            variance_term = torch.stack(neighbor_grads).mean(0) - grad_cls
            grad_cls = grad_cls + variance_term

        grad_cls = ti_smooth(grad_cls)

        # 지각/L2 정규화 손실 — 변환 뷰가 아닌 실제 저장될 이미지(adv_native) 기준
        delta.grad = None
        loss_l2 = (delta ** 2).mean()
        if lpips_net is not None:
            loss_lpips = lpips_net(orig.unsqueeze(0), adv_native.unsqueeze(0)).mean()
            reg_loss = lambda_lpips * loss_lpips + mu_l2 * loss_l2
        else:
            loss_lpips = torch.tensor(0.0)
            reg_loss = mu_l2 * loss_l2
        reg_loss.backward()
        grad_reg = delta.grad.clone() if delta.grad is not None else torch.zeros_like(delta)

        # grad_cls와 매 스텝 부호 다툼을 벌이지 않도록, reg_weight 비율로 상한만 둔다.
        # raw grad_reg가 자연히 작으면(이미 충분히 만족됐으면) 그대로 작게 둔다.
        cls_norm = grad_cls.norm() + 1e-8
        reg_norm = grad_reg.norm() + 1e-8
        scale = torch.clamp(reg_weight * cls_norm / reg_norm, max=1.0)
        grad_reg = grad_reg * scale

        updater.step(grad_cls + grad_reg)

        with torch.no_grad():
            delta.clamp_(-eps, eps)

        target_scores_str = "  ".join(f"{p:.4f}" for p in target_probs)
        log_fn(
            f"[{image_name}] step [{step:0{step_w}d}/{steps}]  "
            f"loss_cls={loss_cls_avg:.6f}  "
            f"loss_kl={loss_kl_avg:.6f}  "
            f"loss_lpips={loss_lpips.item():.6f}  "
            f"loss_l2={loss_l2.item():.6f}  |  "
            f"target: {target_scores_str}"
        )

    elapsed = time.time() - t0
    log_fn(f"[{image_name}] PGD complete  elapsed={elapsed:.1f}s")

    with torch.no_grad():
        result = (orig + delta).cpu().clamp(0, 1)

    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generic PGD adversarial attack against any HuggingFace image "
            "classification model. Target class is specified per-model via -S."
        )
    )

    parser.add_argument("--input",  default="./input",  help="Input image directory")
    parser.add_argument("--output", default="./output", help="Output image directory")

    # ── Primary model/target specification ────────────────────────────────────
    parser.add_argument(
        "-S",
        action="append",
        dest="specs",
        default=None,
        metavar="MODEL_ID:TARGET_CLASS",
        help=(
            "Model and target class to attack (repeatable for ensemble). "
            "TARGET_CLASS can be a label substring (e.g. 'nsfw', 'explicit') "
            "or an integer index (e.g. 0, 1). "
            "Omitting TARGET_CLASS falls back to searching for 'nsfw' in id2label. "
            "Examples: "
            "-S Falconsai/nsfw_image_detection:nsfw  "
            "-S myorg/classifier:1"
        ),
    )

    # ── Legacy --model flag (kept for backward compatibility) ─────────────────
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        metavar="MODEL_ID",
        help=(
            "HuggingFace model ID (legacy; repeatable for ensemble). "
            "Target class defaults to searching for 'nsfw' in id2label. "
            "Prefer -S MODEL_ID:TARGET_CLASS for explicit control."
        ),
    )

    # ── Optimisation hyperparameters ──────────────────────────────────────────
    parser.add_argument(
        "--eps",
        type=float,
        default=0.03,
        help="Linf epsilon — max perturbation per pixel (default 0.03 ≈ 7.6/255)",
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of PGD optimisation steps (default 100)",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Adam learning rate (default 0.005)",
    )
    parser.add_argument(
        "--lambda-lpips",
        type=float,
        default=2.0,
        help="Weight for LPIPS perceptual loss (default 2.0)",
    )
    parser.add_argument(
        "--mu-l2",
        type=float,
        default=0.5,
        help="Weight for L2 pixel regularisation (default 0.5)",
    )
    parser.add_argument(
        "--label-smooth",
        type=float,
        default=0.1,
        help="Soft-label smoothing factor alpha (default 0.1)",
    )
    parser.add_argument(
        "--lambda-kl",
        type=float,
        default=0.3,
        help="Weight for KL divergence regularisation (default 0.3)",
    )
    parser.add_argument(
        "--kl-temp",
        type=float,
        default=2.0,
        help="Temperature for KL divergence scaling (default 2.0)",
    )
    parser.add_argument(
        "--no-lpips",
        action="store_true",
        help="Disable LPIPS perceptual loss (faster, lower perceptual quality)",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Resize input images to this square size before processing (e.g. 224)",
    )

    # ── Robustness options ────────────────────────────────────────────────────
    parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale robustness (evaluate at 0.5×, 0.75×, 1.0× scales)",
    )
    parser.add_argument(
        "--use-jpeg",
        action="store_true",
        help="Enable JPEG compression robustness simulation (qualities: 95, 85, 75)",
    )
    parser.add_argument(
        "--use-transforms",
        action="store_true",
        help="Enable geometric transformation robustness (rotation, scaling, translation)",
    )
    parser.add_argument(
        "--preserve-hf",
        action="store_true",
        help="Enable DCT high-frequency preservation (requires scipy)",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Enable all robustness features (equivalent to --multi-scale --use-jpeg --use-transforms --preserve-hf)",
    )
    parser.add_argument(
        "-A", "--algo",
        choices=[a.value for a in Algo],
        default="vanilla",
        help="delta 업데이트 알고리즘. vanilla가 아니면 --optimizer는 무시됩니다 (default: vanilla)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default=None,
        help=(
            "Optimizer for --algo vanilla. Namespaced as 'torch.<OptimizerClassName>' "
            "(any class in torch.optim, e.g. 'torch.Adam', 'torch.RAdam', 'torch.SGD') "
            "or 'vanilla.<name>' (hand-rolled: 'vanilla.sign', 'vanilla.lion'). Default: torch.Adam."
        ),
    )
    parser.add_argument(
        "--optimizer-kwargs",
        type=str,
        default="",
        help='Extra kwargs for a torch.<...> optimizer as "key=value,key2=value2" '
            '(e.g. "momentum=0.9,nesterov=True"). Ignored for vanilla.* optimizers.',
    )
    parser.add_argument(
        "--lion-beta1", type=float, default=0.9,
        help="vanilla.lion fast (update) momentum coefficient (default: 0.9)",
    )
    parser.add_argument(
        "--lion-beta2", type=float, default=0.99,
        help="vanilla.lion slow (accumulation) momentum coefficient (default: 0.99)",
    )
    parser.add_argument(
        "--momentum-mu", type=float, default=1.0,
        help="mifgsm/nifgsm/vmifgsm 모멘텀 계수 (default: 1.0)",
    )
    parser.add_argument(
        "--vmi-n", type=int, default=3,
        help="vmifgsm 주변 샘플링 개수 — 클수록 정확하지만 스텝당 비용이 커집니다 (default: 3)",
    )
    parser.add_argument(
        "--vmi-beta", type=float, default=1.5,
        help="vmifgsm 주변 샘플링 반경 = vmi_beta * eps (default: 1.5)",
    )
    parser.add_argument(
        "--reg-weight", type=float, default=0.25,
        help="공격 신호 대비 lpips/l2 정규화 그래디언트의 최대 비중 상한 (default: 0.25)",
    )
    parser.add_argument(
        "--soft-temp", type=float, default=0.5,
        help="soft-mifgsm의 tanh temperature. 작을수록 부드럽고 잡음에 둔감, 클수록 sign에 가까움 (default: 0.5)",
    )

    args = parser.parse_args()

    # ── Build unified (model_id, target_raw) list ─────────────────────────────
    # Priority: -S specs > --model (legacy) > built-in default
    if args.specs is not None:
        specs_parsed: list[tuple[str, str | int | None]] = [
            parse_spec(s) for s in args.specs
        ]
    elif args.models is not None:
        # Legacy --model: target defaults to None (→ "nsfw" search)
        specs_parsed = [(m, None) for m in args.models]
    else:
        # Built-in default: original behaviour
        specs_parsed = [("Falconsai/nsfw_image_detection", None)]

    # ── Accelerator setup ─────────────────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=1,
        cpu=False,
    )

    # ── optimizer / algo / lr 확정 ─────────────────────────────────────────
    args.optimizer_kwargs = parse_kv_string(args.optimizer_kwargs)

    if args.algo == "vanilla":
        if args.optimizer is None:
            args.optimizer = "torch.Adam"
    else:
        if args.optimizer is not None:
            log.warning(
                f"-A {args.algo} 지정 시 --optimizer 값({args.optimizer})은 무시되고 "
                f"vanilla.sign 기반 모멘텀-부호 업데이트가 사용됩니다."
            )
        args.optimizer = "vanilla.sign"

    if args.lr is None:
        namespace = args.optimizer.split(".", 1)[0]
        if args.algo != "vanilla" or namespace == "vanilla":
            args.lr = args.eps / args.steps
        else:
            args.lr = 0.005
        log.info(f"--lr 미지정 → optimizer={args.optimizer} algo={args.algo} 기준 자동 설정: {args.lr:.6f}")

    # ── Expand --robust shorthand ─────────────────────────────────────────────
    if args.robust:
        args.multi_scale   = True
        args.use_jpeg      = True
        args.use_transforms = True
        args.preserve_hf   = True

    run_start = time.time()
    device = accelerator.device

    accelerator.print(f"device: {device}")
    accelerator.print(f"distributed_type: {accelerator.distributed_type}")
    accelerator.print(f"num_processes: {accelerator.num_processes}")

    # ── I/O paths ─────────────────────────────────────────────────────────────
    input_dir  = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error(f"no images found in '{input_dir}'")
        sys.exit(1)
    log.info(f"found {len(images)} image(s) in '{input_dir}'")

    # ── Load ensemble ─────────────────────────────────────────────────────────
    wrappers: list[VisionClassifierWrapper] = [
        build_wrapper(s, accelerator) for s in (args.specs or [m for m, _ in specs_parsed])
    ]
    # ── LPIPS network ─────────────────────────────────────────────────────────
    lpips_net = None
    if HAS_LPIPS and not args.no_lpips:
        log.info("loading LPIPS perceptual loss network (AlexNet backbone)")
        lpips_net = _lpips_mod.LPIPS(net="alex").to(device)
        lpips_net.eval()
        log.info(
            f"loss = L_cls + {args.lambda_kl} * L_KL "
            f"+ {args.lambda_lpips} * L_LPIPS + {args.mu_l2} * L_L2"
        )
    else:
        log.info(
            f"LPIPS disabled — loss = L_cls + {args.lambda_kl} * L_KL "
            f"+ {args.mu_l2} * L_L2"
        )

    # ── 각 래퍼 모델의 정보 출력 ───────────────────────────────────────────
    for w in wrappers:
        accelerator.print(f"  [{w.model_id}] input size: {w.input_size}×{w.input_size}")

    # ── Human-readable target label string for logging ────────────────────────
    target_label_str = "/".join(w.id2label[w.target_idx] for w in wrappers)

    # ── Per-image processing loop ─────────────────────────────────────────────
    results: list[dict] = []

    for img_idx, img_path in enumerate(images, start=1):
        log.info("─" * 72)
        log.info(f"processing image [{img_idx}/{len(images)}]: {img_path.name}")

        model_header = "  ".join(f"[{i}] {w.model_id}" for i, w in enumerate(wrappers))
        accelerator.print(f"[{img_path.name}] models: {model_header}")

        size = (args.resize, args.resize) if args.resize else None
        pil_img    = load_image(img_path, size=size)
        orig_tensor = tensor_from_pil(pil_img)

        # Original target-class score (ensemble average)
        orig_tensor = orig_tensor.to(accelerator.device)
        orig_scores = []
        for w in wrappers:
            try:
                score_val = w.score(orig_tensor)
                orig_scores.append(score_val)
            except Exception as e:
                print(f"\nModel exception: {getattr(w, 'model_id', 'Unknown Model')}")
                print(f"Error: {e}")
                raise e

        orig_score_avg = sum(orig_scores) / len(wrappers)

        adv_tensor = adv_tensor = pgd_attack(
            wrappers=wrappers,
            orig_tensor=orig_tensor,
            device=device,
            image_name=img_path.name,
            eps=args.eps,
            steps=args.steps,
            optimizer=args.optimizer,
            algo=args.algo,
            lr=args.lr,
            momentum_mu=args.momentum_mu,
            soft_temp=args.soft_temp,
            lion_beta1=args.lion_beta1,
            lion_beta2=args.lion_beta2,
            optimizer_kwargs=args.optimizer_kwargs,
            vmi_n=args.vmi_n,
            vmi_beta=args.vmi_beta,
            lambda_lpips=args.lambda_lpips,
            mu_l2=args.mu_l2,
            lpips_net=lpips_net,
            accelerator=accelerator,
            label_smooth=args.label_smooth,
            lambda_kl=args.lambda_kl,
            kl_temp=args.kl_temp,
            multi_scale=args.multi_scale,
            use_jpeg=args.use_jpeg,
            use_transforms=args.use_transforms,
            reg_weight=args.reg_weight,
        )

        if args.preserve_hf:
            with torch.no_grad():
                delta_final = (adv_tensor - orig_tensor.cpu()).clamp(-args.eps, args.eps)
                delta_final = preserve_high_frequency_dct(delta_final, alpha=0.5)
                adv_tensor = (orig_tensor.cpu() + delta_final).clamp(0, 1)

        adv_score_avg  = sum(w.score(adv_tensor)  for w in wrappers) / len(wrappers)

        delta_score = adv_score_avg - orig_score_avg

        log.info(
            f"[{img_path.name}] adversarial target ({target_label_str}) avg: "
            f"{adv_score_avg:.4f}"
        )
        log.info(
            f"[{img_path.name}] score delta: "
            f"{orig_score_avg:.4f} → {adv_score_avg:.4f} ({delta_score:+.4f})"
        )

        if accelerator.is_main_process:
            out_path = output_dir / (img_path.stem + "_adv.png")
            save_image(adv_tensor, out_path)
            accelerator.print(f"[{img_path.name}] saved → {out_path}")
        else:
            out_path = output_dir / (img_path.stem + "_adv.png")

        results.append(
            {
                "name":        img_path.name,
                "orig_score":  orig_score_avg,
                "adv_score":   adv_score_avg,
                "delta_score": delta_score,
                "out_path":    out_path,
            }
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - run_start
    avg_elapsed   = total_elapsed / len(results)

    if accelerator.is_main_process:
        log.info("═" * 72)
        log.info("FINISHED")
        log.info(f"  total images  : {len(results)}")
        log.info(f"  target classes: {target_label_str}")
        log.info(f"  total elapsed : {total_elapsed:.1f}s  ({avg_elapsed:.1f}s / image)")
        log.info(f"  output dir    : {output_dir.resolve()}")
        log.info("  results:")
    for r in results:
        log.info(
            f"    {r['name']:<32}  "
            f"score {r['orig_score']:.4f} → {r['adv_score']:.4f}  "
            f"(Δ {r['delta_score']:+.4f})  →  {r['out_path'].name}"
        )
    log.info("═" * 72)


if __name__ == "__main__":
    main()