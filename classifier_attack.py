"""
nsfw_attack.py
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

import argparse
import logging
import sys
import time
from pathlib import Path

from accelerate import Accelerator
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
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
log = logging.getLogger("nsfw_attack")

if not HAS_LPIPS:
    log.warning("lpips not found — falling back to L2 regularization only.")
    log.warning("  install with: uv pip install lpips")

if not HAS_SCIPY:
    log.warning("scipy not found — DCT high-frequency preservation disabled.")
    log.warning("  install with: uv pip install scipy")


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
    Apply an affine transformation (rotation, scaling, translation) using
    Lanczos resampling.
    tensor: CHW [0, 1]
    """
    return transforms.functional.affine(
        tensor,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=0,
        resample=Image.LANCZOS,
    )


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
    lr: float = 0.005,
    lambda_lpips: float = 2.0,
    mu_l2: float = 0.5,
    lpips_net=None,
    accelerator=None,
    label_smooth: float = 0.1,
    lambda_kl: float = 0.3,
    kl_temp: float = 2.0,
) -> torch.Tensor:
    """
    PGD attack maximising the target class probability for each model in the
    ensemble while minimising perceptual distortion.

    Loss:
      L = L_cls(soft-label KL) + lambda_kl * L_KL(orig||adv)
          + lambda_lpips * L_LPIPS + mu_l2 * L_L2
    """
    orig = orig_tensor.to(device)
    delta = torch.zeros_like(orig, requires_grad=True, device=device)

    optimizer = torch.optim.Adam([delta], lr=lr)
    step_w = len(str(steps))

    log_fn = accelerator.print if accelerator else log.info
    log_fn(f"[{image_name}] starting PGD  steps={steps}  eps={eps}  lr={lr}")

    t0 = time.time()
    for step in range(1, steps + 1):
        optimizer.zero_grad()

        adv_native = orig + delta

        # ── 래퍼(Wrapper) 기반 전처리 및 손실 계산 ──
        target_probs: list[torch.Tensor] = []
        loss_cls = torch.tensor(0.0, device=device)
        loss_kl  = torch.tensor(0.0, device=device)

        for w in wrappers:
            x_in      = w.preprocess(adv_native)          # 리사이즈 + 정규화 (미분 가능)
            logits    = w.logits(x_in)                    # 로짓 수집
            
            # 로그 출력을 위한 개별 타겟 클래스 확률 수집
            prob      = w.probs(x_in)[w.target_idx]
            target_probs.append(prob)
            
            # 누적 손실 계산
            loss_cls += w.cls_loss(logits, label_smooth)
            loss_kl  += w.kl_loss(logits, orig, kl_temp)

        # 앙상블 모델 수로 나눠 평균 손실로 변환
        loss_cls = loss_cls / len(wrappers)
        loss_kl  = loss_kl / len(wrappers)
        
        loss_l2  = (delta ** 2).mean()

        # ── Perceptual regularisation (LPIPS) ────────────────────────────────
        if lpips_net is not None:
            loss_lpips = lpips_net(orig.unsqueeze(0), adv_native.unsqueeze(0)).mean()
            loss = (
                loss_cls
                + lambda_kl   * loss_kl
                + lambda_lpips * loss_lpips
                + mu_l2       * loss_l2
            )
        else:
            loss_lpips = torch.tensor(0.0)
            loss = loss_cls + lambda_kl * loss_kl + mu_l2 * loss_l2

        loss.backward()
        optimizer.step()

        # Project delta back onto the Linf ball
        with torch.no_grad():
            delta.clamp_(-eps, eps)

        target_scores_str = "  ".join(f"{p.item():.4f}" for p in target_probs)
        log_fn(
            f"[{image_name}] step [{step:0{step_w}d}/{steps}]  "
            f"loss={loss.item():.6f}  "
            f"loss_cls={loss_cls.item():.6f}  "
            f"loss_kl={loss_kl.item():.6f}  "
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
        "--lr", type=float, default=0.005,
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
        build_wrapper(s, accelerator) for s in args.specs
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
        orig_score_avg = sum(w.score(orig_tensor) for w in wrappers) / len(wrappers)

        log.info(
            f"[{img_path.name}] original target ({target_label_str}) avg: "
            f"{orig_score_avg:.4f}"
        )

        # Run PGD attack (정리된 wrappers 구조만 인자로 전달)
        adv_tensor = pgd_attack(
            wrappers=wrappers,
            orig_tensor=orig_tensor,
            device=device,
            image_name=img_path.name,
            eps=args.eps,
            steps=args.steps,
            lr=args.lr,
            lambda_lpips=args.lambda_lpips,
            mu_l2=args.mu_l2,
            lpips_net=lpips_net,
            accelerator=accelerator,
            label_smooth=args.label_smooth,
            lambda_kl=args.lambda_kl,
            kl_temp=args.kl_temp,
        )

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