"""
vision_classifier_wrapper.py
-----------------------------
Universal wrapper layer that normalises heterogeneous vision classifier
back-ends (HuggingFace AutoModelForImageClassification, timm-native, and
VLM zero-shot models such as CLIP / SigLIP / SigLIP2) into a single
interface consumed by pgd_attack() and score helpers.

Drop-in replacement for the (model_id, feature_extractor, model) tuple
ensemble used in classifier_attack.py.  All preprocessing, forward pass,
and probability extraction are encapsulated here so the PGD loop never
needs to know which back-end it is talking to.

Integration guide (classifier_attack.py diff summary)
------------------------------------------------------
  Line  54  — add imports:
              from vision_classifier_wrapper import (
                  VisionClassifierWrapper, build_wrapper, ActivationFn
              )

  Line  54  — keep existing imports; AutoImageProcessor /
              AutoModelForImageClassification are still used inside
              HFClassifierWrapper, so no removal needed.

  Lines 676-695  (ensemble loading loop) — replace with:
              wrappers: list[VisionClassifierWrapper] = [
                  build_wrapper(spec_str, accelerator)
                  for spec_str in args.specs   # raw "-S" strings
              ]

  Lines 713-733  (norm_params / input_sizes / target_labels) — delete;
              all three are now attributes on each wrapper instance.

  Line  752  (compute_scores_single_class call) — replace with:
              w.score(orig_tensor)

  Lines 764-781  (pgd_attack call) — change signature to:
              pgd_attack(wrappers=wrappers, orig_tensor=..., ...)

  Lines 344-496  (pgd_attack body) — replace inner loops with:
              x_in     = w.preprocess(adv_native)
              logits   = w.logits(x_in)
              loss_cls = w.cls_loss(logits, label_smooth)
              loss_kl  = w.kl_loss(logits, orig, kl_temp)

  Lines 784-788  (adv score) — replace with:
              w.score(adv_tensor)

Spec string format (extended, backward-compatible)
---------------------------------------------------
  hf:ORG/MODEL:TARGET          HuggingFace AutoModelForImageClassification
  timm:timm/MODEL_NAME:TARGET  timm model loaded via HF TimmWrapper
  vlm:ORG/MODEL:LABEL,LABEL    CLIP / SigLIP / SigLIP2 zero-shot
                                comma-separated candidate labels;
                                TARGET is the label to maximise

  Backward-compatible (no prefix) — treated as hf:
  ORG/MODEL:TARGET             same as hf:ORG/MODEL:TARGET

Requirements (additions to existing requirements)
-------------------------------------------------
  uv pip install timm  (only needed for timm: prefix)
"""

from __future__ import annotations

from abc import ABC, abstractmethod  # ← ABC, abstractmethod 정의 해결
from typing import Literal           # ← Literal 정의 해결

import logging
import torch
import torch.nn as nn                  # ← 추가: nn.Parameter 사용용
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForImageClassification
from enum import Enum  # ← 상단 import 문에 추가

log = logging.getLogger("nsfw_attack")

class ActivationFn(str, Enum):
    SOFTMAX = "softmax"
    SIGMOID = "sigmoid"
    NONE = "none"


# ── Abstract base ──────────────────────────────────────────────────────────────

class VisionClassifierWrapper(ABC):
    """
    Common interface for all classifier back-ends.
    ...
    """

    model_id:   str
    target_idx: int
    input_size: int
    id2label:   dict[int, str]
    activation: ActivationFn

    # 🔽 [여기서부터 추가] 누락된 명시적 생성자 및 디바이스 제어 메서드 정의
    def __init__(
        self,
        model_id: str,
        target_idx: int,
        activation: ActivationFn = ActivationFn.SOFTMAX
    ):
        self.model_id = model_id
        self.target_idx = target_idx
        self.activation = activation
        
        # 하위 자식 클래스들이 파라미터 초기화 시 참조할 수 있도록 기본 디바이스 변수 선언
        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "VisionClassifierWrapper":
        """Move any stored tensors (mean/std) and model to device. Returns self."""
        self._device = device
        # 내부에 _model 객체(HuggingFace 가중치 등)가 저장되어 있다면 해당 장치로 완전히 보냄
        if hasattr(self, "_model"):
            self._model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        """Expose the current device of the wrapper."""
        return self._device
    # 🔼 [여기까지 추가]

    # ── Interface methods (must be implemented) ────────────────────────────────

    @abstractmethod
    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert a CHW float [0, 1] tensor (on any device) into the
        batched, normalised, resized input tensor expected by the model.

        Returns a [1, C, H, W] tensor on the model's device.

        IMPORTANT: implementation must use only differentiable torch ops
        (F.interpolate, arithmetic on tensors) so that gradients flow back
        through this function to the input tensor — which is required for
        the PGD optimisation loop.
        """

    @abstractmethod
    def logits(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Run the model forward pass on a preprocessed [1, C, H, W] tensor.

        Returns raw (pre-activation) logits of shape [1, num_classes].
        For VLM zero-shot wrappers this is the cosine-similarity score
        vector (one entry per candidate label), still pre-activation.
        """

    # ── Derived helpers (shared across all back-ends) ──────────────────────────

    def probs(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Apply the correct activation function to logits and return a
        probability vector of shape [num_classes].

        Uses sigmoid for SigLIP-family models (independent per-class
        probabilities) and softmax for all others (mutually exclusive).
        """
        raw = self.logits(preprocessed)  # [1, C]
        if self.activation == "sigmoid":
            return torch.sigmoid(raw)[0]
        return F.softmax(raw, dim=-1)[0]

    @torch.no_grad()
    def score(self, tensor: torch.Tensor) -> float:
        """
        Return the scalar target-class probability for a CHW [0, 1]
        tensor.  Runs under no_grad — used for before/after logging only,
        not inside the optimisation loop.
        """
        x = self.preprocess(tensor)
        return self.probs(x)[self.target_idx].item()

    def cls_loss(
        self,
        logits_adv: torch.Tensor,
        label_smooth: float = 0.1,
    ) -> torch.Tensor:
        """
        Soft-label classification loss that pushes probability mass onto
        target_idx while leaving a small epsilon on all other classes.

        For softmax models: KL( log_softmax(logits) || soft_target )
        For sigmoid models: binary cross-entropy on the target logit,
          which avoids the invalid assumption of mutual exclusivity.

        Parameters
        ----------
        logits_adv  : raw logits [1, C] from the adversarial image
        label_smooth: fraction of probability mass spread over non-target
                      classes (ignored for sigmoid mode)
        """
        if self.activation == "sigmoid":
            # Each class is treated as an independent binary problem.
            # We just maximise the target logit — equivalent to minimising
            # BCE with label = 1 for the target class.
            target_logit = logits_adv[0, self.target_idx].unsqueeze(0)
            target_label = torch.ones(1, device=logits_adv.device)
            return F.binary_cross_entropy_with_logits(target_logit, target_label)

        # Softmax branch — same formula as the original code
        num_classes = logits_adv.shape[1]
        soft_target = torch.full_like(
            logits_adv, label_smooth / max(num_classes - 1, 1)
        )
        soft_target[0, self.target_idx] = 1.0 - label_smooth
        log_probs = F.log_softmax(logits_adv, dim=1)
        return F.kl_div(log_probs, soft_target, reduction="batchmean")

    def kl_loss(
        self,
        logits_adv: torch.Tensor,
        orig_tensor: torch.Tensor,
        kl_temp: float = 2.0,
    ) -> torch.Tensor:
        """
        KL divergence between the original and adversarial output
        distributions, scaled by temperature kl_temp.

        Encourages the perturbed image to stay on the natural image
        manifold by penalising large distribution shifts.

        For sigmoid models the KL is computed per-class independently
        (Bernoulli KL) to stay consistent with the activation semantics.

        Parameters
        ----------
        logits_adv  : adversarial logits [1, C] — must carry grad
        orig_tensor : original CHW [0, 1] tensor (no grad needed)
        kl_temp     : temperature for softening both distributions
        """
        with torch.no_grad():
            x_orig      = self.preprocess(orig_tensor)
            logits_orig = self.logits(x_orig)

        if self.activation == "sigmoid":
            # Bernoulli KL: p*log(p/q) + (1-p)*log((1-p)/(1-q))
            # numerically stable via torch built-ins
            p      = torch.sigmoid(logits_orig / kl_temp)        # [1, C]
            log_q  = F.logsigmoid(logits_adv  / kl_temp)         # [1, C]
            log_1q = F.logsigmoid(-logits_adv / kl_temp)         # [1, C]
            kl = p * (torch.log(p + 1e-8) - log_q) + (1 - p) * (
                torch.log(1 - p + 1e-8) - log_1q
            )
            return kl.mean()

        # Softmax branch — identical to original code
        p_orig    = F.softmax(logits_orig / kl_temp, dim=1)
        log_p_adv = F.log_softmax(logits_adv  / kl_temp, dim=1)
        return F.kl_div(log_p_adv, p_orig, reduction="batchmean")

    # ── Device placement ───────────────────────────────────────────────────────

    def to(self, device: torch.device) -> "VisionClassifierWrapper":
        """Move any stored tensors (mean/std) to device. Returns self."""
        return self  # subclasses override if they hold tensors


# ── Back-end 1: HuggingFace AutoModelForImageClassification ───────────────────

# vision_classifier_wrapper.py 의 관련 부분을 아래와 같이 수정합니다.

class HFClassifierWrapper(VisionClassifierWrapper):
    """
    Wrapper for HuggingFace AutoModelForImageClassification models.
    """
    def __init__(
        self,
        model_id: str,
        processor,
        model,
        target_idx: int,
        activation: ActivationFn = ActivationFn.SOFTMAX
    ):
        model_cls_name = type(model).__name__.lower()
        if "siglip" in model_cls_name or "siglip" in model_id.lower():
            activation = ActivationFn.SIGMOID

        super().__init__(model_id, target_idx, activation)
        self._device = next(model.parameters()).device
        self._model = model
        self.id2label = model.config.id2label

        if hasattr(model.config, "vision_config") and hasattr(model.config.vision_config, "image_size"):
            self.input_size = model.config.vision_config.image_size
        elif hasattr(processor, "size") and isinstance(processor.size, dict):
            self.input_size = processor.size.get("height", processor.size.get("shortest_edge", 224))
        elif hasattr(processor, "size") and isinstance(processor.size, int):
            self.input_size = processor.size
        elif hasattr(model, "config") and hasattr(model.config, "image_size") and model.config.image_size:
            self.input_size = model.config.image_size
        elif hasattr(model, "timm_model") and hasattr(model.timm_model, "default_cfg"):
            self.input_size = model.timm_model.default_cfg.get("input_size", (3, 224, 224))[-1]
        else:
            if "384" in model_id or "siglip" in model_id.lower():
                self.input_size = 384
            elif "448" in model_id:
                self.input_size = 448
            else:
                self.input_size = 224

        import logging
        logging.info(f"  [{model_id}] Resolved execution input size: {self.input_size}x{self.input_size} | Activation: {self.activation}")

        if hasattr(processor, "image_mean"):
            mean_val = processor.image_mean
            std_val = processor.image_std
        elif hasattr(processor, "image_processor") and hasattr(processor.image_processor, "mean"):
            mean_val = processor.image_processor.mean
            std_val = processor.image_processor.std
        else:
            if "siglip" in model_cls_name or "siglip" in model_id.lower():
                mean_val = [0.5, 0.5, 0.5]
                std_val = [0.5, 0.5, 0.5]
            else:
                mean_val = [0.485, 0.456, 0.406]
                std_val = [0.229, 0.224, 0.225]

        self._mean = nn.Parameter(
            torch.tensor(mean_val, dtype=torch.float32).view(1, 3, 1, 1).to(self._device),
            requires_grad=False
        )
        self._std = nn.Parameter(
            torch.tensor(std_val, dtype=torch.float32).view(1, 3, 1, 1).to(self._device),
            requires_grad=False
        )

    # ── [필수] 추상 메서드 실체화 구현 파트 ───────────────────────────────────

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Differentiable PGD-friendly image preprocessing channel.
        Input: [C, H, W] or [B, C, H, W] tensor in range [0, 1]
        """
        x = x.to(self._device)

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-2:] != (self.input_size, self.input_size):
            import torch.nn.functional as F
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False
            )

        preprocessed = (x - self._mean) / self._std
        
        return preprocessed.to(self._device)

    def logits(self, preprocessed_x: torch.Tensor) -> torch.Tensor:
        if preprocessed_x.dim() == 3:
            preprocessed_x = preprocessed_x.unsqueeze(0)
    
        current_device = next(self._model.parameters()).device
        preprocessed_x = preprocessed_x.to(current_device)

        outputs = self._model(pixel_values=preprocessed_x)
        res = outputs.logits  # 원본 그대로 유지

        if res.dim() == 1:
            res = res.unsqueeze(0)
        elif res.dim() > 2:
            res = res.view(1, -1)

        if res.size(0) != 1:
            res = res[:1]

        if res.numel() == 0:
            print(f"\n[WARNING] Unexpected logits shape — model: {getattr(self._model.config, '_name_or_path', 'Unknown')}")
            print(f"Raw outputs.logits shape: {outputs.logits.shape}, after reshape: {res.shape}")
            res = torch.zeros(1, 5, device=current_device)

        return res  # 항상 [1, C]




# ── Back-end 2: VLM zero-shot (CLIP / SigLIP / SigLIP2) ──────────────────────

class VLMZeroShotWrapper(VisionClassifierWrapper):
    """
    Zero-shot classifier backed by any dual-encoder VLM: CLIP, SigLIP,
    or SigLIP2.

    Classification is performed by computing cosine similarity between the
    image embedding and a set of pre-encoded text embeddings (one per
    candidate label), then treating the resulting similarity scores as
    logits.

    Activation function
    -------------------
    CLIP      → softmax  (contrastive, mutually exclusive)
    SigLIP/2  → sigmoid  (independent per-class probabilities)

    The correct activation is detected automatically by checking whether
    the model's config class name contains "siglip".

    Preprocessing
    -------------
    Image pixels are resized and normalised using the model processor's
    stored mean/std — same differentiable path as HFClassifierWrapper.
    Text embeddings are pre-computed once at construction time and cached.

    Parameters
    ----------
    model_id          : HuggingFace model identifier (for logging)
    model             : CLIPModel / SiglipModel / Siglip2Model instance
    processor         : matching AutoProcessor instance
    candidate_labels  : list of text strings, one per class
    target_label      : substring matched against candidate_labels to
                        identify which class index to maximise
    device            : torch.device for all tensor operations
    """

    def __init__(
        self,
        model_id: str,
        model,
        processor,
        candidate_labels: list[str],
        target_label: str,
        device: torch.device,
    ) -> None:
        self.model_id        = model_id
        self.id2label        = {i: lbl for i, lbl in enumerate(candidate_labels)}
        self._model          = model
        self._processor      = processor
        self._device         = device
        self.input_size      = self._resolve_input_size(processor)

        # Detect activation: SigLIP family uses sigmoid, all others softmax
        cfg_cls = type(model.config).__name__.lower()
        self.activation: ActivationFn = (
            "sigmoid" if "siglip" in cfg_cls else "softmax"
        )

        # Resolve target index from label substring match
        target_lower = target_label.lower()
        matches = [
            i for i, lbl in self.id2label.items()
            if target_lower in lbl.lower()
        ]
        if not matches:
            log.warning(
                f"[{model_id}] target '{target_label}' not found in "
                f"{list(self.id2label.values())} — falling back to index 0"
            )
            self.target_idx = 0
        else:
            self.target_idx = matches[0]

        # Pre-compute normalisation tensors
        img_mean = getattr(
            processor, "image_mean", [0.48145466, 0.4578275, 0.40821073]
        )
        img_std = getattr(
            processor, "image_std",  [0.26862954, 0.26130258, 0.27577711]
        )
        self._mean = (
            torch.tensor(img_mean, dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(device)
        )
        self._std = (
            torch.tensor(img_std, dtype=torch.float32)
            .view(1, 3, 1, 1)
            .to(device)
        )

        # Pre-encode all candidate labels once; shape [num_labels, embed_dim]
        # This is the only part that uses the text encoder — never repeated.
        self._text_embeds = self._encode_texts(candidate_labels)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _resolve_input_size(processor) -> int:
        """
        Extract the square input size from a VLM processor.
        Tries several attribute names used by different processor versions.
        """
        for attr in ("size", "image_size", "crop_size"):
            val = getattr(processor, attr, None)
            if val is None:
                continue
            if isinstance(val, int):
                return val
            if isinstance(val, dict):
                return val.get("shortest_edge", val.get("height", 224))
        return 224

    @torch.no_grad()
    def _encode_texts(self, labels: list[str]) -> torch.Tensor:
        """
        Tokenise and encode a list of label strings through the model's
        text encoder.  Returns L2-normalised embeddings [N, D].
        """
        # Construct natural-language prompts identical to zero-shot evaluation
        prompts = [f"a photo of {lbl}" for lbl in labels]
        inputs  = self._processor(
            text=prompts,
            padding="max_length",  # required by SigLIP tokeniser
            return_tensors="pt",
        ).to(self._device)
        # Both CLIPModel and SiglipModel expose get_text_features()
        embeds = self._model.get_text_features(**inputs)  # [N, D]
        return F.normalize(embeds, dim=-1)

    # ── VisionClassifierWrapper implementation ─────────────────────────────────

    def preprocess(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Resize and normalise a CHW [0, 1] tensor.
        Identical differentiable path to HFClassifierWrapper.preprocess().
        """
        x = tensor.to(self._device)
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )
        return (x - self._mean) / self._std

    def logits(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between the image embedding and all
        pre-cached text embeddings, scaled by the model's logit_scale.

        Returns [1, num_labels] — same shape contract as HF classifiers.
        """
        img_embed = self._model.get_image_features(pixel_values=preprocessed)
        img_embed = F.normalize(img_embed, dim=-1)  # [1, D]

        # Retrieve learned temperature; fall back to 1/0.07 ≈ 14.3 (CLIP default)
        logit_scale = getattr(self._model, "logit_scale", None)
        scale = (
            logit_scale.exp().clamp(max=100.0)
            if logit_scale is not None
            else torch.tensor(14.3, device=self._device)
        )

        # [1, D] @ [D, N] → [1, N]
        similarity = img_embed @ self._text_embeds.T
        return similarity * scale

    def to(self, device: torch.device) -> "VLMZeroShotWrapper":
        self._device      = device
        self._mean        = self._mean.to(device)
        self._std         = self._std.to(device)
        self._text_embeds = self._text_embeds.to(device)
        return self


# ── Factory ────────────────────────────────────────────────────────────────────

def _parse_extended_spec(spec: str) -> tuple[str, str, str, list[str]]:
    """
    Parse an extended spec string into (backend, model_id, target, extras).

    Format (colon-separated, left-to-right):
      [BACKEND:]MODEL_ID[:TARGET]

    BACKEND — optional prefix, one of: "hf", "timm", "vlm"
              omitted → treated as "hf" for backward compatibility
    MODEL_ID — HuggingFace repo identifier (may contain '/' and '.')
    TARGET   — class label substring, integer index, or comma-separated
                label list (vlm only)

    Examples
    --------
      "Falconsai/nsfw_image_detection:nsfw"
          → ("hf",   "Falconsai/nsfw_image_detection", "nsfw",  [])
      "hf:AdamCodd/vit-base-nsfw-detector:nsfw"
          → ("hf",   "AdamCodd/vit-base-nsfw-detector",  "nsfw",  [])
      "timm:timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k:283"
          → ("timm", "timm/convnext_xxlarge.*",           "283",   [])
      "vlm:openai/clip-vit-base-patch32:nsfw,explicit,safe"
          → ("vlm",  "openai/clip-vit-base-patch32",     "nsfw",
             ["nsfw", "explicit", "safe"])
    """
    KNOWN_BACKENDS = {"hf", "timm", "vlm"}

    # Check whether the first colon-delimited token is a backend prefix
    parts = spec.split(":", 1)
    if len(parts) == 2 and parts[0].lower() in KNOWN_BACKENDS:
        backend   = parts[0].lower()
        remainder = parts[1]
    else:
        backend   = "hf"
        remainder = spec

    # The remainder may be  MODEL_ID  or  MODEL_ID:TARGET
    # Use rfind so HF org/repo paths with '/' are safe against mis-splitting
    colon = remainder.rfind(":")
    if colon == -1:
        model_id = remainder
        target   = ""
    else:
        model_id = remainder[:colon]
        target   = remainder[colon + 1:]

    # For vlm backend the target field is comma-separated candidate labels
    extras = [lbl.strip() for lbl in target.split(",")] if backend == "vlm" else []

    return backend, model_id, target, extras

def parse_spec(spec_str: str) -> dict:
    """
    '-S' 옵션 인자를 안전하게 분해하여 딕셔너리로 반환합니다.
    """
    backend = "hf"
    
    # 1. 백엔드 접두사 검사 (hf:, vlm:, timm:)
    for prefix in ["hf:", "vlm:", "timm:"]:
        if spec_str.startswith(prefix):
            backend = prefix[:-1]  # 오타 수정 완료
            spec_str = spec_str[len(prefix):]
            break
            
    # 2. 역방향 탐색으로 타겟 클래스 분리
    colon_idx = spec_str.rfind(":")
    if colon_idx != -1:
        model_id = spec_str[:colon_idx]
        target_cls = spec_str[colon_idx + 1:]
    else:
        model_id = spec_str
        target_cls = None

    if "siglip" in model_id.lower():
        activation = ActivationFn.SIGMOID
    else:
        activation = ActivationFn.SOFTMAX
        
    return {
        "backend": backend,
        "model_id": model_id,
        "target_cls": target_cls,
        "activation": activation
    }

def build_wrapper(spec_str: str, accelerator=None) -> VisionClassifierWrapper:
    """
    Parses a spec string and constructs the appropriate VisionClassifierWrapper.
    """
    spec = parse_spec(spec_str)
    backend = spec["backend"]
    model_id = spec["model_id"]
    target_cls = spec["target_cls"]
    activation = spec["activation"]

    device = accelerator.device if accelerator is not None else torch.device("cpu")

    if backend == "hf":
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()

        try:
            processor = AutoImageProcessor.from_pretrained(model_id)
            if processor is None:
                raise ValueError("Processor returned None")
        except Exception:
            logging.warning(f"[{model_id}] preprocessor_config.json not found — falling back to default processor config.")
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
            if hasattr(model.config, "image_size"):
                target_sz = model.config.image_size
                if isinstance(target_sz, int):
                    processor.size = {"height": target_sz, "width": target_sz}

        target_idx = _resolve_target_idx_from_model(model, target_cls)

        wrapper = HFClassifierWrapper(
            model_id=model_id,
            processor=processor,
            model=model,
            target_idx=target_idx,
            activation=activation
        )
        wrapper.to(device)

        logging.info(
            f"  loaded: {model_id}  target_idx={wrapper.target_idx}  "
            f"target_label='{wrapper.id2label[wrapper.target_idx]}'  "
            f"all_labels={wrapper.id2label}"
        )
        return wrapper

    elif backend == "vlm":
        from transformers import AutoModel, AutoProcessor

        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(model_id)

        candidate_labels = [
            lbl.strip() for lbl in (target_cls or "nsfw,safe").split(",") if lbl.strip()
        ]
        if len(candidate_labels) < 2:
            candidate_labels.append("safe" if candidate_labels[0].lower() != "safe" else "nsfw")
        target_label = candidate_labels[0]

        wrapper = VLMZeroShotWrapper(
            model_id=model_id,
            model=model,
            processor=processor,
            candidate_labels=candidate_labels,
            target_label=target_label,
            device=device,
        )

        logging.info(
            f"  loaded (vlm): {model_id}  candidate_labels={candidate_labels}  "
            f"target_idx={wrapper.target_idx}  "
            f"target_label='{candidate_labels[wrapper.target_idx]}'"
        )
        return wrapper

    else:
        raise NotImplementedError(f"Backend '{backend}' is not yet implemented.")

def _resolve_target_idx_from_model(model, target: str | int | None) -> int:
    """
    Resolve a class index from a loaded HF model.

    Extracted from the original resolve_target_idx() so it can be called
    inside build_wrapper() without requiring a separate function call.

    Rules (unchanged from original classifier_attack.py)
    -----------------------------------------------------
    int  → validated against id2label, returned as-is
    str  → case-insensitive substring search in id2label values
    None → falls back to searching for "nsfw"
    """
    id2label: dict = model.config.id2label

    if isinstance(target, int):
        if target not in id2label:
            raise ValueError(
                f"Target index {target} out of range for id2label: {id2label}"
            )
        return target

    # Accept numeric strings passed from the spec parser (e.g. ":1")
    if isinstance(target, str) and target.isdigit():
        idx = int(target)
        if idx not in id2label:
            raise ValueError(
                f"Target index {idx} out of range for id2label: {id2label}"
            )
        return idx

    search = (target or "nsfw").lower()
    for idx, label in id2label.items():
        if search in label.lower():
            return int(idx)

    log.warning(
        f"Label '{search}' not found in {id2label} — falling back to index 1"
    )
    return 1