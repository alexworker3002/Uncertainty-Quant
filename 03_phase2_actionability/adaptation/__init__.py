# Phase 2: LoRA and safe adaptation
from .lora_surgery import inject_lora_decoder, freeze_backbone
from .safe_anchor_loss import GeometricFidelityLoss, geometric_fidelity_loss
