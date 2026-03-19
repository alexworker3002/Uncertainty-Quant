"""
LoRA injection for Decoder-only lightweight adaptation.

Injects low-rank parameters (r=4 or 8) into 3D U-Net Decoder layers via PEFT.
Enables test-time adaptation without modifying the frozen backbone.
"""

from typing import List, Optional

try:
    import torch
    import torch.nn as nn
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    torch = None
    nn = None
    LoraConfig = None
    get_peft_model = None
    TaskType = None


def inject_lora_decoder(
    model: "nn.Module",
    r: int = 8,
    lora_alpha: int = 16,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
) -> "nn.Module":
    """
    Inject LoRA into Decoder layers only.

    Parameters
    ----------
    model : nn.Module
        Segmentation model (e.g. 3D U-Net) with named modules.
    r : int
        LoRA rank (default 8).
    lora_alpha : int
        LoRA alpha scaling.
    target_modules : list, optional
        Module name patterns for LoRA (e.g. ["decoder.*.conv", "decoder.*.up"]).
        If None, uses common Decoder patterns.
    modules_to_save : list, optional
        Additional modules to train (e.g. segmentation head).

    Returns
    -------
    nn.Module
        Model with LoRA applied.
    """
    if get_peft_model is None or LoraConfig is None:
        raise ImportError(
            "PEFT required for LoRA. Install with: pip install peft"
        )

    if target_modules is None:
        target_modules = _default_decoder_targets(model)

    config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        inference_mode=False,
    )

    return get_peft_model(model, config)


def _default_decoder_targets(model: "nn.Module") -> List[str]:
    """Heuristic: find linear/conv layers in decoder submodules."""
    targets = []
    for name, _ in model.named_modules():
        if "dec" in name.lower() and ("linear" in name or "conv" in name):
            parts = name.split(".")
            if len(parts) >= 2:
                targets.append(parts[-1])
    return list(set(targets)) if targets else ["down"]


def freeze_backbone(model: "nn.Module", decoder_prefix: str = "decoder") -> None:
    """
    Freeze all parameters except decoder (and LoRA).

    Parameters
    ----------
    model : nn.Module
        Model with named parameters.
    decoder_prefix : str
        Prefix for decoder parameters to keep unfrozen.
    """
    for name, param in model.named_parameters():
        if "lora" in name.lower() or decoder_prefix.lower() in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False
