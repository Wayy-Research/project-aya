"""
Activation extraction utilities for teacher and student models.

Registers forward hooks to capture intermediate representations
without modifying model forward passes. Designed for memory efficiency:
activations are collected layer-by-layer, not all at once.

Supports:
    - HuggingFace CohereForCausalLM (Aya teacher)
    - Aetheris HybridMambaMoE (student)

Wayy Research, 2024-2026.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Activation store
# ---------------------------------------------------------------------------

@dataclass
class ActivationStore:
    """
    Collects activations from registered hooks.

    Each hook appends to a buffer keyed by layer name. After a forward
    pass, call .collect() to get flattened (n_tokens, d_model) tensors
    and .clear() to free memory.
    """

    _buffers: dict[str, list[torch.Tensor]] = field(
        default_factory=dict, repr=False
    )
    _hooks: list[torch.utils.hooks.RemovableHook] = field(
        default_factory=list, repr=False
    )
    detach: bool = True
    device: str = "cpu"

    def _make_hook(
        self, name: str
    ) -> Callable[[nn.Module, Any, Any], None]:
        """Create a forward hook that stores the layer output."""

        def hook_fn(
            module: nn.Module,
            input: Any,
            output: Any,
        ) -> None:
            # Handle different output formats
            if isinstance(output, tuple):
                # Transformer layers often return (hidden_states, ...)
                # MoE layers return (hidden_states, aux_loss)
                act = output[0]
            elif isinstance(output, torch.Tensor):
                act = output
            else:
                warnings.warn(
                    f"Unexpected output type {type(output)} from {name}. "
                    "Skipping activation capture."
                )
                return

            if self.detach:
                act = act.detach()

            # Flatten to (n_tokens, d_model): (B, L, D) -> (B*L, D)
            if act.dim() == 3:
                act = act.reshape(-1, act.shape[-1])
            elif act.dim() == 2:
                pass  # Already (n, d)
            else:
                warnings.warn(
                    f"Unexpected activation shape {act.shape} from {name}."
                )
                return

            # Move to storage device to free GPU memory
            act = act.to(self.device, dtype=torch.float32)

            if name not in self._buffers:
                self._buffers[name] = []
            self._buffers[name].append(act)

        return hook_fn

    def register(self, module: nn.Module, name: str) -> None:
        """Register a forward hook on a module."""
        hook = module.register_forward_hook(self._make_hook(name))
        self._hooks.append(hook)

    def collect(self) -> dict[str, torch.Tensor]:
        """
        Concatenate buffered activations into single tensors.

        Returns:
            {layer_name: (total_tokens, d_model)} dict.
        """
        result: dict[str, torch.Tensor] = {}
        for name, tensors in self._buffers.items():
            if tensors:
                result[name] = torch.cat(tensors, dim=0)
        return result

    def clear(self) -> None:
        """Free activation buffers (keep hooks registered)."""
        for tensors in self._buffers.values():
            tensors.clear()
        self._buffers.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks and clear buffers."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self.clear()

    def __del__(self) -> None:
        self.remove_hooks()


# ---------------------------------------------------------------------------
# Teacher hook registration (HuggingFace Cohere / generic transformer)
# ---------------------------------------------------------------------------

def register_teacher_hooks(
    model: nn.Module,
    store: ActivationStore,
    layer_indices: Optional[list[int]] = None,
    hook_attention: bool = True,
    hook_ffn: bool = True,
) -> None:
    """
    Register hooks on a HuggingFace transformer teacher model.

    Targets the output of attention and MLP sub-layers within each
    transformer block.

    For CohereForCausalLM, the layer structure is:
        model.model.layers[i].self_attn  (attention)
        model.model.layers[i].mlp        (FFN)

    Falls back to generic patterns if Cohere-specific paths are not found.

    Args:
        model: HuggingFace causal LM.
        store: ActivationStore to collect into.
        layer_indices: which layers to hook (None = all).
        hook_attention: whether to hook attention outputs.
        hook_ffn: whether to hook FFN/MLP outputs.
    """
    # Try Cohere / Llama-style paths
    layers = _get_transformer_layers(model)

    if layers is None:
        raise ValueError(
            "Could not locate transformer layers in model. "
            "Expected model.model.layers or model.transformer.h"
        )

    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    for idx in layer_indices:
        if idx >= len(layers):
            warnings.warn(f"Layer index {idx} out of range, skipping.")
            continue

        layer = layers[idx]

        if hook_attention:
            attn_module = _get_attention_module(layer)
            if attn_module is not None:
                store.register(attn_module, f"teacher.attn.{idx}")
            else:
                warnings.warn(f"No attention module found in layer {idx}")

        if hook_ffn:
            ffn_module = _get_ffn_module(layer)
            if ffn_module is not None:
                store.register(ffn_module, f"teacher.ffn.{idx}")
            else:
                warnings.warn(f"No FFN module found in layer {idx}")


def _get_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the main layer list in a HuggingFace model."""
    # Cohere / Llama / Mistral path
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2 / GPT-Neo path
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # Direct model (no LM head wrapper)
    if hasattr(model, "layers") and isinstance(model.layers, nn.ModuleList):
        return model.layers
    return None


def _get_attention_module(layer: nn.Module) -> Optional[nn.Module]:
    """Locate the attention sub-module in a transformer layer."""
    for attr in ("self_attn", "attn", "attention"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


def _get_ffn_module(layer: nn.Module) -> Optional[nn.Module]:
    """Locate the FFN/MLP sub-module in a transformer layer."""
    for attr in ("mlp", "ffn", "feed_forward", "fc"):
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return None


# ---------------------------------------------------------------------------
# Student hook registration (Aetheris HybridMambaMoE)
# ---------------------------------------------------------------------------

def register_student_hooks(
    model: nn.Module,
    store: ActivationStore,
    layer_indices: Optional[list[int]] = None,
) -> None:
    """
    Register hooks on Aetheris HybridMambaMoE student model.

    Architecture layout (from model.py):
        - Even layers (0, 2, 4, ...): SSMBlock
        - Odd layers (1, 3, 5, ...): SparseMoELayer

    Hooks are placed on each layer's forward output.

    Args:
        model: Aetheris HybridMambaMoE instance.
        store: ActivationStore to collect into.
        layer_indices: which layers to hook (None = all).
    """
    if not hasattr(model, "layers"):
        raise ValueError(
            "Student model must have a .layers attribute "
            "(expected HybridMambaMoE)."
        )

    if layer_indices is None:
        layer_indices = list(range(len(model.layers)))

    for idx in layer_indices:
        if idx >= len(model.layers):
            warnings.warn(f"Layer index {idx} out of range, skipping.")
            continue

        layer = model.layers[idx]
        layer_type = "ssm" if idx % 2 == 0 else "moe"
        store.register(layer, f"student.{layer_type}.{idx}")


# ---------------------------------------------------------------------------
# Convenience context manager
# ---------------------------------------------------------------------------

@contextmanager
def activation_capture(
    teacher: nn.Module,
    student: nn.Module,
    teacher_layers: Optional[list[int]] = None,
    student_layers: Optional[list[int]] = None,
    device: str = "cpu",
    detach: bool = True,
):
    """
    Context manager for capturing activations from both models.

    Usage:
        with activation_capture(teacher, student) as (t_store, s_store):
            teacher(input_ids)
            student(input_ids)
            t_acts = t_store.collect()
            s_acts = s_store.collect()

    Hooks are automatically removed on exit.
    """
    t_store = ActivationStore(detach=detach, device=device)
    s_store = ActivationStore(detach=detach, device=device)

    register_teacher_hooks(teacher, t_store, teacher_layers)
    register_student_hooks(student, s_store, student_layers)

    try:
        yield t_store, s_store
    finally:
        t_store.remove_hooks()
        s_store.remove_hooks()


# ---------------------------------------------------------------------------
# Layer mapping utilities
# ---------------------------------------------------------------------------

def build_layer_mapping(
    n_teacher_layers: int,
    n_student_layers: int,
    strategy: str = "interleaved",
) -> list[tuple[int, int, str]]:
    """
    Build teacher->student layer mapping for alignment.

    The student alternates SSM (even) and MoE (odd) blocks. The teacher
    has uniform transformer layers (attention + FFN each).

    Strategy "interleaved":
        - Student SSM layer i (even) aligns with teacher attention at
          mapped index
        - Student MoE layer i (odd) aligns with teacher FFN at
          mapped index
        Linear mapping: teacher_idx = i * n_teacher / n_student

    Args:
        n_teacher_layers: number of transformer layers in teacher.
        n_student_layers: number of layers in student.
        strategy: mapping strategy name.

    Returns:
        List of (teacher_layer_idx, student_layer_idx, component) tuples
        where component is "attn->ssm" or "ffn->moe".
    """
    if strategy != "interleaved":
        raise ValueError(f"Unknown mapping strategy: {strategy}")

    mapping: list[tuple[int, int, str]] = []
    ratio = n_teacher_layers / n_student_layers

    for s_idx in range(n_student_layers):
        t_idx = int(s_idx * ratio)
        t_idx = min(t_idx, n_teacher_layers - 1)

        if s_idx % 2 == 0:
            # SSM block <- teacher attention
            mapping.append((t_idx, s_idx, "attn->ssm"))
        else:
            # MoE block <- teacher FFN
            mapping.append((t_idx, s_idx, "ffn->moe"))

    return mapping
