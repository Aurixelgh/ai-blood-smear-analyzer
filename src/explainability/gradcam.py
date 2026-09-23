"""
Grad-CAM explainability for the trained classifier (spec section 8).

Precedent for this exact problem domain: Chen et al. 2022 (BMC Bioinformatics,
DOI 10.1186/s12859-022-04824-6) used Grad-CAM occlusion testing on a WBC
classifier trained on Raabin-WBC/LISC/BCCD -- see reports/architecture_decision.md
section 5.

This module returns ONLY "model evidence": which pixels the trained network's
final convolutional layer weighted most heavily for the predicted class. It
must never be presented by the UI as a measurement of the cell itself (that is
the morphology engine's job, src/morphology/features.py) or as general
scientific knowledge about the cell type (that is static reference text) --
spec section 8 requires these three sources to stay visibly distinct.
"""
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class GradCAMResult:
    heatmap: np.ndarray          # HxW, values in [0, 1], same size as input image
    predicted_class_index: int
    target_layer_name: str


class GradCAM:
    """Grad-CAM for a torchvision CNN classifier (works with MobileNetV3 and
    ResNet backbones as produced by scripts/train_classifier.py)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activation)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_index: int = None) -> GradCAMResult:
        """input_tensor: (1, C, H, W), already normalized as at training time."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_index is None:
            class_index = int(output.argmax(dim=1).item())

        self.model.zero_grad()
        score = output[0, class_index]
        score.backward(retain_graph=False)

        gradients = self._gradients[0]          # (C, h, w)
        activations = self._activations[0]       # (C, h, w)
        weights = gradients.mean(dim=(1, 2))     # (C,) -- global-average-pooled gradients

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * activations[c]
        cam = F.relu(cam)

        cam = cam - cam.min()
        max_val = cam.max()
        if max_val > 0:
            cam = cam / max_val

        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        return GradCAMResult(
            heatmap=cam_resized,
            predicted_class_index=class_index,
            target_layer_name=str(self.target_layer.__class__.__name__),
        )

    def close(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def get_default_target_layer(model: torch.nn.Module, backbone: str) -> torch.nn.Module:
    """Returns the last convolutional block -- the conventional Grad-CAM target."""
    if backbone == "mobilenet_v3_small":
        return model.features[-1]
    if backbone == "resnet18":
        return model.layer4[-1]
    raise ValueError(f"No default Grad-CAM target layer configured for backbone: {backbone}")
