# C4 — vision

- Date: 2026-10-29
- Status: **NO-GO — not implemented**

The 328 MB bf16 vision tower (14 layers, gated 3×3 conv + bidirectional
attention) and the 50 MB projector are loaded by the safetensors loader but are
never executed. Image inputs are not accepted by the API.

Not done: ImageIO decode, smart_resize/bicubic, patchify in merge-block order,
axial RoPE, flash bidirectional attention, projector, image-token splice.
