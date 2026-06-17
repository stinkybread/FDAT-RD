# FDAT-RD

`fdat_rd` (Fast Dual-Attention Transformer, Rectangular + Dictionary) is an
evolution of **FDAT** for cel-animation super-resolution (DVD/LD → BluRay),
targeting TensorRT-static deployment via traiNNer-redux. It keeps every FDAT
deployment invariant (static ONNX export, tensor-core alignment, reshape-over-
view, no data-dependent control flow) while measurably improving recovery on the
hardest frames — thin lineart, sub-pixel chroma in tight structures, eye/hair
curves — the cases conventional models blur.

## Lineage

```
FDAT
 └─ rectangular alternating windows  (anisotropic lineart prior, resolution-aligned)
     └─ token-dictionary cross-attention  (global learned structure prior)
         └─ FDAT-RD
```

FDAT is the body this builds on: residual groups alternating a windowed
**spatial** attention and a transposed **channel** (MDTA-style) attention, each
fused with a depthwise-conv branch through the AIM gate and followed by a
depthwise-mix FFN, with an unshuffle frontend and a `UniUpsampleV3` tail. It is
fast and exports cleanly, but two limitations surface on the hardest ~10% of
frames:

1. **Square windows are isotropic.** The extended thin structures that define
   cel art are poorly served by a square box, and the box rarely tiles the
   production feature grid, so inference wastes compute on window padding.
2. **All attention is local or per-channel.** No path lets a pixel query
   semantically related structure elsewhere in the frame — and that selectivity,
   not receptive field, is what separates strong restorers on hard cases.

FDAT-RD addresses each with one change.

### Evolution 1 — Rectangular alternating windows

Spatial blocks use a rectangular window (default `10×30`) whose orientation
alternates `10×30 / 30×10` across consecutive spatial blocks, giving each
position long-axis coverage along both axes over a pair of blocks — an
anisotropic prior toward extended line structure. The window also tiles the
`720×540` resolution pyramid exactly (everything is a multiple of `30`), so
production inference is window-aligned. A single upfront reflect-pad to
`unshuffle × lcm(split_size)` keeps the body aligned at **any** input size, so
the per-block attention pad is always a no-op and there is no boundary seam.

### Evolution 2 — Token-dictionary cross-attention

A third block type carries a fixed-size learned dictionary of typical cel
structures and lets every pixel query it: the dictionary first refines against
the current image, then image features are enhanced against the refined
dictionary. This is the deployable subset of ATD — fixed-`M` cross-attention
only, **no** dynamic category grouping — so it stays static-shape and `O(N·M)`
linear, hence TRT-clean. It is the architectural generalization of a hand-built
edge/lineart prior: instead of coding the prior, the dictionary learns the
structure bank from the data. Dictionary blocks are lean (no conv/AIM branch);
the global prior is the point and is not diluted with redundant local mixing.

## What was tried and reverted

A Swin-style **relative-position-bias table** replacing the dense `(nh, N, N)`
spatial bias (~98% fewer params, translation-invariant) was tested and
**reverted**: the dense bias's extra capacity was doing real work on the hard
cases, and the spatial blocks feed the dictionary, so starving them measurably
hurt detail recovery at matched iterations. Parameter efficiency is not the
objective; tail performance is. The dense bias stays.

## Kept optimizations

- **SDPA** (FlashAttention / mem-efficient) in the spatial and dictionary
  attention — identical math to manual softmax, lower training VRAM and faster,
  with the gain scaling as crops grow. The standalone converter and the spandrel
  arch use manual softmax for the cleanest export graph; weights are identical.
- **Reflect → replicate pad fallback** so tiny inputs don't crash.
- **Aligned dimensions** — `head_dim 32`, dictionary size `M` a multiple of 32.

## Variants

| variant          | embed_dim | heads | groups | M   | blocks/group |
|------------------|-----------|-------|--------|-----|--------------|
| `fdat_rd_small`  | 96        | 3     | 4      | 64  | 6            |
| `fdat_rd_medium` | 128       | 4     | 4      | 128 | 6            |
| `fdat_rd_large`  | 192       | 6     | 6      | 256 | 6            |

`fdat_rd_aligned` is an alias of `medium`. All use `split_size (10,30)`,
`group_block_pattern [spatial, channel, dictionary]`, and optional
`use_checkpoint` (train-only, no-op at eval/export).

## Training notes

- `split_size (10,30)` → `lcm 30`. With `2× + unshuffle` (feature = lq/2), train
  at **`lq_size 120`** (feature `60×60`, divisible by both 10 and 30 → zero
  window padding).
- The window token count `10×30 = 300` is not a multiple of 8, so TRT tile-pads
  the sequence dim — a known, accepted trade for the anisotropic prior.
- `use_checkpoint` is the VRAM lever when scaling to larger crops or `large`.

```yaml
network_g:
  type: fdat_rd_medium
  scale: 2
  unshuffle_mod: true
  use_checkpoint: true   # optional
# lq_size: 120
```

## Deployment

- **ONNX/TRT:** `fdat_rd_converter.py` (standalone, autodetects all dims; pass
  `--split-size 10 30`). Verify tiny first, then export at production size on
  GPU:
  ```
  python fdat_rd_converter.py model.safetensors -f onnx-static --input-size 60 60
  python fdat_rd_converter.py model.safetensors -f onnx-static --input-size 540 720 --device cuda --no-verify
  ```
- **chaiNNer / spandrel:** `fdat_rd.py` (arch) + `__init__.py` (`FDATRDArch`),
  drop into `spandrel_extra_arches`. Detection keys on the dictionary parameter,
  so it is distinct from `fdat2`/`dat2rt2`. Two notes:
  - **Register `FDATRDArch` before `FDATArch`.** A `fdat_rd` checkpoint also
    satisfies FDAT's current detect (its first block is spatial), so first-match
    order must put `fdat_rd` ahead — or add a `not has dictionary` guard to
    FDAT's detect.
  - The `split_size` factorization is not stored in weights; the spandrel `load`
    assumes the standard `(10, 30)`. Add a registered buffer if you ever vary it.

## Results (vs FDAT, cel-animation data)

Faster and lighter to train, clearly better on hard-case crops, at a modest
inference cost:

- training throughput up (`fdat_rd_medium` ~1.8 it/s vs FDAT ~1.4 it/s, batch 16)
- training VRAM down (~13.3 GB vs ~13.8 GB)
- clearly improved detail recovery across crops (thin lineart, eye/hair)
- ~20% slower at inference — the price of the `300`-token rectangular windows
  that don't tile to a multiple of 8; best evaluated against models at the same
  latency tier rather than against FDAT directly.
