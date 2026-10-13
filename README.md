# SparSTAR

### Sparse Attention for SpaceTime AutoRegressive Video Synthesis

Jongbeom Lee<sup>\*</sup>, Hyunwoo Yu<sup>\*</sup>, Jincheol Yang, Jaemin Choi, Suk-Ju Kang<sup>†</sup>

Sogang University &nbsp;·&nbsp; <sup>\*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author

[**Project Page**](https://jigsaw0612.github.io/SparSTAR_project_page/) &nbsp;|&nbsp; [**arXiv**](https://arxiv.org/abs/2608.10519) &nbsp;|&nbsp; [**Paper (PDF)**](https://arxiv.org/pdf/2608.10519)

![SparSTAR teaser](https://jigsaw0612.github.io/SparSTAR_project_page/static/images/teaser.png)

---

## Abstract

InfinityStar extends visual autoregressive generation to video through a sequence of image and clip
pyramids. Its changing scale and cross-clip context, however, leave late-stage attention costly and make
sparse patterns reused from diffusion or image VAR models unreliable. We introduce **SparSTAR**, a
training-free block-sparse attention method tailored to this setting. At each expensive scale and attention
head, SparSTAR scores contiguous key blocks from the current query and key activations, retains required
conditioning context, and executes the selected blocks through a forward-only sparse path. We analyze
cross-scale consistency within a clip, pattern persistence across clip boundaries, and quality degradation
as reuse spans increasingly distant scales. Across these analyses, important key blocks shift, showing that
recomputing block selection at each target scale is more reliable than reusing a transferred mask. On 720p
text-to-video and image-to-video generation, SparSTAR preserves every token and refinement scale while
providing about a **1.6× end-to-end speedup** and maintaining VBench and paired-output reconstruction
fidelity close to dense InfinityStar.

## About this repository

This repository is the official [InfinityStar](https://github.com/FoundationVision/InfinityStar) codebase
(commit [`7465753`](https://github.com/FoundationVision/InfinityStar/commit/7465753cef02b206530b380c15f583db534c828f))
with SparSTAR added on top. SparSTAR is training-free and inference-only: it runs the released InfinityStar
checkpoints as they are. The InfinityStar code is otherwise unchanged; the commit
`Import InfinityStar (FoundationVision/InfinityStar@7465753)` is an exact copy of upstream, so

```bash
git diff d5c7c2a HEAD
```

shows everything SparSTAR adds. This release covers 720p 5 s text-to-video and image-to-video generation (Tables 1 and 2 of the paper).

## Where is SparSTAR in the code?

The main implementation of SparSTAR is highlighted with `[SparSTAR]` annotations, mainly in

| File | Role |
|---|---|
| [`infinity/models/sparstar.py`](infinity/models/sparstar.py) | **New.** Block selection and forward-only block-sparse attention ([`block_sparse_attention`](infinity/models/sparstar.py#L92)) |
| [`infinity/models/basic.py`](infinity/models/basic.py#L257) | `SelfAttention.forward` calls SparSTAR instead of dense flash-attention at the sparse scales |
| [`infinity/models/infinity.py`](infinity/models/infinity.py#L561) | `ar_infer_infinity_elegant` reads the options of each generation and implements the optional [+Guidance-off and +Scale-skip](infinity/models/infinity.py#L618) of Table 2 |
| [`infinity/utils/arg_util.py`](infinity/utils/arg_util.py#L205) | SparSTAR options (`sparstar*` fields of `Args`) |
| [`tools/infer_video_720p.py`](tools/infer_video_720p.py#L142) | Command-line flags for the 720p demo |
| [`tools/gen_vbench_720p.py`](tools/gen_vbench_720p.py) | **New.** VBench generation protocol of Tables 1 and 2 |

At every sparse scale (by default the last four scales of the 720p video clip, 26–29), and for every attention head:

1. Queries and keys are split into contiguous blocks of 128 tokens, and every (query block, key block) pair is scored by the dot product of the two block means.
2. The KV-cached conditioning prefix (reference scales and text) is always kept.
3. Among the key blocks of the current scale, the top ⌈ρ<sub>s</sub> · N<sub>s</sub>⌉ are kept, where ρ<sub>s</sub> is the density of scale *s* and N<sub>s</sub> its number of key blocks.
4. FlexAttention runs on the selected blocks only. The block mask is built for the forward pass only (the backward-only index tables are skipped).

The selection is recomputed at every scale and repetition step; no mask is reused. All other scales use dense attention, and every token and refinement scale is still generated.

## Setup

**Environment.** Follow InfinityStar:

```bash
pip3 install -r requirements.txt
pip3 install flash-attn --no-build-isolation   # dense attention of InfinityStar
```

The paper results were produced with Python 3.11, PyTorch 2.5.1 (CUDA 12.4) and flash-attn 2.7.4.post1 on NVIDIA H100 NVL GPUs.
SparSTAR builds FlexAttention `BlockMask` objects directly and therefore expects PyTorch 2.5.1, the version pinned in `requirements.txt`.

**Checkpoints.** SparSTAR uses the released InfinityStar 720p checkpoints:

```bash
hf download FoundationVision/InfinityStar --revision 3480598274c0c87c342a24d201fa3549506b5be5 \
    --include "infinitystar_8b_720p_weights/*" --include "text_encoder/*" --include "infinitystar_videovae.pth" \
    --local-dir ./checkpoints
```

```
checkpoints/
├── infinitystar_8b_720p_weights/
├── infinitystar_videovae.pth
└── text_encoder/flan-t5-xl-official/
```

## Inference

`tools/infer_video_720p.py` is the InfinityStar 720p demo with SparSTAR flags added. Without flags it runs dense InfinityStar.

```bash
# Text-to-video
python3 tools/infer_video_720p.py --checkpoints_dir ./checkpoints --image_path '' \
    --prompt "A corgi running on the beach at sunset" --sparstar 1 --save_path output/t2v_sparstar.mp4

# Image-to-video (default reference image: assets/reference_image.webp)
python3 tools/infer_video_720p.py --checkpoints_dir ./checkpoints --sparstar 1 --save_path output/i2v_sparstar.mp4

# Table 2 add-ons
python3 tools/infer_video_720p.py --checkpoints_dir ./checkpoints --sparstar 1 --sparstar_guidance_off 26   # + Guidance-off
python3 tools/infer_video_720p.py --checkpoints_dir ./checkpoints --sparstar 1 --sparstar_scale_skip 29     # + Scale-skip
```

| Option | Default | Meaning |
|---|---|---|
| `--sparstar` | `0` | `1` enables block-sparse attention |
| `--sparstar_density` | `26:0.625,27:0.1875,28:0.1875,29:0.09375` | `scale:ρ` pairs: the sparse scales and the fraction of current-scale key blocks kept at each (paper setting) |
| `--sparstar_guidance_off` | `-1` | Scale index from which the unconditional CFG branch is dropped (`26` in Table 2); `-1` disables it |
| `--sparstar_scale_skip` | `''` | Video scales that are not run (`29` in Table 2) |

The first SparSTAR generation in a process also compiles the FlexAttention kernels, so time the second and later generations.
A 720p generation needs about 70 GB of GPU memory. The speedups in the paper are end-to-end wall-clock times on a single H100 NVL GPU that no other job was using.

## Reproducing Tables 1 and 2

`tools/gen_vbench_720p.py` generates the 720p 5 s VBench videos with the protocol of the paper.

- **T2V.** 944 unique VBench prompts × 5 seeds (41–45). The GPT-refined prompts in `evaluation/VBench_rewrited_prompt.json` are the model input, and files are named after the official VBench prompts.
- **I2V.** 355 VBench-I2V images (16:9 crop), seed 41. Put `i2v-bench-info.json` and the cropped images (`crop/16-9/`) from [VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) under `data/vbench_i2v/`.

```bash
CKPT=./checkpoints
for TASK in t2v i2v; do
  python3 tools/gen_vbench_720p.py --task $TASK --checkpoints_dir $CKPT --out_dir output/vbench/${TASK}_dense
  python3 tools/gen_vbench_720p.py --task $TASK --checkpoints_dir $CKPT --out_dir output/vbench/${TASK}_sparstar --sparstar 1
  python3 tools/gen_vbench_720p.py --task $TASK --checkpoints_dir $CKPT --out_dir output/vbench/${TASK}_goff      --sparstar 1 --sparstar_guidance_off 26
  python3 tools/gen_vbench_720p.py --task $TASK --checkpoints_dir $CKPT --out_dir output/vbench/${TASK}_skip      --sparstar 1 --sparstar_scale_skip 29
done
```

Samples have a flat index (`item × samples + sample`), so a run can be split across GPUs with `--start/--end`.
Finished videos are skipped, so a stopped run can be restarted with the same command.
Each run writes `mp4/` and `latency.jsonl`. Restarting the process every ~100 videos keeps the memory used by the compiled sparse kernels bounded.

- **VBench.** Run the official [VBench](https://github.com/Vchitect/VBench) / VBench-I2V evaluation on `<out_dir>/mp4`.
- **Fidelity to dense InfinityStar (PSNR / SSIM / LPIPS).** Files with the same name in the dense and SparSTAR folders share the prompt and seed. Compare them frame by frame.

## Citation

```bibtex
@article{lee2026sparstar,
  title   = {SparSTAR: Sparse Attention for SpaceTime AutoRegressive Video Synthesis},
  author  = {Lee, Jongbeom and Yu, Hyunwoo and Yang, Jincheol and Choi, Jaemin and Kang, Suk-Ju},
  journal = {arXiv preprint arXiv:2608.10519},
  year    = {2026}
}
```

SparSTAR is built on InfinityStar; please also cite it:

```bibtex
@misc{InfinityStar,
  title         = {InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation},
  author        = {Jinlai Liu and Jian Han and Bin Yan and Hui Wu and Fengda Zhu and Xing Wang and Yi Jiang and Bingyue Peng and Zehuan Yuan},
  year          = {2025},
  eprint        = {2511.04675},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.04675}
}
```

## Acknowledgement

This code is built on [InfinityStar](https://github.com/FoundationVision/InfinityStar). The sparse kernels use PyTorch [FlexAttention](https://pytorch.org/blog/flexattention/).

## License

This project is released under the MIT License, the license of InfinityStar. See [LICENSE](LICENSE).
