# SparSTAR: Sparse Attention for SpaceTime AutoRegressive Video Synthesis

<p align="center">
  <a href="https://arxiv.org/abs/2608.10519">📄 Paper</a> &nbsp;|&nbsp;
  <a href="https://jigsaw0612.github.io/SparSTAR_project_page/">🌐 Project Page</a>
</p>

<p align="center">
  Jongbeom Lee<sup>*</sup>, Hyunwoo Yu<sup>*</sup>, Jincheol Yang, Jaemin Choi, Suk-Ju Kang<sup>†</sup><br>
  Sogang University &nbsp;·&nbsp; <sup>*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author
</p>

![SparSTAR teaser](assets/sparstar/teaser.png)

## 🔥 News

- **[2026/10/13]** Code for 720p and 480p (5 s) text-to-video and image-to-video generation is released!

## 📚 Overview

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

<p align="center">
  <img src="assets/sparstar/method_overview.png" width="90%">
</p>

## 📈 Results

**720p text-to-video and image-to-video** (PSNR / SSIM / LPIPS are measured against the dense InfinityStar output with the same prompt and seed)

<p align="center"><img src="assets/sparstar/table1_720p.png" width="95%"></p>

**Combination with Scale-skip and Guidance-off (720p)**

<p align="center"><img src="assets/sparstar/table2_compat.png" width="95%"></p>

**480p text-to-video and image-to-video (5 s)**

<p align="center"><img src="assets/sparstar/table3_480p_5s.png" width="95%"></p>

Video comparisons with dense InfinityStar are on the [project page](https://jigsaw0612.github.io/SparSTAR_project_page/).

## 🛠️ Installation

### Environment

SparSTAR uses the same environment as InfinityStar and needs no additional package.

```bash
git clone https://github.com/JIGSAW0612/SparSTAR.git
cd SparSTAR

conda create -n sparstar python=3.11 -y
conda activate sparstar
pip install -r requirements.txt                          # includes torch==2.5.1
pip install packaging psutil ninja                       # build requirements of flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation  # dense attention at the non-sparse scales
```

We recommend CUDA 12.4 + PyTorch 2.5.1 (pinned in `requirements.txt`; SparSTAR uses its FlexAttention API).

### Checkpoints

SparSTAR is training-free and runs the released InfinityStar checkpoints from Hugging Face 🤗 ([FoundationVision/InfinityStar](https://huggingface.co/FoundationVision/InfinityStar)).

| Component | Path in the Hugging Face repo |
|---|---|
| 720p model (5 s) | `infinitystar_8b_720p_weights/` |
| 480p model (5 s) | `infinitystar_8b_480p_weights/` |
| Video VAE | `infinitystar_videovae.pth` |
| Text encoder | `text_encoder/flan-t5-xl-official/` |

```bash
hf download FoundationVision/InfinityStar --revision 3480598274c0c87c342a24d201fa3549506b5be5 \
    --include "infinitystar_8b_720p_weights/*" "infinitystar_8b_480p_weights/*" \
              "text_encoder/*" "infinitystar_videovae.pth" \
    --local-dir ./checkpoints
```

Download only the resolution you need. The commands below assume the checkpoints are in `./checkpoints`.

## 🚀 Inference

SparSTAR runs **by default**; add `--dense` to run dense InfinityStar.

### 720p

```bash
# Text-to-Video
python tools/infer_video_720p.py --checkpoints_dir ./checkpoints --image_path '' \
    --prompt "A corgi running on the beach at sunset" --save_path output/720p_t2v.mp4

# Image-to-Video
python tools/infer_video_720p.py --checkpoints_dir ./checkpoints \
    --image_path assets/reference_image.webp --save_path output/720p_i2v.mp4

# + Guidance-off / + Scale-skip (Table 2)
python tools/infer_video_720p.py --checkpoints_dir ./checkpoints --sparstar_guidance_off 26
python tools/infer_video_720p.py --checkpoints_dir ./checkpoints --sparstar_scale_skip 29

# Dense InfinityStar
python tools/infer_video_720p.py --checkpoints_dir ./checkpoints --dense
```

### 480p

```bash
# Text-to-Video / Image-to-Video
python tools/infer_video_480p.py --checkpoints_dir ./checkpoints --image_path '' \
    --prompt "A corgi running on the beach at sunset" --save_path output/480p_t2v.mp4
python tools/infer_video_480p.py --checkpoints_dir ./checkpoints \
    --image_path assets/reference_image.webp --save_path output/480p_i2v.mp4
```

| Option | Default | Description |
|---|---|---|
| `--dense` | off | Dense InfinityStar attention instead of SparSTAR |
| `--sparstar_density` | paper setting | `scale:ratio` pairs: the sparse scales and the fraction of current-scale key blocks kept at each. Default: `26:0.625,27:0.1875,28:0.1875,29:0.09375` at 720p, `24:0.625,25:0.1875,26:0.1875,27:0.09375` at 480p |
| `--sparstar_guidance_off` | `-1` (off) | 720p. Scale index from which the unconditional CFG branch is dropped (`26` in Table 2) |
| `--sparstar_scale_skip` | `''` (off) | 720p. Video scales that are skipped (`29` in Table 2) |

> [!NOTE]
> - The first SparSTAR generation in a process also compiles the FlexAttention kernels, so measure speed from the second generation on.
> - A 720p generation needs about 70 GB of GPU memory. The paper's speedups are end-to-end times on a single H100 NVL GPU.

## 📊 Evaluation

`tools/gen_vbench.py` generates the VBench videos used in the paper. For I2V, download the [VBench-I2V](https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v) images into `data/vbench_i2v/` first.

```bash
# 720p (Tables 1 and 2)
python tools/gen_vbench.py --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/720p_t2v_sparstar
python tools/gen_vbench.py --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/720p_t2v_dense --dense
python tools/gen_vbench.py --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/720p_t2v_goff --sparstar_guidance_off 26
python tools/gen_vbench.py --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/720p_t2v_skip --sparstar_scale_skip 29

# 480p (Supplementary Table 7)
python tools/gen_vbench.py --resolution 480p --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/480p_t2v_sparstar
python tools/gen_vbench.py --resolution 480p --task t2v --checkpoints_dir ./checkpoints --out_dir output/vbench/480p_t2v_dense --dense

# I2V: the same commands with --task i2v
```

Runs can be split across GPUs with `--start/--end`, and finished videos are skipped when a run is restarted.
Run the official [VBench](https://github.com/Vchitect/VBench) / VBench-I2V evaluation on `<out_dir>/mp4`.

## 🧩 Implementation

The main implementation of SparSTAR is highlighted with `[SparSTAR]` annotations, mainly in

| File | Role |
|---|---|
| [`infinity/models/sparstar.py`](infinity/models/sparstar.py) | Default sparse scales ([`configure`](infinity/models/sparstar.py#L62)), block selection and forward-only block-sparse attention ([`block_sparse_attention`](infinity/models/sparstar.py#L109)) |
| [`infinity/models/basic.py`](infinity/models/basic.py#L257) | Calls SparSTAR instead of dense attention at the sparse scales |
| [`infinity/models/infinity.py`](infinity/models/infinity.py#L580) | Reads the SparSTAR options per generation; [+Guidance-off and +Scale-skip](infinity/models/infinity.py#L619) |
| [`infinity/utils/arg_util.py`](infinity/utils/arg_util.py#L205) | SparSTAR options |
| [`tools/infer_video_720p.py`](tools/infer_video_720p.py#L142), [`tools/infer_video_480p.py`](tools/infer_video_480p.py#L158) | Command-line options |
| [`tools/gen_vbench.py`](tools/gen_vbench.py) | VBench generation |

## 🔗 BibTeX

If you find SparSTAR useful for your research and applications, please cite our work using this BibTeX:

```bibtex
@article{lee2026sparstar,
  title   = {SparSTAR: Sparse Attention for SpaceTime AutoRegressive Video Synthesis},
  author  = {Lee, Jongbeom and Yu, Hyunwoo and Yang, Jincheol and Choi, Jaemin and Kang, Suk-Ju},
  journal = {arXiv preprint arXiv:2608.10519},
  year    = {2026}
}
```

## 📜 License

This project is released under the MIT license.

## 🙏 Acknowledgement

We appreciate the open-source efforts of [InfinityStar](https://github.com/FoundationVision/InfinityStar) and PyTorch [FlexAttention](https://pytorch.org/blog/flexattention/).
