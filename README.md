# SparSTAR

### Sparse Attention for SpaceTime AutoRegressive Video Synthesis

Jongbeom Lee<sup>\*</sup>, Hyunwoo Yu<sup>\*</sup>, Jincheol Yang, Jaemin Choi, Suk-Ju Kang<sup>†</sup>

Sogang University &nbsp;·&nbsp; <sup>\*</sup>Equal contribution &nbsp;·&nbsp; <sup>†</sup>Corresponding author

[**Project Page**](https://jigsaw0612.github.io/SparSTAR_project_page/) &nbsp;|&nbsp; [**arXiv**](https://arxiv.org/abs/2608.10519) &nbsp;|&nbsp; [**Paper (PDF)**](https://arxiv.org/pdf/2608.10519)

![SparSTAR teaser](https://jigsaw0612.github.io/SparSTAR_project_page/static/images/teaser.png)

---

## Code release

**Coming soon.** The implementation is being prepared for release and is not in this repository yet.

Watch or star this repository to be notified when it lands. In the meantime, the
[project page](https://jigsaw0612.github.io/SparSTAR_project_page/) carries the qualitative
comparisons, the analysis figures, and the full result tables.

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

## Citation

```bibtex
@article{lee2026sparstar,
  title   = {SparSTAR: Sparse Attention for SpaceTime AutoRegressive Video Synthesis},
  author  = {Lee, Jongbeom and Yu, Hyunwoo and Yang, Jincheol and Choi, Jaemin and Kang, Suk-Ju},
  journal = {arXiv preprint arXiv:2608.10519},
  year    = {2026}
}
```
