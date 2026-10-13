# Copyright (c) 2026 SparSTAR Authors
# SPDX-License-Identifier: MIT
"""
[SparSTAR] 720p 5 s VBench generation used for Tables 1 and 2 of the paper (dense InfinityStar or SparSTAR).

T2V : 944 unique VBench prompts x 5 seeds (41..45). The model input is the GPT-refined prompt from
      evaluation/VBench_rewrited_prompt.json; files are named after the official VBench prompt
      ("{prompt_en}-{sample}.mp4") so the standard VBench evaluation can be run on the output folder.
I2V : 355 VBench-I2V images (16:9 crop) x 1 seed (41), named "{image name}-{sample}.mp4".
Samples are addressed by a flat index  idx = item_index * nsamples + sample_index,  and finished
samples are skipped, so a run can be split with --start/--end and resumed.

  # dense InfinityStar
  python tools/gen_vbench_720p.py --task t2v --out_dir output/vbench/t2v_dense
  # SparSTAR (Table 1)
  python tools/gen_vbench_720p.py --task t2v --out_dir output/vbench/t2v_sparstar --sparstar 1
  # + Guidance-off / + Scale-skip (Table 2)
  python tools/gen_vbench_720p.py --task t2v --out_dir output/vbench/t2v_goff --sparstar 1 --sparstar_guidance_off 26
  python tools/gen_vbench_720p.py --task t2v --out_dir output/vbench/t2v_skip --sparstar 1 --sparstar_scale_skip 29
  # I2V: same flags with --task i2v
"""
import os
import os.path as osp
import sys
import json
import time
import random
import argparse

import cv2
import numpy as np
import torch
from PIL import Image
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from tools.run_infinity import gen_one_example, load_tokenizer, load_transformer, load_visual_tokenizer, save_video, transform
from infinity.models.self_correction import SelfCorrection
from infinity.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
from infinity.schedules import get_encode_decode_func
from infinity.utils.arg_util import Args

REPO = osp.dirname(osp.dirname(osp.abspath(__file__)))


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['t2v', 'i2v'], required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--checkpoints_dir', default='./')
    parser.add_argument('--start', type=int, default=0, help='first flat index')
    parser.add_argument('--end', type=int, default=-1, help='last flat index (exclusive); -1: all')
    parser.add_argument('--nsamples', type=int, default=-1, help='samples per prompt/image; -1: 5 for T2V, 1 for I2V (paper setting)')
    parser.add_argument('--seed_base', type=int, default=41)
    parser.add_argument('--vbench_i2v_dir', default='data/vbench_i2v', help='folder with i2v-bench-info.json and crop/16-9/')
    parser.add_argument('--warmup', type=int, default=1, help='1: generate the first sample once before timing (compiles the FlexAttention kernels)')
    # SparSTAR options (see infinity/utils/arg_util.py)
    parser.add_argument('--sparstar', type=int, default=0)
    parser.add_argument('--sparstar_density', default='26:0.625,27:0.1875,28:0.1875,29:0.09375')
    parser.add_argument('--sparstar_guidance_off', type=int, default=-1)
    parser.add_argument('--sparstar_scale_skip', default='')
    return parser


def build_args(cli):
    # 720p 5 s inference configuration of tools/infer_video_720p.py
    args = Args()
    args.pn = '0.90M'
    args.fps = 16
    args.video_frames = 81
    args.model_path = osp.join(cli.checkpoints_dir, 'infinitystar_8b_720p_weights')
    args.checkpoint_type = 'torch_shard'
    args.vae_path = osp.join(cli.checkpoints_dir, 'infinitystar_videovae.pth')
    args.text_encoder_ckpt = osp.join(cli.checkpoints_dir, 'text_encoder/flan-t5-xl-official/')
    args.model_type = 'infinity_qwen8b'
    args.text_channels = 2048
    args.dynamic_scale_schedule = 'infinity_elegant_clip20frames_v2'
    args.bf16 = 1
    args.use_apg = 1
    args.use_cfg = 0
    args.cfg = 34
    args.tau_image = 1
    args.tau_video = 0.4
    args.apg_norm_threshold = 0.05
    args.image_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]'
    args.video_scale_repetition = '[3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 1, 1]'
    args.append_duration2caption = 1
    args.use_two_stage_lfq = 1
    args.detail_scale_min_tokens = 750
    args.semantic_scales = 12
    args.max_repeat_times = 10000
    args.enable_rewriter = 0
    args.context_from_largest_no = 1
    # SparSTAR
    args.sparstar = cli.sparstar
    args.sparstar_density = cli.sparstar_density
    args.sparstar_guidance_off = cli.sparstar_guidance_off
    args.sparstar_scale_skip = cli.sparstar_scale_skip
    return args


def safe_name(name):
    return name.replace('/', '_')


def load_items(cli):
    """List of (file name stem, model input text, reference image path or None)."""
    items = []
    if cli.task == 't2v':
        seen = set()
        for d in json.load(open(osp.join(REPO, 'evaluation/VBench_rewrited_prompt.json'))):
            if d['prompt_en'] not in seen:  # 946 entries -> 944 unique prompts
                seen.add(d['prompt_en'])
                items.append((safe_name(d['prompt_en']), d['refined_prompt'], None))
    else:
        for d in json.load(open(osp.join(cli.vbench_i2v_dir, 'i2v-bench-info.json'))):
            image_path = osp.join(cli.vbench_i2v_dir, 'crop/16-9', d['file_name'])
            if osp.exists(image_path):
                items.append((safe_name(d['file_name'][:-4]), f"{d['caption']}, Close-up on big objects, emphasize scale and detail", image_path))
    return items


class Generator:
    def __init__(self, args):
        self.args = args
        self.text_tokenizer, self.text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        self.vae = load_visual_tokenizer(args).float().to('cuda')
        self.infinity = load_transformer(self.vae, args)
        self.self_correction = SelfCorrection(self.vae, args)
        self.video_encode, _, self.get_visual_rope_embeds, get_scale_pack_info = get_encode_decode_func(args.dynamic_scale_schedule)
        self.dynamic_resolution_h_w, h_div_w_templates = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 0.571))]
        self.scale_schedule = self.dynamic_resolution_h_w[h_div_w_template][args.pn]['pt2scale_schedule'][(args.video_frames - 1) // 4 + 1]
        args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(self.scale_schedule)
        args.tower_split_index = args.first_full_spatial_size_scale_index + 1
        self.context_info = get_scale_pack_info(self.scale_schedule, args.first_full_spatial_size_scale_index, args)
        num_scales = len(self.scale_schedule)
        self.tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (num_scales - args.tower_split_index)
        self.cfg_list = [args.cfg] * num_scales
        self.tgt_h, self.tgt_w = self.scale_schedule[-1][1] * 16, self.scale_schedule[-1][2] * 16

    def encode_reference_image(self, image_path, seed):
        # The reference-image encoder draws from the global RNGs, so they are seeded for reproducible I2V outputs.
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        ref_img_T3HW = torch.stack([transform(Image.fromarray(cv2.imread(image_path)[:, :, ::-1]).convert("RGB"), self.tgt_h, self.tgt_w)], 0)
        ref_img_bcthw = ref_img_T3HW.permute(1, 0, 2, 3).unsqueeze(0)
        _, _, gt_ls_Bl, _, _, _ = self.video_encode(self.vae, ref_img_bcthw.cuda(), vae_features=None, self_correction=self.self_correction,
                                                    args=self.args, infer_mode=True, dynamic_resolution_h_w=self.dynamic_resolution_h_w)
        return gt_ls_Bl

    def __call__(self, text, image_path, seed):
        gt_leak, gt_ls_Bl = -1, None
        if image_path is not None:
            gt_ls_Bl = self.encode_reference_image(image_path, seed)
            gt_leak = self.args.tower_split_index
        torch.cuda.synchronize(); start = time.time()
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
            video, _ = gen_one_example(
                self.infinity, self.vae, self.text_tokenizer, self.text_encoder, '<<<t=5s>>>' + text,
                negative_prompt='', g_seed=seed, gt_leak=gt_leak, gt_ls_Bl=gt_ls_Bl, cfg_list=self.cfg_list, tau_list=self.tau,
                scale_schedule=self.scale_schedule, cfg_insertion_layer=[0], vae_type=self.args.vae_type, sampling_per_bits=1,
                enable_positive_prompt=0, low_vram_mode=True, args=self.args, get_visual_rope_embeds=self.get_visual_rope_embeds,
                context_info=self.context_info, noise_list=None)
        torch.cuda.synchronize(); latency = time.time() - start
        video = video.detach().float().cpu().numpy() if video.ndim == 4 else video.unsqueeze(0).detach().float().cpu().numpy()
        for b in self.infinity.unregistered_blocks:
            b.attn.cached_k, b.attn.cached_v = {}, {}
        torch.cuda.empty_cache()
        return np.clip(video, 0, 255).astype(np.uint8), latency


def main():
    cli = get_parser().parse_args()
    nsamples = cli.nsamples if cli.nsamples > 0 else (5 if cli.task == 't2v' else 1)
    items = load_items(cli)
    end = len(items) * nsamples if cli.end < 0 else min(cli.end, len(items) * nsamples)
    os.makedirs(osp.join(cli.out_dir, 'mp4'), exist_ok=True)
    print(f'[{cli.task}] {len(items)} items x {nsamples} samples, range [{cli.start}, {end}), sparstar={cli.sparstar} '
          f'density={cli.sparstar_density} guidance_off={cli.sparstar_guidance_off} scale_skip={cli.sparstar_scale_skip!r}', flush=True)

    args = build_args(cli)
    generate = Generator(args)
    todo = []
    for idx in range(cli.start, end):
        item_idx, sample_idx = divmod(idx, nsamples)
        name, text, image_path = items[item_idx]
        mp4 = osp.join(cli.out_dir, 'mp4', f'{name}-{sample_idx}.mp4')
        if not osp.exists(mp4):
            todo.append((idx, text, image_path, cli.seed_base + sample_idx, mp4))
    if cli.warmup and todo:
        generate(*todo[0][1:4])
    for idx, text, image_path, seed, mp4 in todo:
        video, latency = generate(text, image_path, seed)
        save_video(video, fps=args.fps, save_filepath=mp4 + '.tmp.mp4')
        os.replace(mp4 + '.tmp.mp4', mp4)
        with open(osp.join(cli.out_dir, 'latency.jsonl'), 'a') as f:
            f.write(json.dumps({'idx': idx, 'seed': seed, 'latency': latency, 'mp4': osp.basename(mp4)}) + '\n')
        print(f'[{idx:05d}] {latency:.1f}s {osp.basename(mp4)}', flush=True)


if __name__ == '__main__':
    main()
