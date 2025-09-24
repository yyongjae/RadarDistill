import _init_path
import sys
sys.path.append('.')
import argparse
import datetime
import os
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import math
from matplotlib.path import Path as MplPath  # [NEW] polygon mask

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ==============================
# 공통 유틸
# ==============================
def world_to_bev_pixel(x, y, pc_range, bev_h, bev_w):
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    u = (x - x_min) / (x_max - x_min + 1e-12)  # X→width
    v = (y - y_min) / (y_max - y_min + 1e-12)  # Y→height
    col = np.clip(u * bev_w, 0, bev_w - 1)
    row = np.clip(v * bev_h, 0, bev_h - 1)
    return row, col  # (row, col)

def box_corners_world(x, y, dx, dy, heading):
    c, s = math.cos(heading), math.sin(heading)
    hx, hy = dx/2.0, dy/2.0
    corners_local = np.array([[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]], dtype=np.float32)
    R = np.array([[c,-s],[s, c]], dtype=np.float32)
    return corners_local @ R.T + np.array([x,y], dtype=np.float32)  # [4,2]

def polygon_mask_in_feature(corners_xy, pc_range, H, W):
    """월드 코너 4점 → feature 픽셀 다각형 마스크 (H,W, bool)."""
    pixels = [world_to_bev_pixel(cx, cy, pc_range, H, W) for cx, cy in corners_xy]
    verts = [(float(col), float(row)) for (row, col) in pixels]
    poly = MplPath(verts)
    cols = [v[0] for v in verts]; rows = [v[1] for v in verts]
    cmin, cmax = int(max(0, math.floor(min(cols)))), int(min(W-1, math.ceil(max(cols))))
    rmin, rmax = int(max(0, math.floor(min(rows)))), int(min(H-1, math.ceil(max(rows))))
    if cmax < cmin or rmax < rmin:
        return None, None, None
    grid_x, grid_y = np.meshgrid(np.arange(cmin, cmax+1), np.arange(rmin, rmax+1))
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (N,2) (x=col,y=row)
    inside = poly.contains_points(pts).reshape(grid_y.shape)
    return inside, (rmin, rmax), (cmin, cmax)

def l2_normalize_rows(t):
    return torch.nn.functional.normalize(t, p=2, dim=1)

# ==============================
# CKA 유틸
# ==============================
def _debiased_dot_product_similarity(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n):
    return (2 * xty - squared_norm_x * sum_squared_rows_y - sum_squared_rows_x * squared_norm_y) / (n * (n-1))

def cka_linear(x, y, debiased=False):
    x = x.cuda(); y = y.cuda()
    xty = torch.dot(x.flatten(), y.flatten())
    if debiased:
        n = x.shape[0]
        sum_squared_rows_x = torch.sum(x * x, dim=1)
        sum_squared_rows_y = torch.sum(y * y, dim=1)
        squared_norm_x = torch.sum(sum_squared_rows_x)
        squared_norm_y = torch.sum(sum_squared_rows_y)
        xty = _debiased_dot_product_similarity(xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y, n)
        xtx = _debiased_dot_product_similarity(torch.dot(x.flatten(), x.flatten()), sum_squared_rows_x, sum_squared_rows_x, squared_norm_x, squared_norm_x, n)
        yty = _debiased_dot_product_similarity(torch.dot(y.flatten(), y.flatten()), sum_squared_rows_y, sum_squared_rows_y, squared_norm_y, squared_norm_y, n)
    else:
        xtx = torch.dot(x.flatten(), x.flatten())
        yty = torch.dot(y.flatten(), y.flatten())
    return (xty / torch.sqrt(xtx * yty)).cpu().item()

def cka_rbf(x, y, debiased=False, sigma=None):
    x = x.cuda(); y = y.cuda()
    if sigma is None:
        sigma = torch.sqrt(0.5 * (torch.median(torch.cdist(x,x)**2) + torch.median(torch.cdist(y,y)**2)))
    
    gram_x = torch.exp(-torch.cdist(x,x)**2 / (2 * sigma**2))
    gram_y = torch.exp(-torch.cdist(y,y)**2 / (2 * sigma**2))
    
    return cka_linear(gram_x, gram_y, debiased)


# ==============================
# 인자 파서
# ==============================
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for teacher model (e.g., pillarnet.yaml)')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='teacher_eval', help='extra tag for this experiment')
    parser.add_argument('--teacher_ckpt', type=str, default=None, help='teacher model checkpoint (.pth file)')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)

    parser.add_argument('--max_waiting_mins', type=int, default=30)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--eval_tag', type=str, default='teacher')
    parser.add_argument('--eval_all', action='store_true', default=False)
    parser.add_argument('--ckpt_dir', type=str, default=None)
    parser.add_argument('--save_to_file', action='store_true', default=False)
    parser.add_argument('--infer_time', action='store_true', default=False)
    parser.add_argument('--cal_params', action='store_true', default=False)

    parser.add_argument('--features_to_analyze', type=str, default=None,
                        help='Comma-separated list of features to analyze, or "all".')

    # [NEW] 유사도/시각화 옵션
    parser.add_argument('--save_class_similarity', action='store_true', default=False,
                        help='전 테스트셋 누적 class-class 유사도 맵 저장')
    parser.add_argument('--save_scene_instance_similarity', action='store_true', default=False,
                        help='scene별 첫 샘플의 instance-level 유사도 맵 저장')
    parser.add_argument('--similarity_pooling', type=str, default='avg',
                        choices=['avg', 'max'], help='bbox 1x1 풀링 방식(avg or max)')
    parser.add_argument('--max_scene_instance_plots', type=int, default=99999,
                        help='instance 맵 저장 최대 개수')
    parser.add_argument('--min_instances_per_scene_plot', type=int, default=2,
                        help='instance 맵 저장 최소 인스턴스 수')
    parser.add_argument('--save_scene_bev_vis', action='store_true', default=True,
                        help='scene별 첫 샘플 저장 시 BEV+GT 시각화도 함께 저장')  # [NEW]

    # (기존 visualize_n_samples 제거)
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    # Correct EXP_GROUP_PATH to be relative to 'cfgs' directory
    path_parts = args.cfg_file.split('/')
    try:
        cfgs_index = path_parts.index('cfgs')
        cfg.EXP_GROUP_PATH = '/'.join(path_parts[cfgs_index + 1:-1])
    except ValueError:
        # If 'cfgs' is not in the path, fall back to original logic but handle it better
        cfg.EXP_GROUP_PATH = '/'.join(path_parts[1:-1]) if len(path_parts) > 2 else ''

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return args, cfg


def load_teacher_model(model, teacher_ckpt_path, logger, to_cpu=False):
    if not os.path.isfile(teacher_ckpt_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_ckpt_path}")
    logger.info('==> Loading teacher model parameters from %s to %s' % (teacher_ckpt_path, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(teacher_ckpt_path, map_location=loc_type)

    if 'model_state' in checkpoint: model_state_disk = checkpoint['model_state']
    elif 'state_dict' in checkpoint: model_state_disk = checkpoint['state_dict']
    else: model_state_disk = checkpoint
    try:
        model.load_state_dict(model_state_disk, strict=True); logger.info('==> loaded with strict=True')
    except RuntimeError as e:
        logger.warning(f'==> strict load failed: {e}')
        model.load_state_dict(model_state_disk, strict=False); logger.info('==> loaded with strict=False')

    version = checkpoint.get("version", None)
    if version is not None: logger.info('==> ckpt version: %s' % version)
    epoch = checkpoint.get("epoch", "unknown")
    logger.info('==> ckpt epoch: %s' % epoch)


def _find_cls_conv_module(model, logger):
    candidate_attr_names = ['dense_head','point_head','roi_head','head','det_head']
    for attr in candidate_attr_names:
        head = getattr(model, attr, None)
        if head is None: continue
        for name, module in head.named_modules():
            lname = name.lower()
            if (('conv_cls' in lname) or ('cls_pred' in lname) or ('cls_head' in lname) or ('cls' in lname and 'conv' in lname)) and hasattr(module, 'weight'):
                logger.info(f"Found classification module at {attr}.{name}")
                return module
    for name, module in model.named_modules():
        lname = name.lower()
        if (('conv_cls' in lname) or ('cls_pred' in lname) or ('cls_head' in lname)) and hasattr(module, 'weight'):
            logger.info(f"Found classification module at {name}"); return module
    logger.warning('Classification conv layer not found.')
    return None

def compute_class_vectors_from_head(model, class_names, logger):
    cls_module = _find_cls_conv_module(model, logger)
    if cls_module is None: return None
    W = cls_module.weight.detach().float().cpu()
    out_channels = W.shape[0]
    num_classes = len(class_names)
    if out_channels % num_classes != 0:
        logger.warning(f'conv_cls out_channels {out_channels} not divisible by num_classes {num_classes}.')
        return None
    num_anchors = out_channels // num_classes
    W_flat = W.view(out_channels, -1)
    W_ank_cls = W_flat.view(num_anchors, num_classes, -1)
    class_vecs = W_ank_cls.mean(dim=0)
    return class_vecs.numpy()

def cosine_similarity_matrix(class_vecs):
    X = class_vecs
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


class BEVSimilarityEngine:
    def __init__(self, feature_name, feature_key_path, class_names, pc_range, logger, pooling='center',
                 save_scene_instance=False, max_scene_plots=99999, min_inst_for_plot=2,
                 save_scene_bev_vis=True):
        self.feature_name = feature_name
        self.feature_key_path = feature_key_path.split('.')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.pc_range = pc_range
        self.logger = logger
        self.pooling = pooling
        self.save_scene_instance = save_scene_instance
        self.max_scene_plots = max_scene_plots
        self.min_inst_for_plot = min_inst_for_plot
        self.save_scene_bev_vis = save_scene_bev_vis

        # Cosine, CKA Linear, CKA RBF
        self.cos_sim_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.cka_linear_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.cka_rbf_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.sim_counts = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)

        self.scene_first_done = set()
        self.scene_payloads = []

    @staticmethod
    def _get_scene_id(meta):
        for k in ['scene_name', 'scene_id', 'sequence_name', 'sequence', 'token', 'log_token']:
            v = meta.get(k, None)
            if v is not None: return str(v)
        return str(meta.get('frame_id', meta.get('sample_idx', 'unknown_scene')))

    def _extract_feature_for_box(self, bev_feat_chw, box8, pooling):
        C, H, W = bev_feat_chw.shape
        x, y, z, dx, dy, dz, heading, cls = box8[:8]
        row, col = world_to_bev_pixel(float(x), float(y), self.pc_range, H, W)
        r_i = int(round(row)); c_i = int(round(col))
        r_i = max(0, min(H-1, r_i)); c_i = max(0, min(W-1, c_i))
        center_pixel_feature = bev_feat_chw[:, r_i, c_i]

        # For avg or max pooling, use polygon mask
        corners_xy = box_corners_world(float(x), float(y), float(dx), float(dy), float(heading))
        mask, (rmin, rmax), (cmin, cmax) = polygon_mask_in_feature(corners_xy, self.pc_range, H, W)
        
        if mask is None or not mask.any():
            return center_pixel_feature

        slice_feat = bev_feat_chw[:, rmin:rmax+1, cmin:cmax+1]
        mask_t = torch.from_numpy(mask.astype(np.bool_)).to(slice_feat.device)
        masked = slice_feat[:, mask_t]  # [C, K]

        if masked.numel() == 0:
            return center_pixel_feature

        if pooling == 'avg':
            return masked.mean(dim=1)
        elif pooling == 'max':
            return torch.max(masked, dim=1).values
        else:
            # Fallback to center pixel if pooling method is not recognized (should not happen with argparse choices)
            return center_pixel_feature

    def _accumulate_class_sim(self, feats_C, labels_0b):
        if feats_C.shape[0] < 2: return
        
        # 1. Cosine Similarity
        P_norm = l2_normalize_rows(feats_C)
        S_cos = (P_norm @ P_norm.T).detach().cpu().numpy()
        
        # 2. CKA Similarities
        N = feats_C.shape[0]
        S_cka_linear = np.zeros((N, N), dtype=np.float32)
        S_cka_rbf = np.zeros((N, N), dtype=np.float32)

        for i in range(N):
            for j in range(i, N):
                f_i = feats_C[i:i+1]
                f_j = feats_C[j:j+1]
                if i == j:
                    S_cka_linear[i, j] = 1.0
                    S_cka_rbf[i, j] = 1.0
                else:
                    lin_cka = cka_linear(f_i, f_j, debiased=True)
                    rbf_cka = cka_rbf(f_i, f_j, debiased=True)
                    S_cka_linear[i, j] = S_cka_linear[j, i] = lin_cka
                    S_cka_rbf[i, j] = S_cka_rbf[j, i] = rbf_cka

        L = labels_0b.detach().cpu().numpy()
        for i in range(N):
            ci = L[i]
            if not (0 <= ci < self.num_classes): continue
            for j in range(N):
                if i == j: continue
                cj = L[j]
                if 0 <= cj < self.num_classes:
                    self.cos_sim_sums[ci, cj] += float(S_cos[i, j])
                    self.cka_linear_sums[ci, cj] += float(S_cka_linear[i, j])
                    self.cka_rbf_sums[ci, cj] += float(S_cka_rbf[i, j])
                    self.sim_counts[ci, cj] += 1.0

    def _maybe_capture_scene_first(self, scene_id, feats_C, labels_0b, bev_i, boxes_i, meta_i):
        if (not self.save_scene_instance) or (scene_id in self.scene_first_done):
            return
        if feats_C.shape[0] < self.min_inst_for_plot:
            return
        P = l2_normalize_rows(feats_C).detach().cpu()
        L = labels_0b.detach().cpu()
        # 시각화 재료 저장 (CPU로)
        bev_cpu = bev_i.detach().cpu()                   # [C,H,W]
        boxes_cpu = boxes_i.detach().cpu().numpy()       # [M,8]
        sample_idx = 0
        try:
            sample_idx = meta_i.get('sample_idx', 0)
        except Exception:
            pass

        self.scene_payloads.append({
            'scene_id': scene_id, 'P': P, 'labels': L,
            'bev': bev_cpu, 'boxes': boxes_cpu, 'sample_idx': sample_idx
        })
        self.scene_first_done.add(scene_id)

    def process_batch(self, batch_dict):
        # Dynamically get the feature from batch_dict using the key path
        bev = batch_dict
        try:
            for key in self.feature_key_path:
                bev = bev[key]
        except (KeyError, TypeError):
            return  # Silently skip if feature not found

        gt_boxes = batch_dict.get('gt_boxes', None)
        metas = batch_dict.get('metadata', None)
        if (bev is None) or (gt_boxes is None) or (metas is None): 
            return
        
        # Handle SparseConvTensor by converting to dense tensor
        if not isinstance(bev, torch.Tensor):
            if hasattr(bev, 'dense'):  # SparseConvTensor
                bev = bev.dense()
            else:
                return

        bev = bev.detach()
        gt_boxes = gt_boxes.detach()

        B, C, H, W = bev.shape
        for i in range(B):
            meta = metas[i]
            scene_id = self._get_scene_id(meta)
            boxes = gt_boxes[i]  # [M,8]
            valid = boxes[:, -1] > 0
            boxes = boxes[valid]
            if boxes.numel() == 0: continue

            feats, labels0 = [], []
            for k in range(boxes.shape[0]):
                cls1 = int(boxes[k, -1].item())
                cls0 = cls1 - 1
                if not (0 <= cls0 < self.num_classes): continue
                f = self._extract_feature_for_box(bev[i], boxes[k], pooling=self.pooling)
                feats.append(f.view(1, -1)); labels0.append(cls0)

            if len(feats) < 2: continue
            feats = torch.cat(feats, dim=0)  # [N,C]
            labels0 = torch.tensor(labels0, device=feats.device, dtype=torch.long)

            self._accumulate_class_sim(feats, labels0)
            self._maybe_capture_scene_first(scene_id, feats, labels0, bev[i], boxes, meta)

    def _save_similarity_map(self, sim_sums, out_dir, file_prefix, title):
        valid_mask = self.sim_counts > 0
        S = np.zeros_like(sim_sums, dtype=np.float64)
        S[valid_mask] = sim_sums[valid_mask] / (self.sim_counts[valid_mask] + 1e-12)
        np.fill_diagonal(S, 1.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(out_dir / f'{file_prefix}.npy', S)
        
        fig, ax = plt.subplots(figsize=(0.6*self.num_classes+2, 0.6*self.num_classes+2))
        im = ax.imshow(S, cmap='coolwarm', vmin=0.0, vmax=1.0)
        ax.set_xticks(range(self.num_classes)); ax.set_yticks(range(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.set_yticklabels(self.class_names)
        ax.set_title(f'{title} (BEV features, 1x1 pooled)')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout(); fig.savefig(out_dir / f'{file_prefix}.png', dpi=200); plt.close(fig)
        self.logger.info(f"[Similarity] Saved {title} to {out_dir}")

    def _save_instance_similarity_for_scene(self, file_stem, scene_id, P, labels, out_dir):
        S = (P @ P.T).cpu().numpy()
        labels_np = labels.cpu().numpy()
        sort_idx = np.argsort(labels_np)
        S_sorted = S[sort_idx][:, sort_idx]
        labels_sorted = labels_np[sort_idx]
        inst_ticks = [self.class_names[int(cls)] for cls in labels_sorted]

        out_dir.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(min(10, max(6, 0.15*len(inst_ticks))), min(8, max(5, 0.15*len(inst_ticks)))))
        plt.imshow(S_sorted, vmin=-1, vmax=1, cmap='coolwarm')
        plt.title(f'Instance-level Similarity | scene={scene_id} | N={S_sorted.shape[0]}')
        plt.xticks(range(len(inst_ticks)), inst_ticks, rotation=90, fontsize=6)
        plt.yticks(range(len(inst_ticks)), inst_ticks, fontsize=6)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_dir / f'{file_stem}.png', dpi=200); plt.close(fig)

        np.save(out_dir / f'{file_stem}.npy', S_sorted)

    def _save_bev_vis_for_scene(self, file_stem, scene_id, bev_chw_cpu, boxes_np, pc_range, class_names, out_dir, sample_idx=0):
        bev_np = bev_chw_cpu.numpy()  # [C,H,W]
        H, W = bev_np.shape[1], bev_np.shape[2]
        feat_map = bev_np.mean(0)
        denom = (feat_map.max() - feat_map.min())
        norm_map = (feat_map - feat_map.min()) / (denom + 1e-12) if denom >= 1e-12 else np.zeros_like(feat_map)

        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(norm_map, cmap='viridis', origin='lower')

        valid = boxes_np[:, -1] > 0
        boxes_np = boxes_np[valid]
        for k in range(len(boxes_np)):
            x, y, z, dx, dy, dz, heading, cls = boxes_np[k, :8]
            corners_xy = box_corners_world(float(x), float(y), float(dx), float(dy), float(heading))
            pixels = [world_to_bev_pixel(cx, cy, pc_range, H, W) for cx, cy in corners_xy]
            verts = [(float(col), float(row)) for (row, col) in pixels]
            ax.add_patch(patches.Polygon(verts, closed=True, fill=False, edgecolor='red', linewidth=1.5))
            row, col = world_to_bev_pixel(float(x), float(y), pc_range, H, W)
            cls_idx = int(cls) - 1
            if 0 <= cls_idx < len(class_names):
                ax.text(col, row, f'{class_names[cls_idx]}', color='white', fontsize=8,
                        ha='center', va='center', path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        ax.set_title(f'BEV + GT (scene={scene_id}, sample={sample_idx})')
        ax.set_axis_off()
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(out_dir / f'{file_stem}.png', dpi=200, bbox_inches='tight'); plt.close(fig)

    def finalize(self, result_dir, save_class=True):
        base_output_dir = result_dir / 'similarity' / self.feature_name
        class_sim_dir = base_output_dir / 'class'
        inst_sim_dir = base_output_dir / 'instances'
        bev_vis_dir = base_output_dir / 'instances_bev'

        if save_class:
            self._save_similarity_map(self.cos_sim_sums, class_sim_dir, 'class_cosine_similarity', f'Cosine Sim. ({self.feature_name})')
            self._save_similarity_map(self.cka_linear_sums, class_sim_dir, 'class_cka_linear_similarity', f'Linear CKA ({self.feature_name})')
            self._save_similarity_map(self.cka_rbf_sums, class_sim_dir, 'class_cka_rbf_similarity', f'RBF CKA ({self.feature_name})')

        if self.save_scene_instance:
            saved_cnt = 0
            for i, payload in enumerate(self.scene_payloads):
                if i >= self.max_scene_plots: break
                scene_id = payload['scene_id']; P = payload['P']; L = payload['labels']
                if P.shape[0] >= self.min_inst_for_plot:
                    file_idx_str = f"{i:06d}"
                    self._save_instance_similarity_for_scene(file_idx_str, scene_id, P, L, inst_sim_dir)
                    if self.save_scene_bev_vis:
                        self._save_bev_vis_for_scene(file_idx_str, scene_id, payload['bev'], payload['boxes'],
                                                     self.pc_range, self.class_names, bev_vis_dir, payload['sample_idx'])
                    saved_cnt += 1
            self.logger.info(f"[Similarity] Instance-level plots saved: {saved_cnt} scenes; BEV vis saved: {saved_cnt if self.save_scene_bev_vis else 0}")


def eval_teacher_model(model, test_loader, args, eval_output_dir, logger, dist_test=False, sim_engines=None):
    # Use OpenPCDet's standard loading method like test.py
    model.load_params_from_file(filename=args.teacher_ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()


    # (요청) 별도의 초반 N개 시각화는 제거함


    # 원래 평가 루틴 (test.py와 동일하게)
    epoch_id = "teacher"
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, bev_similarity_engines=sim_engines
    )

    # 최종 저장 (class-level, scene-level 인스턴스 맵 + scene 첫 샘플 BEV 시각화)
    # Save similarity results BEFORE NuScenes evaluation to avoid OOM
    if sim_engines and cfg.LOCAL_RANK == 0:
        logger.info("Saving similarity results for all features...")
        for engine in sim_engines:
            engine.finalize(eval_output_dir, save_class=args.save_class_similarity)
        logger.info("All similarity results saved successfully!")
        
        # Clear memory to prevent OOM
        del sim_engines
        torch.cuda.empty_cache()
        import gc
        gc.collect()


def main():
    args, cfg_loaded = parse_config()

    if args.teacher_ckpt is None:
        raise ValueError("--teacher_ckpt must be specified")

    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False; total_gpus = 1
    else:
        total_gpus, cfg_loaded.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg_loaded.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    # Simplified output directory structure: output/[extra_tag]/eval
    output_dir = cfg_loaded.ROOT_DIR / 'output' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = eval_output_dir / ('log_eval_teacher_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg_loaded.LOCAL_RANK)

    logger.info('**********************Start Teacher Model Evaluation**********************')
    gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL'); logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg_loaded, logger=logger)

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg_loaded.DATA_CONFIG,
        class_names=cfg_loaded.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg_loaded.MODEL, num_class=len(cfg_loaded.CLASS_NAMES), dataset=test_set)

    # Define feature maps to analyze for the teacher model
    # Corrected feature maps based on debug logs. The teacher is a standard PillarNet.
    feature_map_definitions = {
        'teacher': {
            'low_lidar_bev': 'spatial_features_2d',
            'low_lidar_de_8x': 'spatial_features_2d_8x',
            'high_lidar_bev': 'multi_scale_2d_features.x_conv4',
            'high_lidar_bev_8x': 'multi_scale_2d_features.x_conv5',
        }
    }

    features_to_run = []
    if args.features_to_analyze:
        if args.features_to_analyze.lower() == 'all':
            features_to_run = list(feature_map_definitions.get('teacher', {}).keys())
        else:
            features_to_run = [f.strip() for f in args.features_to_analyze.split(',')]

    sim_engines = []
    if features_to_run:
        logger.info(f"Preparing similarity engines for features: {features_to_run}")
        all_feature_maps = feature_map_definitions.get('teacher', {})
        for feature_name in features_to_run:
            if feature_name in all_feature_maps:
                engine = BEVSimilarityEngine(
                    feature_name=feature_name,
                    feature_key_path=all_feature_maps[feature_name],
                    class_names=cfg_loaded.CLASS_NAMES,
                    pc_range=cfg_loaded.DATA_CONFIG.POINT_CLOUD_RANGE,
                    logger=logger,
                    pooling=args.similarity_pooling,
                    save_scene_instance=args.save_scene_instance_similarity,
                    max_scene_plots=args.max_scene_instance_plots,
                    min_inst_for_plot=args.min_instances_per_scene_plot,
                    save_scene_bev_vis=args.save_scene_bev_vis
                )
                sim_engines.append(engine)
            else:
                logger.warning(f"Feature '{feature_name}' not defined for teacher model. Skipping.")
    
    eval_teacher_model(model, test_loader, args, eval_output_dir, logger, dist_test=dist_test, sim_engines=sim_engines)


if __name__ == '__main__':
    main()
