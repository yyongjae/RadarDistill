import _init_path
import argparse
import datetime
import glob
import os
import re
import time
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import math
from matplotlib.path import Path as MplPath
from tqdm import tqdm
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not installed. UMAP visualization will be disabled.")

# ==============================
# 공통 유틸 (from test_teacher.py)
# ==============================
def world_to_bev_pixel(x, y, pc_range, bev_h, bev_w):
    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    u = (x - x_min) / (x_max - x_min + 1e-12)
    v = (y - y_min) / (y_max - y_min + 1e-12)
    col = np.clip(u * bev_w, 0, bev_w - 1)
    row = np.clip(v * bev_h, 0, bev_h - 1)
    return row, col

def box_corners_world(x, y, dx, dy, heading):
    c, s = math.cos(heading), math.sin(heading)
    hx, hy = dx/2.0, dy/2.0
    corners_local = np.array([[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy]], dtype=np.float32)
    R = np.array([[c,-s],[s, c]], dtype=np.float32)
    return corners_local @ R.T + np.array([x,y], dtype=np.float32)

def polygon_mask_in_feature(corners_xy, pc_range, H, W):
    pixels = [world_to_bev_pixel(cx, cy, pc_range, H, W) for cx, cy in corners_xy]
    verts = [(float(col), float(row)) for (row, col) in pixels]
    poly = MplPath(verts)
    cols = [v[0] for v in verts]; rows = [v[1] for v in verts]
    cmin, cmax = int(max(0, math.floor(min(cols)))), int(min(W-1, math.ceil(max(cols))))
    rmin, rmax = int(max(0, math.floor(min(rows)))), int(min(H-1, math.ceil(max(rows))))
    if cmax < cmin or rmax < rmin:
        return None, None, None
    grid_x, grid_y = np.meshgrid(np.arange(cmin, cmax+1), np.arange(rmin, rmax+1))
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
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


class BEVSimilarityEngine:
    def __init__(self, feature_name, feature_key_path, class_names, pc_range, logger, result_dir, pooling='avg', save_scene_instance=False, max_scene_plots=4, min_inst_for_plot=2):
        self.feature_name = feature_name
        self.feature_key_path = feature_key_path.split('.')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.pc_range = pc_range
        self.logger = logger
        self.result_dir = result_dir
        self.pooling = pooling
        self.save_scene_instance = save_scene_instance
        self.max_scene_plots = max_scene_plots
        self.min_inst_for_plot = min_inst_for_plot
        # Cosine, CKA Linear, CKA RBF
        self.cos_sim_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.cka_linear_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.cka_rbf_sums = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.sim_counts = np.zeros((self.num_classes, self.num_classes), dtype=np.float64)
        self.scene_payloads = [] # Now only used for class similarity accumulation
        self.saved_sample_count = 0

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

        # Pairwise CKA (can be slow, but avoids large Gram matrices)
        # For CKA, we compare the representations (features) directly, not normalized ones.
        # And we compare them as matrices [1, C] vs [1, C]
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

    def _save_instance_data(self, scene_id, feats_C, labels_0b, bev_i, boxes_i, meta_i):
        # 저장 옵션이 꺼져있거나, 최대 저장 개수를 넘었으면 반환
        if not self.save_scene_instance or self.saved_sample_count >= self.max_scene_plots:
            return

        # 최소 인스턴스 개수 조건 제거 - 모든 샘플 저장
        # if feats_C.shape[0] < self.min_inst_for_plot:
        #     return

        # --- 저장 로직 ---
        P = l2_normalize_rows(feats_C).detach().cpu()
        L = labels_0b.detach().cpu()

        # 저장할 기본 폴더 설정 (씬별 하위 폴더 없음)
        base_output_dir = self.result_dir / 'similarity' / self.feature_name
        inst_sim_dir = base_output_dir / 'instances'

        # 순차적인 번호를 파일 이름으로 사용 (1부터 시작)
        file_idx_str = f"{self.saved_sample_count + 1:06d}"
        self._save_instance_similarity_for_scene(file_idx_str, scene_id, P, L, inst_sim_dir)

        self.saved_sample_count += 1

    def _save_instance_data_with_zeros(self, scene_id, dummy_feat, dummy_label, meta, reason):
        """Save zero-filled instance data for cases where feature extraction failed or no valid boxes"""
        # 저장 옵션이 꺼져있거나, 최대 저장 개수를 넘었으면 반환
        if not self.save_scene_instance or self.saved_sample_count >= self.max_scene_plots:
            return

        # Create single dummy feature and label
        feats_C = dummy_feat.unsqueeze(0)  # [1, C]
        labels_0b = dummy_label.unsqueeze(0)  # [1]
        
        # --- 저장 로직 ---
        P = l2_normalize_rows(feats_C).detach().cpu()
        L = labels_0b.detach().cpu()

        # 저장할 기본 폴더 설정
        base_output_dir = self.result_dir / 'similarity' / self.feature_name
        inst_sim_dir = base_output_dir / 'instances'

        # 순차적인 번호를 파일 이름으로 사용 (1부터 시작)
        file_idx_str = f"{self.saved_sample_count + 1:06d}_zero_{reason}"
        self._save_instance_similarity_for_scene(file_idx_str, scene_id, P, L, inst_sim_dir)

        self.saved_sample_count += 1

    def process_batch(self, batch_dict):
        # Dynamically get the feature from batch_dict using the key path
        bev = batch_dict
        feature_extraction_failed = False
        try:
            for key in self.feature_key_path:
                bev = bev[key]
        except (KeyError, TypeError):
            feature_extraction_failed = True
            bev = None

        gt_boxes = batch_dict.get('gt_boxes', None)
        metas = batch_dict.get('metadata', None)
        if (gt_boxes is None) or (metas is None): 
            return
        
        # Handle SparseConvTensor by converting to dense tensor
        if bev is not None and not isinstance(bev, torch.Tensor):
            if hasattr(bev, 'dense'):  # SparseConvTensor
                bev = bev.dense()
            else:
                feature_extraction_failed = True
                bev = None
        
        # Debug: Log successful feature extraction
        if not feature_extraction_failed and not hasattr(self, '_success_logged'):
            self.logger.info(f"[{self.feature_name}] Successfully extracted feature with shape {bev.shape}")
            self._success_logged = True
        
        if not feature_extraction_failed:
            bev = bev.detach()
        gt_boxes = gt_boxes.detach()
        
        # Get feature dimensions from successful extraction or use default
        if not feature_extraction_failed:
            B, C, H, W = bev.shape
        else:
            B = gt_boxes.shape[0]
            C, H, W = 256, 432, 432  # Default dimensions for missing features
        
        for i in range(B):
            meta = metas[i]
            scene_id = self._get_scene_id(meta)
            boxes = gt_boxes[i]
            valid = boxes[:, -1] > 0
            boxes = boxes[valid]
            
            # Handle cases with no valid boxes or feature extraction failure
            if boxes.numel() == 0 or feature_extraction_failed:
                # Create zero-filled dummy data for consistent processing
                if feature_extraction_failed:
                    # Create dummy features with zeros
                    dummy_feat = torch.zeros(C, device=gt_boxes.device)
                    dummy_label = torch.tensor([0], dtype=torch.long, device=gt_boxes.device)  # Use first class
                    self._save_instance_data_with_zeros(scene_id, dummy_feat, dummy_label, meta, "feature_extraction_failed")
                else:
                    # Create dummy features for no boxes case
                    dummy_feat = torch.zeros(C, device=gt_boxes.device)
                    dummy_label = torch.tensor([0], dtype=torch.long, device=gt_boxes.device)  # Use first class
                    self._save_instance_data_with_zeros(scene_id, dummy_feat, dummy_label, meta, "no_valid_boxes")
                continue
                
            feats_C = []
            labels_0b = []
            for j, box in enumerate(boxes):
                if feature_extraction_failed:
                    # Use zero features when extraction failed
                    feat = torch.zeros(C, device=gt_boxes.device)
                else:
                    feat = self._extract_feature_for_box(bev[i], box, self.pooling)
                    
                if feat is not None:
                    feats_C.append(feat)
                    labels_0b.append(int(box[-1]) - 1)
            
            # Always process even if no features extracted successfully
            if len(feats_C) == 0:
                # Create dummy features with zeros
                dummy_feat = torch.zeros(C, device=gt_boxes.device)
                dummy_label = torch.tensor([0], dtype=torch.long, device=gt_boxes.device)
                self._save_instance_data_with_zeros(scene_id, dummy_feat, dummy_label, meta, "feature_extraction_all_failed")
                continue
                
            feats_C = torch.stack(feats_C)
            labels_0b = torch.tensor(labels_0b, dtype=torch.long, device=feats_C.device)
            
            self._accumulate_class_sim(feats_C, labels_0b)

            # Save instance-level data for the current sample
            self._save_instance_data(scene_id, feats_C, labels_0b, bev[i] if not feature_extraction_failed else None, boxes, meta)

    def _save_similarity_map(self, sim_sums, out_dir, file_prefix, title):
        valid_mask = self.sim_counts > 0
        S = np.zeros_like(sim_sums, dtype=np.float64)
        S[valid_mask] = sim_sums[valid_mask] / (self.sim_counts[valid_mask] + 1e-12)
        np.fill_diagonal(S, 1.0)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save similarity data with metadata (no PNG generation)
        save_data = {
            'similarity_matrix': S,
            'class_names': self.class_names,
            'title': title,
            'similarity_counts': self.sim_counts,
            'feature_name': self.feature_name,
            'pooling_method': self.pooling
        }
        np.save(out_dir / f'{file_prefix}.npy', save_data)
        self.logger.info(f"[Similarity] Saved {title} to {out_dir}")

    def _save_instance_similarity_for_scene(self, file_stem, scene_id, P, labels, out_dir):
        S = (P @ P.T).cpu().numpy()
        labels_np = labels.cpu().numpy()
        sort_idx = np.argsort(labels_np)
        S_sorted = S[sort_idx][:, sort_idx]
        labels_sorted = labels_np[sort_idx]
        inst_ticks = [self.class_names[int(cls)] for cls in labels_sorted]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure S_sorted is 2D
        if S_sorted.ndim > 2:
            S_sorted = S_sorted.squeeze()
        if S_sorted.ndim == 0:  # Single value case
            S_sorted = S_sorted.reshape(1, 1)
        elif S_sorted.ndim == 1:  # 1D case
            S_sorted = S_sorted.reshape(1, -1)
        
        # Save similarity matrix with metadata (no PNG generation)
        save_data = {
            'similarity_matrix': S_sorted,
            'labels': labels_sorted,
            'class_names': self.class_names,
            'sample_id': file_stem,
            'scene_id': scene_id,
            'num_instances': S_sorted.shape[0]
        }
        np.save(out_dir / f'{file_stem}.npy', save_data)
        
        # Save sample-level class similarity map
        self._save_sample_class_similarity_map(file_stem, scene_id, labels_np, out_dir)

    def _save_sample_class_similarity_map(self, file_stem, scene_id, labels_np, out_dir):
        """Create class similarity map for this specific sample"""
        unique_classes = np.unique(labels_np)
        # Remove minimum class condition - save all samples
        # if len(unique_classes) < 2:
        #     return  # Skip if only one class present
            
        # Create class-level similarity matrix for this sample
        class_sim_matrix = np.zeros((self.num_classes, self.num_classes))
        class_counts = np.zeros((self.num_classes, self.num_classes))
        
        # Count co-occurrences in this sample
        for i in range(len(labels_np)):
            for j in range(len(labels_np)):
                if i != j:
                    ci, cj = int(labels_np[i]), int(labels_np[j])
                    if 0 <= ci < self.num_classes and 0 <= cj < self.num_classes:
                        class_sim_matrix[ci, cj] = 1.0  # Co-occurrence indicator
                        class_counts[ci, cj] = 1.0
        
        # Set diagonal to 1 for present classes
        for cls in unique_classes:
            if 0 <= cls < self.num_classes:
                class_sim_matrix[cls, cls] = 1.0
        
        # Save class similarity data with metadata (no PNG generation)
        class_sim_dir = out_dir.parent / 'class_similarity_per_sample'
        class_sim_dir.mkdir(parents=True, exist_ok=True)
        
        class_save_data = {
            'class_similarity_matrix': class_sim_matrix,
            'unique_classes': unique_classes,
            'class_names': self.class_names,
            'sample_id': file_stem,
            'scene_id': scene_id,
            'num_unique_classes': len(unique_classes)
        }
        np.save(class_sim_dir / f'{file_stem}_class_sim.npy', class_save_data)

    def finalize(self, result_dir, save_class=True, dist_test=False):
        if dist_test:
            # 모든 GPU에서 수집된 scene_payloads를 rank 0으로 병합
            all_payloads = common_utils.merge_results_dist(self.scene_payloads, len(self.scene_payloads), tmpdir=result_dir / 'tmpdir')
            if cfg.LOCAL_RANK == 0:
                # 중복된 scene_id 제거 (각 GPU가 동일 scene의 다른 샘플을 볼 수 있으므로)
                seen_scenes = set()
                unique_payloads = []
                for p in all_payloads:
                    if p['scene_id'] not in seen_scenes:
                        unique_payloads.append(p)
                        seen_scenes.add(p['scene_id'])
                self.scene_payloads = unique_payloads
            else:
                self.scene_payloads = []

        base_output_dir = result_dir / 'similarity' / self.feature_name
        class_sim_dir = base_output_dir / 'class'
        inst_sim_dir = base_output_dir / 'instances'
        bev_vis_dir = base_output_dir / 'instances_bev'

        # Save class-level similarities
        if save_class:
            self._save_similarity_map(self.cos_sim_sums, class_sim_dir, 'class_cosine_similarity', f'Cosine Sim. ({self.feature_name})')
            self._save_similarity_map(self.cka_linear_sums, class_sim_dir, 'class_cka_linear_similarity', f'Linear CKA ({self.feature_name})')
            self._save_similarity_map(self.cka_rbf_sums, class_sim_dir, 'class_cka_rbf_similarity', f'RBF CKA ({self.feature_name})')

        # Log how many instance-level plots were saved in real-time
        if self.save_scene_instance and cfg.LOCAL_RANK == 0:
            self.logger.info(f"[Similarity] Instance-level plots for all samples saved in real-time: {self.saved_sample_count} total samples.")


class BEVUMAPVisualizer:
    """Visualize BEV feature distribution using UMAP dimensionality reduction"""
    def __init__(self, feature_name, feature_key_path, class_names, pc_range, logger, result_dir, 
                 pooling='avg', n_neighbors=15, min_dist=0.1, n_components=2, max_samples=10000,
                 tsne_perplexity=30.0, tsne_n_iter=1000):
        self.feature_name = feature_name
        self.feature_key_path = feature_key_path.split('.')
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.pc_range = pc_range
        self.logger = logger
        self.result_dir = result_dir
        self.pooling = pooling
        
        # UMAP parameters
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.n_components = n_components
        self.max_samples = max_samples
        
        # t-SNE parameters
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        
        # Storage for features and labels
        self.features_list = []
        self.labels_list = []
        self.sample_count = 0
        
        if not UMAP_AVAILABLE:
            self.logger.warning(f"[{self.feature_name}] UMAP not available. Install with: pip install umap-learn")
    
    def _extract_feature_for_box(self, bev_feat_chw, box8, pooling):
        """Extract feature for a single box (same as BEVSimilarityEngine)"""
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
            return center_pixel_feature
    
    def process_batch(self, batch_dict):
        """Collect features from a batch"""
        if not UMAP_AVAILABLE or self.sample_count >= self.max_samples:
            return
        
        # Extract BEV feature
        bev = batch_dict
        feature_extraction_failed = False
        try:
            for key in self.feature_key_path:
                bev = bev[key]
        except (KeyError, TypeError) as e:
            feature_extraction_failed = True
            bev = None
            # Debug: Log available keys on first failure
            if not hasattr(self, '_keys_logged'):
                self.logger.warning(f"[{self.feature_name}] UMAP:wd Failed to extract feature with key path {'.'.join(self.feature_key_path)}")
                self.logger.warning(f"[{self.feature_name}] UMAP: Available keys in batch_dict: {list(batch_dict.keys())}")
                self._keys_logged = True
        
        gt_boxes = batch_dict.get('gt_boxes', None)
        if (gt_boxes is None) or feature_extraction_failed:
            return
        
        # Handle SparseConvTensor
        if bev is not None and not isinstance(bev, torch.Tensor):
            if hasattr(bev, 'dense'):
                bev = bev.dense()
            else:
                return
        
        if not hasattr(self, '_success_logged'):
            self.logger.info(f"[{self.feature_name}] UMAP: Successfully extracted feature with shape {bev.shape}")
            self._success_logged = True
        
        bev = bev.detach()
        gt_boxes = gt_boxes.detach()
        B, C, H, W = bev.shape
        
        for i in range(B):
            if self.sample_count >= self.max_samples:
                break
                
            boxes = gt_boxes[i]
            valid = boxes[:, -1] > 0
            boxes = boxes[valid]
            
            if boxes.numel() == 0:
                continue
            
            for j, box in enumerate(boxes):
                if self.sample_count >= self.max_samples:
                    break
                    
                feat = self._extract_feature_for_box(bev[i], box, self.pooling)
                if feat is not None:
                    self.features_list.append(feat.cpu().numpy())
                    self.labels_list.append(int(box[-1]) - 1)  # 0-indexed class
                    self.sample_count += 1
    
    def finalize(self, result_dir, dist_test=False):
        """Apply t-SNE and generate visualizations"""
        if not UMAP_AVAILABLE:
            return
        
        if dist_test:
            # Merge features from all GPUs
            all_features = common_utils.merge_results_dist(self.features_list, len(self.features_list), 
                                                          tmpdir=result_dir / 'tmpdir')
            all_labels = common_utils.merge_results_dist(self.labels_list, len(self.labels_list),
                                                        tmpdir=result_dir / 'tmpdir')
            if cfg.LOCAL_RANK == 0:
                self.features_list = all_features[:self.max_samples]
                self.labels_list = all_labels[:self.max_samples]
            else:
                return
        
        if len(self.features_list) < 2:
            self.logger.warning(f"[{self.feature_name}] UMAP: Not enough samples ({len(self.features_list)})")
            return
        
        self.logger.info(f"[{self.feature_name}] Dimensionality Reduction: Processing {len(self.features_list)} samples...")
        
        # Convert to numpy arrays
        features = np.array(self.features_list)  # [N, C]
        labels = np.array(self.labels_list)  # [N]
        
        # Create output directory
        umap_dir = result_dir / 'umap' / self.feature_name
        umap_dir.mkdir(parents=True, exist_ok=True)
        
        # Import t-SNE
        try:
            from sklearn.manifold import TSNE
            tsne_available = True
        except ImportError:
            tsne_available = False
            self.logger.warning(f"[{self.feature_name}] t-SNE not available (sklearn not installed)")
        
        # Generate visualizations for different sample counts
        sample_counts = [5000, 10000]
        for n_samples in sample_counts:
            if len(features) < n_samples:
                self.logger.info(f"[{self.feature_name}] Skipping {n_samples} samples (only {len(features)} available)")
                continue
            
            self.logger.info(f"[{self.feature_name}] Generating visualizations for {n_samples} samples...")
            
            # Subsample features and labels
            features_subset = features[:n_samples]
            labels_subset = labels[:n_samples]
            
            # ========== t-SNE ==========
            if tsne_available:
                self.logger.info(f"[{self.feature_name}] Applying t-SNE (perplexity={self.tsne_perplexity}, n_iter={self.tsne_n_iter}, n_components={self.n_components})...")
                
                # t-SNE parameters
                # Adjust perplexity for small datasets if needed, but respect user setting if possible
                perplexity = min(self.tsne_perplexity, n_samples // 4)
                if perplexity < self.tsne_perplexity:
                    self.logger.warning(f"[{self.feature_name}] t-SNE perplexity adjusted from {self.tsne_perplexity} to {perplexity} due to small sample size")
                
                reducer_tsne = TSNE(
                    n_components=self.n_components,
                    perplexity=perplexity,
                    random_state=42,
                    n_iter=self.tsne_n_iter,
                    verbose=0
                )
                
                embedding_tsne = reducer_tsne.fit_transform(features_subset)
                
                # Generate t-SNE visualizations
                self._plot_umap(embedding_tsne, labels_subset, umap_dir, suffix=f'_{n_samples}_tsne', method='t-SNE')
                
                self.logger.info(f"[{self.feature_name}] t-SNE visualization for {n_samples} samples saved")
            
            self.logger.info(f"[{self.feature_name}] All visualizations for {n_samples} samples completed")
    
    def _plot_umap(self, embedding, labels, output_dir, suffix='', method='UMAP'):
        """Generate dimensionality reduction scatter plots"""
        # Define colors for each class
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_classes))
        
        if self.n_components == 2:
            # 2D scatter plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            for cls_idx in range(self.num_classes):
                mask = labels == cls_idx
                if mask.sum() > 0:
                    ax.scatter(
                        embedding[mask, 0], 
                        embedding[mask, 1],
                        c=[colors[cls_idx]], 
                        label=self.class_names[cls_idx],
                        alpha=0.6,
                        s=20,
                        edgecolors='none'
                    )
            
            ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
            ax.set_title(f'{method} Visualization: {self.feature_name}\n'
                        f'(n_samples={len(labels)})',
                        fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'umap_2d{suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Also create interactive HTML plot using plotly if available
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                for cls_idx in range(self.num_classes):
                    mask = labels == cls_idx
                    if mask.sum() > 0:
                        fig.add_trace(go.Scatter(
                            x=embedding[mask, 0],
                            y=embedding[mask, 1],
                            mode='markers',
                            name=self.class_names[cls_idx],
                            marker=dict(size=5, opacity=0.6),
                            text=[self.class_names[cls_idx]] * mask.sum(),
                            hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f'{method} Visualization: {self.feature_name}',
                    xaxis_title=f'{method} Dimension 1',
                    yaxis_title=f'{method} Dimension 2',
                    hovermode='closest',
                    width=1200,
                    height=900
                )
                
                fig.write_html(output_dir / f'umap_2d_interactive{suffix}.html')
            except ImportError:
                pass  # plotly not available
        
        elif self.n_components == 3:
            # 3D scatter plot
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            for cls_idx in range(self.num_classes):
                mask = labels == cls_idx
                if mask.sum() > 0:
                    ax.scatter(
                        embedding[mask, 0],
                        embedding[mask, 1],
                        embedding[mask, 2],
                        c=[colors[cls_idx]],
                        label=self.class_names[cls_idx],
                        alpha=0.6,
                        s=20,
                        edgecolors='none'
                    )
            
            ax.set_xlabel(f'{method} Dimension 1', fontsize=12)
            ax.set_ylabel(f'{method} Dimension 2', fontsize=12)
            ax.set_zlabel(f'{method} Dimension 3', fontsize=12)
            ax.set_title(f'{method} Visualization: {self.feature_name}\n'
                        f'(n_samples={len(labels)})',
                        fontsize=14)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'umap_3d{suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Interactive 3D plot with plotly
            try:
                import plotly.graph_objects as go
                
                fig = go.Figure()
                for cls_idx in range(self.num_classes):
                    mask = labels == cls_idx
                    if mask.sum() > 0:
                        fig.add_trace(go.Scatter3d(
                            x=embedding[mask, 0],
                            y=embedding[mask, 1],
                            z=embedding[mask, 2],
                            mode='markers',
                            name=self.class_names[cls_idx],
                            marker=dict(size=3, opacity=0.6),
                            text=[self.class_names[cls_idx]] * mask.sum(),
                            hovertemplate='%{text}<br>x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>'
                        ))
                
                fig.update_layout(
                    title=f'{method} Visualization: {self.feature_name}',
                    scene=dict(
                        xaxis_title=f'{method} Dimension 1',
                        yaxis_title=f'{method} Dimension 2',
                        zaxis_title=f'{method} Dimension 3'
                    ),
                    hovermode='closest',
                    width=1200,
                    height=900
                )
                
                fig.write_html(output_dir / f'umap_3d_interactive{suffix}.html')
            except ImportError:
                pass  # plotly not available




def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    parser.add_argument('--infer_time', action='store_true', default=False, help='calculate inference latency')
    parser.add_argument('--cal_params', action='store_true', default=False, help='')

    # Arguments for similarity map generation
    parser.add_argument('--model_type', type=str, default='student', choices=['student', 'baseline', 'teacher'],
                        help='Type of the model to determine which features to analyze.')
    parser.add_argument('--features_to_analyze', type=str, default=None,
                        help='Comma-separated list of features to analyze, or "all".')

    parser.add_argument('--save_class_similarity', action='store_true', default=False,
                        help='Save accumulated class-class similarity map for the entire test set')
    parser.add_argument('--save_scene_instance_similarity', action='store_true', default=False,
                        help='Save instance-level similarity map for the first sample of each scene')
    parser.add_argument('--similarity_pooling', type=str, default='max',
                        choices=['avg', 'max'], help='Pooling method for 1x1 bbox features (avg or max)')
    parser.add_argument('--max_scene_instance_plots', type=int, default=99999,
                        help='Maximum number of instance map plots to save')
    parser.add_argument('--min_instances_per_scene_plot', type=int, default=2,
                        help='Minimum number of instances required to save an instance map')
    
    # Arguments for UMAP visualization
    parser.add_argument('--save_umap_visualization', action='store_true', default=False,
                        help='Generate UMAP visualization of BEV features')
    parser.add_argument('--umap_n_neighbors', type=int, default=15,
                        help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap_min_dist', type=float, default=0.1,
                        help='UMAP min_dist parameter (default: 0.1)')
    parser.add_argument('--umap_n_components', type=int, default=2, choices=[2, 3],
                        help='UMAP n_components: 2 for 2D or 3 for 3D visualization (default: 2)')
    parser.add_argument('--umap_max_samples', type=int, default=10000,
                        help='Maximum number of samples to collect for UMAP (default: 10000)')
    
    # Arguments for t-SNE visualization
    parser.add_argument('--tsne_perplexity', type=float, default=70.0,
                        help='t-SNE perplexity parameter')
    parser.add_argument('--tsne_n_iter', type=int, default=2000,
                        help='t-SNE number of iterations')


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


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False, sim_engines=None, umap_visualizers=None):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test, 
                                pre_trained_path=args.pretrained_model)
    model.cuda()
    

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, args, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, bev_similarity_engines=sim_engines, umap_visualizers=umap_visualizers
    )

    if sim_engines:
        logger.info("Saving similarity results for all features...")
        for engine in sim_engines:
            engine.finalize(eval_output_dir, save_class=args.save_class_similarity, dist_test=dist_test)
        if cfg.LOCAL_RANK == 0:
            logger.info("All similarity results saved successfully!")
    
    if umap_visualizers:
        logger.info("Generating UMAP/t-SNE visualizations for all features...")
        for visualizer in umap_visualizers:
            visualizer.finalize(eval_output_dir, dist_test=dist_test)
        if cfg.LOCAL_RANK == 0:
            logger.info("All visualizations saved successfully!")

def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False, sim_engines=None, umap_visualizers=None):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            if cfg.LOCAL_RANK == 0:
                print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                      % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()


        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, args, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, bev_similarity_engines=sim_engines, umap_visualizers=umap_visualizers
        )

        # Finalize and save similarity maps
        if sim_engines:
            logger.info("Saving similarity results for all features...")
            for engine in sim_engines:
                engine.finalize(eval_output_dir, save_class=args.save_class_similarity, dist_test=dist_test)
            if cfg.LOCAL_RANK == 0:
                logger.info("All similarity results saved successfully!")
        
        # Finalize and save UMAP visualizations
        if umap_visualizers:
            logger.info("Generating UMAP visualizations for all features...")
            for visualizer in umap_visualizers:
                visualizer.finalize(eval_output_dir, dist_test=dist_test)
            if cfg.LOCAL_RANK == 0:
                logger.info("All UMAP visualizations saved successfully!")

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def main():
    args, cfg = parse_config()
    print(cfg)
    if args.infer_time:
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = 32
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    # Simplified output directory structure: output/[extra_tag]/eval
    output_dir = cfg.ROOT_DIR / 'output' / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract epoch_id for eval_single_ckpt function
    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
    else:
        epoch_id = 'eval_all'
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # Define feature maps to analyze based on model type
    # Corrected feature maps based on debug logs. The running model seems to be a standard PillarNet.
    feature_map_definitions = {
        'student': {
            # Using standard PillarNet features as radar-specific ones are not found
            'low_radar_bev': 'radar_multi_scale_2d_features.radar_spatial_features_8x_2',
            'low_radar_de_8x': 'radar_multi_scale_2d_features.radar_spatial_features_8x_1',
            'high_radar_bev': 'radar_spatial_features_2d', 
            'high_radar_bev_8x': 'radar_spatial_features_2d_8x',
        },
        'baseline': {
            # Using standard PillarNet features
            'low_radar_bev': 'radar_multi_scale_2d_features.radar_spatial_features_8x_2',
            'low_radar_de_8x': 'radar_multi_scale_2d_features.radar_spatial_features_8x_1',
            'high_radar_bev': 'radar_spatial_features_2d', 
            'high_radar_bev_8x': 'radar_spatial_features_2d_8x',
        },
        'teacher': {
            'low_lidar_bev': 'spatial_features_2d',
            'high_lidar_bev': 'multi_scale_2d_features.x_conv4',
        }
    }

    features_to_run = []
    if args.features_to_analyze:
        if args.features_to_analyze.lower() == 'all':
            features_to_run = list(feature_map_definitions.get(args.model_type, {}).keys())
        else:
            features_to_run = [f.strip() for f in args.features_to_analyze.split(',')]

    sim_engines = []
    if features_to_run:
        logger.info(f"Preparing similarity engines for features: {features_to_run}")
        all_feature_maps = feature_map_definitions.get(args.model_type, {})
        for feature_name in features_to_run:
            if feature_name in all_feature_maps:
                engine = BEVSimilarityEngine(
                    feature_name=feature_name,
                    feature_key_path=all_feature_maps[feature_name],
                    class_names=cfg.CLASS_NAMES,
                    pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                    logger=logger,
                    result_dir=eval_output_dir,  # Pass the output directory for real-time saving
                    pooling=args.similarity_pooling,
                    save_scene_instance=args.save_scene_instance_similarity,
                    max_scene_plots=args.max_scene_instance_plots,
                    min_inst_for_plot=args.min_instances_per_scene_plot
                )
                sim_engines.append(engine)
            else:
                logger.warning(f"Feature '{feature_name}' not defined for model type '{args.model_type}'. Skipping.")
    
    # Create UMAP visualizers if requested
    umap_visualizers = []
    if args.save_umap_visualization and features_to_run:
        logger.info(f"Preparing UMAP visualizers for features: {features_to_run}")
        all_feature_maps = feature_map_definitions.get(args.model_type, {})
        for feature_name in features_to_run:
            if feature_name in all_feature_maps:
                visualizer = BEVUMAPVisualizer(
                    feature_name=feature_name,
                    feature_key_path=all_feature_maps[feature_name],
                    class_names=cfg.CLASS_NAMES,
                    pc_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                    logger=logger,
                    result_dir=eval_output_dir,
                    pooling=args.similarity_pooling,
                    n_neighbors=args.umap_n_neighbors,
                    min_dist=args.umap_min_dist,
                    n_components=args.umap_n_components,
                    max_samples=args.umap_max_samples,
                    tsne_perplexity=args.tsne_perplexity,
                    tsne_n_iter=args.tsne_n_iter
                )
                umap_visualizers.append(visualizer)
            else:
                logger.warning(f"Feature '{feature_name}' not defined for model type '{args.model_type}'. Skipping.")

    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test, 
                           sim_engines=sim_engines, umap_visualizers=umap_visualizers)
        else:
            eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test, 
                           sim_engines=sim_engines, umap_visualizers=umap_visualizers)



if __name__ == '__main__':
    main()