# ==============================
# radar_distill_multi_sweep_teacher.py
# (Gater Ver4 적용 + 마스크/시간임베딩/EMA 일관화 반영)
# ==============================
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.box_utils import center_to_corner_box2d
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from functools import partial
import cv2
from .base_bev_backbone import BaseBEVBackboneV2

# ▼▼ Gater V4로 교체: 경로는 실제 파일명/패키지에 맞춰 수정하세요.
# 예) 같은 디렉토리에 gating_ver4.py가 있으면:
from .gating_ver4 import SweepGaterV4


def clip_sigmoid(x, eps=1e-4):
    return torch.clamp(x.sigmoid(), min=eps, max=1 - eps)


class Radar_Distill_Multi_Sweep_Teacher(BaseBEVBackboneV2):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.model_cfg = model_cfg
        
        # (원본) encoder/decoder/agg
        self.encoder_1 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_1 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.encoder_2 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_2 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        
        self.encoder_3 = nn.Sequential(
            ConvNeXtBlock(dim=256,downsample=True),
            ConvNeXtBlock(dim=256,downsample=False),
        )
        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(256,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        self.agg_3 = nn.Sequential(
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

        self.voxel_size = self.model_cfg.VOXEL_SIZE
        self.point_cloud_range = self.model_cfg.POINT_CLOUD_RANGE

        # ---------- Gater V4 설정 (YAML에서 전체 제어) ----------
        self.num_teachers = self.model_cfg.get('NUM_TEACHERS', 3)
        
        # Time normalization settings
        self.time_normalize = self.model_cfg.get('GATE_TIME_NORMALIZE', True)
        self.time_max = self.model_cfg.get('GATE_TIME_MAX', 1.0)
        
        # Gating loss weights
        gate_lb_coeff = self.model_cfg.get('GATE_LB_COEFF', 0.01)
        gate_ent_coeff = self.model_cfg.get('GATE_ENT_COEFF', 0.0)
        gate_tv_coeff = self.model_cfg.get('GATE_TV_COEFF', 0.001)
        
        # Heuristic weights (advantage/band/UCB)
        gate_alpha_adv = self.model_cfg.get('GATE_ALPHA_ADV', 1.0)
        gate_beta_band = self.model_cfg.get('GATE_BETA_BAND', 0.0)
        gate_gamma_ucb = self.model_cfg.get('GATE_GAMMA_UCB', 0.0)
        gate_band_L = self.model_cfg.get('GATE_BAND_L', 0.05)
        gate_band_H = self.model_cfg.get('GATE_BAND_H', 0.20)
        
        # Router settings
        gate_w_heur = self.model_cfg.get('GATE_W_HEUR', 0.5)
        gate_w_lear = self.model_cfg.get('GATE_W_LEAR', 0.5)
        gate_use_adapter = self.model_cfg.get('GATE_USE_ADAPTER', True)
        gate_ema_momentum = self.model_cfg.get('GATE_EMA_MOMENTUM', 0.99)
        gate_router_hidden = self.model_cfg.get('GATE_ROUTER_HIDDEN', 64)
        gate_router_spatial = self.model_cfg.get('GATE_ROUTER_SPATIAL', False)
        
        # V4 features (can be disabled for V3 compatibility)
        gate_use_pos = self.model_cfg.get('GATE_USE_POS', False)
        gate_use_conf = self.model_cfg.get('GATE_USE_CONF', False)
        gate_use_entropy = self.model_cfg.get('GATE_USE_ENTROPY', False)
        
        # Runtime debugging (from nested GATER config)
        gater_cfg = self.model_cfg.get('GATER', {})
        use_runtime_assert = gater_cfg.get('use_runtime_assert', False)
        
        # Resolution downsampling
        gating_downsample = self.model_cfg.get('GATING_DOWNSAMPLE', 1)
        
        # Build gater kwargs with all YAML-controlled parameters
        gater_kwargs = dict(
            C=256, num_sweeps=self.num_teachers,
            learn_router=True, router_hidden=gate_router_hidden,
            router_use_spatial=gate_router_spatial,
            mode='soft', k=self.model_cfg.get('GATE_K', 2),
            temp=0.7,
            gating_downsample=gating_downsample,
            # V4 features (controlled by YAML)
            use_pos=gate_use_pos, pos_mode='delta', pos_channels=1,
            use_conf=gate_use_conf, use_entropy=gate_use_entropy,
            # Capacity factor
            capacity_factor=self.model_cfg.get('CAPACITY_FACTOR', 1.25),
            # Heuristic weights
            alpha_adv=gate_alpha_adv,
            beta_band=gate_beta_band,
            gamma_ucb=gate_gamma_ucb,
            band_L=gate_band_L,
            band_H=gate_band_H,
            # Router balance
            w_heur=gate_w_heur,
            w_lear=gate_w_lear,
            # EMA and adapter
            use_adapter=gate_use_adapter,
            ema_momentum=gate_ema_momentum,
            # Loss coefficients
            lb_coeff=gate_lb_coeff,
            ent_coeff=gate_ent_coeff,
            tv_coeff=gate_tv_coeff,
            # EMA persistence
            persistent_ema=True,
            # Runtime debugging
            use_runtime_assert=use_runtime_assert,
        )
        self.gater_low  = SweepGaterV4(**gater_kwargs)
        self.gater_high = SweepGaterV4(**gater_kwargs)
        
        # Warmup tracking
        self._ema_warmed_up = False
        
        # Paper statistics tracking (lightweight)
        self.gating_stats_history = []  # Evolution over epochs
        self.scene_gating_stats = {}    # Per-scene analysis

    # ---------------- low/high 게이트용 마스크 ----------------
    def get_low_gate_mask(self, low_radar_bev, T_low, tau=1e-3):
        """
        low_radar_bev: [B,C,H8,W8]
        T_low:        [B,N,C,H8,W8]
        return:       [B,N,1,H8,W8]
        """
        radar_occ8 = (low_radar_bev.sum(1, keepdim=True) > 0).float()            # [B,1,H8,W8]
        teach_occ8 = (T_low.abs().max(dim=2, keepdim=True)[0] > tau).float()     # [B,N,1,H8,W8]
        mask_low   = teach_occ8 * radar_occ8.unsqueeze(1)                        # [B,N,1,H8,W8]
        return mask_low

    def get_high_gate_mask(self, high_lidar_bev, radar_preds, thr_teacher=1e-3, thr_radar=0.1):
        """
        High-level gating mask: teacher activation + radar prediction activation
        
        Args:
            high_lidar_bev: [B,C,H,W] aggregated or max teacher features
            radar_preds: list of prediction dicts with 'hm' key
            thr_teacher: threshold for teacher activation
            thr_radar: threshold for radar prediction
        Returns:
            [B,1,H,W] binary mask indicating valid regions
        """
        radar_batch_hm = torch.cat([clip_sigmoid(r['hm']) for r in radar_preds], dim=1)  # [B,Ch,H,W]
        radar_max = radar_batch_hm.amax(dim=1, keepdim=True)                              # [B,1,H,W]
        teach_max = high_lidar_bev.abs().amax(dim=1, keepdim=True)                        # [B,1,H,W]
        gt_mask   = (teach_max > thr_teacher)
        radar_pos = (radar_max > thr_radar)
        mask_high = (gt_mask | radar_pos).float()  # Union of teacher & radar active regions
        return mask_high

    # ---------------- distill losses (원본 유지) ----------------
    def low_loss(self, lidar_bev, radar_bev, mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask):
        B, _, H, W = radar_bev.shape
        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / max(B, 1)
        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / max(B, 1)

        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        loss = nn.L1Loss()
        radar_sigmoid = torch.clamp(radar_mask.sigmoid(), min=1e-7, max=1-1e-7)
        lidar_clamped = torch.clamp(lidar_mask, min=1e-7, max=1-1e-7)
        mask_loss = loss(radar_sigmoid, lidar_clamped)
        return feature_loss, mask_loss

    def high_loss(self, radar_bev, radar_bev2, lidar_bev, lidar_bev2, high_mask):
        scaled_radar_bev  = radar_bev.softmax(1)
        scaled_lidar_bev  = lidar_bev.softmax(1)
        scaled_radar_bev2 = radar_bev2.softmax(1)
        scaled_lidar_bev2 = lidar_bev2.softmax(1)
        mask_sum = high_mask.sum().clamp(min=1)
        high_loss = F.l1_loss(scaled_radar_bev,  scaled_lidar_bev,  reduction='none') * high_mask
        high_loss = high_loss.sum() / mask_sum
        high_8x_loss = F.l1_loss(scaled_radar_bev2, scaled_lidar_bev2, reduction='none') * high_mask
        high_8x_loss = high_8x_loss.sum() / mask_sum
        return 0.5 * (high_loss + high_8x_loss)

    # ---------------- 마스크(로우/하이) 생성 (원본 유지) ----------------
    def get_low_mask(self, radar_bev, lidar_bev):
        B, _, H, W = radar_bev.shape
        lidar_mask = (lidar_bev.sum(1).unsqueeze(1) > 0).float()
        radar_mask = (radar_bev.sum(1).unsqueeze(1))
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5
        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_lidar[activate_map==1.5] = 1
        mask_radar_de_lidar[activate_map==1.0] = 1
        denom = mask_radar_de_lidar.sum().clamp(min=1)
        mask_radar_de_lidar *= (mask_radar_lidar.sum() / denom).clamp(min=1e-6)
        return mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask

    def get_low_mask_from_agg(self, radar_bev, lidar_bev_agg):
        B, _, H, W = radar_bev.shape
        lidar_mask = (lidar_bev_agg.sum(1, keepdim=True) > 0).float()
        radar_mask = (radar_bev.sum(1, keepdim=True))
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5
        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_lidar[activate_map == 1.5] = 1
        mask_radar_de_lidar[activate_map == 1.0] = 1
        denom = mask_radar_de_lidar.sum().clamp(min=1)
        mask_radar_de_lidar *= (mask_radar_lidar.sum() / denom).clamp(min=1e-6)
        return mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask

    def get_high_mask_from_agg(self, high_lidar_bev, radar_preds, thr_teacher=1e-3, thr_radar=0.1):
        radar_batch_hm = torch.cat([clip_sigmoid(r['hm']) for r in radar_preds], dim=1)
        radar_max = radar_batch_hm.max(dim=1, keepdim=True)[0]
        teach_max = high_lidar_bev.abs().max(dim=1, keepdim=True)[0]
        gt_mask   = (teach_max > thr_teacher)
        radar_pos = (radar_max > thr_radar)
        radar_neg = ~radar_pos
        tp = gt_mask & radar_pos
        fn = gt_mask & radar_neg
        fp = (~gt_mask) & radar_pos
        weight = torch.zeros_like(radar_max, dtype=radar_max.dtype)
        denom_tpfn = (tp | fn).sum().clamp_min(1)
        denom_fp   = fp.sum().clamp_min(1)
        weight[tp | fn] = 5.0 / denom_tpfn
        weight[fp]      = 1.0 / denom_fp
        return weight

    # ---------------- 핵심: get_loss ----------------
    def get_loss(self, batch_dict):
        # Radar (student)
        low_radar_bev     = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x   = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev    = batch_dict['radar_spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        radar_pred_dicts  = batch_dict['radar_pred_dicts']
        gt_heatmaps       = batch_dict['target_dicts']['heatmaps']

        # LiDAR teachers
        T_low   = batch_dict['lidar_teachers_low']    # [B,N,C,H8,W8]
        T_high  = batch_dict['lidar_teachers_high']   # [B,N,C,H, W]
        T_high8 = batch_dict['lidar_teachers_high_8x']
        N = T_low.shape[1]

        # Sweep time info for delta positional embedding
        sweep_times_raw = batch_dict.get('sweep_times', None)
        
        # Normalize sweep times to [0, 1] for stability
        if sweep_times_raw is not None and self.time_normalize:
            sweep_times = sweep_times_raw / max(self.time_max, sweep_times_raw.max().item() + 1e-6)
        else:
            sweep_times = sweep_times_raw

        # 1) Compute gating masks (valid regions for low/high resolution)
        mask_low_gate   = self.get_low_gate_mask(low_radar_bev, T_low)                   # [B,N,1,H8,W8]
        # Use max across teachers (preserves strongest signal) instead of mean
        mask_high_gate1 = self.get_high_gate_mask(T_high.amax(dim=1), radar_pred_dicts) # [B,1,H,W]
        mask_high_gate  = mask_high_gate1.unsqueeze(1).expand(-1, N, -1, -1, -1)         # [B,N,1,H,W]

        # 2) EMA warmup on first batch (prevent initial instability)
        if not self._ema_warmed_up:
            with torch.no_grad():
                # V4 warmup: call update_ema multiple times
                if hasattr(self.gater_low, 'warmup_ema'):
                    self.gater_low.warmup_ema(T_low, steps=5, mask=mask_low_gate)
                    self.gater_high.warmup_ema(T_high, steps=5, mask=mask_high_gate)
                else:
                    # V3 fallback: manual warmup
                    for _ in range(5):
                        self.gater_low.update_ema(T_low, mask=mask_low_gate)
                        self.gater_high.update_ema(T_high, mask=mask_high_gate)
                self._ema_warmed_up = True
        
        # Regular EMA update with valid masks
        with torch.no_grad():
            self.gater_low.update_ema(T_low,  mask=mask_low_gate)
            self.gater_high.update_ema(T_high, mask=mask_high_gate)

        # 3) Compute gating weights with masks & time info
        alpha_low,  aux_low,  stats_low  = self.gater_low(
            S=low_radar_bev, T=T_low, mask=mask_low_gate, return_aux=True,
            sweep_times=sweep_times
        )
        alpha_high, aux_high, stats_high = self.gater_high(
            S=high_radar_bev, T=T_high, mask=mask_high_gate, return_aux=True,
            sweep_times=sweep_times
        )

        # Auto-schedule gating mode & temperature based on epoch
        if 'epoch' in batch_dict:
            epoch = int(batch_dict['epoch'])
            self.gater_low.auto_step(epoch, weights=alpha_low, avg_weights=stats_low['avg_weights'])
            self.gater_high.auto_step(epoch, weights=alpha_high, avg_weights=stats_high['avg_weights'])

        # 4) Interpolate alpha_high to 8x resolution for high_8x features
        B = alpha_high.shape[0]
        H8, W8 = T_high8.shape[-2], T_high8.shape[-1]
        ah4 = alpha_high.view(B*N, 1, alpha_high.shape[-2], alpha_high.shape[-1])
        ah8 = F.interpolate(ah4, size=(H8, W8), mode='bilinear', align_corners=False)
        alpha_high_8x = ah8.view(B, N, 1, H8, W8)

        # 5) Aggregate teacher features using gating weights
        low_lidar_bev     = (alpha_low     * T_low  ).sum(dim=1)  # [B,C,H8,W8]
        high_lidar_bev    = (alpha_high    * T_high ).sum(dim=1)  # [B,C,H,W]
        high_lidar_bev_8x = (alpha_high_8x * T_high8).sum(dim=1)  # [B,C,H8,W8]

        # 6) Compute distillation masks based on aggregated teacher
        mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask = \
            self.get_low_mask_from_agg(low_radar_bev, low_lidar_bev)
        high_mask = self.get_high_mask_from_agg(high_lidar_bev, radar_pred_dicts)

        # 7) Compute distillation losses
        feature_loss, mask_loss = self.low_loss(
            low_lidar_bev, low_radar_bev,
            mask_radar_lidar, mask_radar_de_lidar,
            radar_mask, lidar_mask
        )
        de_8x_feature_loss, de_8x_mask_loss = self.low_loss(
            low_lidar_bev, low_radar_de_8x,
            mask_radar_lidar, mask_radar_de_lidar,
            radar_mask, lidar_mask
        )
        low_distill_loss = (0.5*(feature_loss + de_8x_feature_loss) + 
                            0.5*(mask_loss + de_8x_mask_loss)) * 5.0

        high_distill_loss = self.high_loss(
            high_radar_bev, high_radar_bev_8x,
            high_lidar_bev, high_lidar_bev_8x,
            high_mask
        ) * 25.0

        # 8) Gating auxiliary losses (load balance, entropy, TV smoothing)
        gate_loss = 0.0
        for loss_name, loss_val in aux_low.items():
            gate_loss = gate_loss + loss_val
        for loss_name, loss_val in aux_high.items():
            gate_loss = gate_loss + loss_val

        # Total loss
        distill_loss = low_distill_loss + high_distill_loss + gate_loss

        # 9) Logging statistics
        # IMPORTANT: argmax는 가장 큰 weight 하나만 선택 (top-k=2여도 argmax는 1개만)
        choice_low  = alpha_low.argmax(dim=1)  # [B, H, W]
        choice_high = alpha_high.argmax(dim=1)
        low_counts_t  = torch.stack([(choice_low == i).sum() for i in range(N)]).float()
        high_counts_t = torch.stack([(choice_high == i).sum() for i in range(N)]).float()
        low_total  = int(choice_low.numel())
        high_total = int(choice_high.numel())
        eps = 1e-6
        low_ratios  = (low_counts_t / (float(low_total) + eps)).tolist()
        high_ratios = (high_counts_t / (float(high_total) + eps)).tolist()
        low_counts  = [int(x.item()) for x in low_counts_t]
        high_counts = [int(x.item()) for x in high_counts_t]

        tb_dict = {
            'low_distill_loss': low_distill_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
            'gate_loss': gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
            'distill_loss': distill_loss.item(),
            'gate_low_total': low_total,
            'gate_high_total': high_total,
        }
        # Log counter for periodic metrics
        if hasattr(self, '_log_counter'):
            self._log_counter += 1
        else:
            self._log_counter = 0
        if self._log_counter % 100 == 0 and sweep_times is not None:
            tb_dict['debug_sweep_times_mean'] = float(sweep_times.mean().item())
            tb_dict['debug_sweep_times_max'] = float(sweep_times.max().item())
        # Add individual gating aux losses
        for loss_name, loss_val in aux_low.items():
            tb_dict[f'gate_low_{loss_name}'] = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
        for loss_name, loss_val in aux_high.items():
            tb_dict[f'gate_high_{loss_name}'] = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
        # Per-teacher statistics with sweep time info (for debugging)
        sweeps = batch_dict.get('teacher_sweeps_order', None)
        if isinstance(sweeps, (list, tuple)) and len(sweeps) == N:
            for i, sw in enumerate(sweeps):
                # Sweep-labeled (e.g., s1, s5, s10)
                tb_dict[f'gate_low_s{sw}_count']  = low_counts[i]
                tb_dict[f'gate_high_s{sw}_count'] = high_counts[i]
                tb_dict[f'gate_low_s{sw}_ratio']  = low_ratios[i]
                tb_dict[f'gate_high_s{sw}_ratio'] = high_ratios[i]
                # Index-labeled (backward compatibility)
                tb_dict[f'gate_low_t{i}']  = low_counts[i]
                tb_dict[f'gate_high_t{i}'] = high_counts[i]
                tb_dict[f'gate_low_p{i}']  = low_ratios[i]
                tb_dict[f'gate_high_p{i}'] = high_ratios[i]
        else:
            for i in range(N):
                tb_dict[f'gate_low_t{i}'] = low_counts[i]
                tb_dict[f'gate_high_t{i}'] = high_counts[i]
                tb_dict[f'gate_low_p{i}'] = low_ratios[i]
                tb_dict[f'gate_high_p{i}'] = high_ratios[i]
        
        # Note: Gating visualization은 학습 후 checkpoint에서 생성
        # 학습 중에는 통계(JSON)만 저장 (I/O 최소화)
        
        # ============================================================
        # Paper Statistics: Teacher Selection Evolution (JSON, super fast)
        # ============================================================
        # Accumulate stats for epoch-level summary
        if not hasattr(self, '_epoch_gating_buffer'):
            self._epoch_gating_buffer = {'low': [], 'high': []}
        
        self._epoch_gating_buffer['low'].append(low_ratios)
        self._epoch_gating_buffer['high'].append(high_ratios)
        
        # Save epoch summary when epoch changes
        if 'epoch' in batch_dict and hasattr(self, '_last_saved_epoch'):
            current_epoch = int(batch_dict['epoch'])
            if current_epoch != self._last_saved_epoch and len(self._epoch_gating_buffer['low']) > 0:
                # Compute epoch average
                import json
                from pathlib import Path
                
                low_avg = np.mean(self._epoch_gating_buffer['low'], axis=0).tolist()
                high_avg = np.mean(self._epoch_gating_buffer['high'], axis=0).tolist()
                
                epoch_stats = {
                    'epoch': self._last_saved_epoch,
                    'low_ratios': low_avg,
                    'high_ratios': high_avg,
                    'teacher_sweeps': batch_dict.get('teacher_sweeps_order', [1, 5, 10])
                }
                
                self.gating_stats_history.append(epoch_stats)
                
                # Save to JSON (append mode, very fast)
                stats_file = Path(self.model_cfg.get('VIS_DIR', './output/multisweep/visualizations')) / 'gating_evolution.json'
                stats_file.parent.mkdir(parents=True, exist_ok=True)
                with open(stats_file, 'w') as f:
                    json.dump(self.gating_stats_history, f, indent=2)
                
                # Clear buffer
                self._epoch_gating_buffer = {'low': [], 'high': []}
        
        if 'epoch' in batch_dict:
            self._last_saved_epoch = int(batch_dict['epoch'])
        
        return distill_loss, tb_dict

    def forward(self, data_dict):
        # (원본) Radar BEV 생성 파이프라인
        spatial_features = data_dict['radar_multi_scale_2d_features']['x_conv4']
        ups = []

        en_16x = self.encoder_1(spatial_features)
        de_8x = torch.cat((self.decoder_1(en_16x), spatial_features), dim=1)
        de_8x = self.agg_1(de_8x)
        
        en_32x = self.encoder_2(en_16x)
        de_16x = torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1)
        de_16x = self.agg_2(de_16x)

        x = torch.cat((self.decoder_3(de_16x), de_8x), dim=1)
        x_conv4 = self.agg_3(x)

        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4       # low_radar_bev
        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x         # low_radar_de_8x

        x_conv5 = data_dict['radar_multi_scale_2d_features']['x_conv5']
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        data_dict['radar_spatial_features_2d_8x'] = ups[-1]                                       # high_radar_bev_8x

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        data_dict['radar_spatial_features_2d'] = x                                                # high_radar_bev
        return data_dict
