# ==============================
# radar_distill_multi_sweep_teacher.py
# (Gater Ver5 적용: Object-wise Relative Knowledge Gain)
# ==============================
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from .base_bev_backbone import BaseBEVBackboneV2

# ▼▼ Gater V5로 교체
from .gating_ver5 import SweepGaterV5

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

        # ---------- Gater V5 설정 ----------
        self.num_teachers = self.model_cfg.get('NUM_TEACHERS', 3)
        
        # Gating Parameters
        gate_temp = self.model_cfg.get('GATE_TEMP', 0.1) # 낮을수록 확신을 가짐
        
        gater_kwargs = dict(
            num_sweeps=self.num_teachers,
            C=256,
            temp=gate_temp
        )
        
        self.gater_low  = SweepGaterV5(**gater_kwargs)
        self.gater_high = SweepGaterV5(**gater_kwargs)
        
        # ---------- Baseline Student 설정 ----------
        self.baseline_model = None
        # Default path
        default_ckpt = '/home/yongjae/4drkd/RadarDistill/output/radar_distill/radar_distill_train/baseline_b8/ckpt/checkpoint_epoch_40.pth'
        baseline_ckpt = self.model_cfg.get('BASELINE_CHECKPOINT', default_ckpt)
        
        if baseline_ckpt is not None:
            print(f"Loading Baseline Student from {baseline_ckpt}...")
            baseline_cfg = copy.deepcopy(self.model_cfg)
            if 'BASELINE_CHECKPOINT' in baseline_cfg:
                del baseline_cfg['BASELINE_CHECKPOINT']
            
            # Instantiate Baseline Student
            self.baseline_model = self.__class__(baseline_cfg, **kwargs)
            
            # Load Weights
            checkpoint = torch.load(baseline_ckpt, map_location='cpu')
            state_dict = checkpoint.get('model_state', checkpoint)
            
            # Load and Freeze
            missing, unexpected = self.baseline_model.load_state_dict(state_dict, strict=False)
            print(f"Baseline Student Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            
            for param in self.baseline_model.parameters():
                param.requires_grad = False
            self.baseline_model.eval()
        else:
            print("WARNING: BASELINE_CHECKPOINT not found! RKG won't work properly.")

    # ---------------- distill losses (원본 유지) ----------------
    def low_loss(self, lidar_bev, radar_bev):
        B, _, H, W = radar_bev.shape
        lidar_mask = (lidar_bev.sum(1).unsqueeze(1) > 0).float()
        radar_mask = (radar_bev.sum(1).unsqueeze(1))
        activate_map = (radar_mask > 0).float() + lidar_mask * 0.5

        mask_radar_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_de_lidar = torch.zeros_like(activate_map, dtype=torch.float)
        mask_radar_lidar[activate_map==1.5] = 1
        mask_radar_de_lidar[activate_map==1.0] = 1

        mask_radar_de_lidar *= (mask_radar_lidar.sum() / mask_radar_de_lidar.sum().clamp(min=1))

        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / max(B, 1)
        
        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / max(B, 1)

        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        loss = nn.L1Loss()
        mask_loss = loss(radar_sigmoid := clip_sigmoid(radar_mask), lidar_mask)

        return feature_loss, mask_loss
    
    def high_loss(self, radar_bev, radar_bev2, lidar_bev, lidar_bev2, heatmaps, radar_preds):
        thres = 0.1
        gt_thres = 0.1
        gt_batch_hm = torch.cat(heatmaps, dim=1)
        gt_batch_hm_max = torch.max(gt_batch_hm, dim=1, keepdim=True)[0]
        
        radar_batch_hm = [(clip_sigmoid(radar_pred_dict['hm'])) for radar_pred_dict in radar_preds]
        radar_batch_hm = torch.cat(radar_batch_hm, dim=1)
        radar_batch_hm_max = torch.max(radar_batch_hm, dim=1, keepdim=True)[0]
        
        radar_fp_mask = torch.logical_and(gt_batch_hm_max < gt_thres, radar_batch_hm_max > thres)
        radar_fn_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max < thres)
        radar_tp_mask = torch.logical_and(gt_batch_hm_max > gt_thres, radar_batch_hm_max > thres)
        
        weight = torch.zeros_like(radar_batch_hm_max)
        weight[radar_tp_mask + radar_fn_mask] = 5 / (radar_tp_mask + radar_fn_mask).sum().clamp(min=1)
        weight[radar_fp_mask] = 1 / (radar_fp_mask).sum().clamp(min=1)
        
        scaled_radar_bev = radar_bev.softmax(1)
        scaled_lidar_bev = lidar_bev.softmax(1)
        scaled_radar_bev2 = radar_bev2.softmax(1)
        scaled_lidar_bev2 = lidar_bev2.softmax(1)
        
        high_loss = F.l1_loss(scaled_radar_bev, scaled_lidar_bev, reduction='none') * weight
        high_loss = high_loss.sum()
        high_8x_loss = F.l1_loss(scaled_radar_bev2, scaled_lidar_bev2, reduction='none') * weight
        high_8x_loss = high_8x_loss.sum()
        return 0.5 * (high_loss + high_8x_loss)
    
    def get_loss(self, batch_dict):
        # 1. Features (Radar Student)
        low_radar_bev = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev = batch_dict['radar_spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        radar_pred_dicts = batch_dict['radar_pred_dicts']
        gt_heatmaps = batch_dict['target_dicts']['heatmaps']
        gt_boxes = batch_dict.get('gt_boxes', None)
        
        # Validation: gt_boxes required for Object-wise Gating
        if gt_boxes is None:
            raise ValueError("gt_boxes is required for Gating Ver5 (Object-wise RKG)")

        # LiDAR Teachers (Ver4 compatible - already stacked format)
        T_low = batch_dict['lidar_teachers_low']      # [B, N, C, H8, W8]
        T_high = batch_dict['lidar_teachers_high']    # [B, N, C, H, W]
        T_high8 = batch_dict['lidar_teachers_high_8x'] # [B, N, C, H8, W8]
        N = T_low.shape[1]
        
        # 2. Baseline Inference
        if self.baseline_model is not None:
            batch_dict_base = batch_dict.copy()
            batch_dict_base['radar_multi_scale_2d_features'] = batch_dict['radar_multi_scale_2d_features'].copy()
            with torch.no_grad():
                self.baseline_model(batch_dict_base)
            base_low_radar_bev = batch_dict_base['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        else:
            base_low_radar_bev = low_radar_bev.detach()

        # 3. Gating (Object-wise)
        gt_boxes_info = {
            'boxes': gt_boxes,
            'pc_range': self.point_cloud_range
        }
        
        # weights_low: [B, N, 1, H, W] - 객체 영역만 채워져 있음 (배경은 0)
        weights_low, _, stats_low = self.gater_low(
            S_curr=low_radar_bev,
            S_base=base_low_radar_bev,
            T=T_low,
            gt_boxes_info=gt_boxes_info
        )

        # 4. Aggregation
        # Object-wise weighted sum (배경은 0)
        low_lidar_bev_agg = (weights_low * T_low).sum(dim=1)  # [B, C, H, W]
        
        # High Level: 단순히 s10 사용 (가장 정보량이 많은 sweep)
        # NOTE: High Level에도 Object-wise Gating을 적용하려면 self.gater_high 추가 필요
        high_lidar_bev = T_high[:, -1, :, :, :]      # [B, C, H, W] - s10
        high_lidar_bev_8x = T_high8[:, -1, :, :, :]  # [B, C, H8, W8] - s10

        # 5. Loss Calculation
        feature_loss, mask_loss = self.low_loss(low_lidar_bev_agg, low_radar_bev)
        de_8x_feature_loss, de_8x_mask_loss = self.low_loss(low_lidar_bev_agg, low_radar_de_8x)
        
        high_distill_loss = self.high_loss(high_radar_bev, high_radar_bev_8x, high_lidar_bev, high_lidar_bev_8x, gt_heatmaps, radar_pred_dicts)
        high_distill_loss *= 25
        
        low_distill_loss = 0.5 * (feature_loss + de_8x_feature_loss) + 0.5 * (mask_loss + de_8x_mask_loss)
        low_distill_loss *= 5
        
        distill_loss = low_distill_loss + high_distill_loss
        
        # 6. Logging - Object-wise sweep selection statistics
        # weights_low: [B, N, 1, H, W] - extract argmax for dominant teacher per pixel
        choice_low = weights_low.argmax(dim=1).squeeze(1)  # [B, H, W]
        
        # Filter out background (where all weights are 0)
        # Create mask: sum across N dimension to find non-zero regions
        weight_sum = weights_low.sum(dim=1).squeeze(1)  # [B, H, W]
        valid_mask = (weight_sum > 1e-6)  # [B, H, W] - True for object regions
        
        # Count selections only in valid (object) regions
        low_counts_t = torch.stack([
            ((choice_low == i) & valid_mask).sum() for i in range(N)
        ]).float()
        
        low_total = int(valid_mask.sum().item())  # Total object pixels
        eps = 1e-6
        low_ratios = (low_counts_t / (float(low_total) + eps)).tolist()
        low_counts = [int(x.item()) for x in low_counts_t]
        
        tb_dict = {
            'low_feature_loss': low_distill_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
            'distill_loss': distill_loss.item(),
            'gate_low_total': low_total,  # Total object pixels
        }
        
        # Add RKG stats for all teachers with actual sweep numbers (dynamic)
        sweeps = batch_dict.get('teacher_sweeps_order', None)
        if sweeps is not None and len(sweeps) == N:
            # All teachers with sweep-specific labels
            for i, sw in enumerate(sweeps):
                tb_dict[f'rkg/gain_s{sw}'] = stats_low[f'gain_t{i}']
                tb_dict[f'rkg/prob_s{sw}'] = stats_low[f'prob_t{i}']
        else:
            # Fallback to generic index-based names
            for i in range(N):
                tb_dict[f'rkg/gain_t{i}'] = stats_low[f'gain_t{i}']
                tb_dict[f'rkg/prob_t{i}'] = stats_low[f'prob_t{i}']
        
        # Per-teacher selection statistics (argmax-based)
        if isinstance(sweeps, (list, tuple)) and len(sweeps) == N:
            for i, sw in enumerate(sweeps):
                # Sweep-labeled (e.g., s1, s5, s10)
                tb_dict[f'gate_low_s{sw}_count'] = low_counts[i]
                tb_dict[f'gate_low_s{sw}_ratio'] = low_ratios[i]
                # Index-labeled (backward compatibility)
                tb_dict[f'gate_low_t{i}'] = low_counts[i]
                tb_dict[f'gate_low_p{i}'] = low_ratios[i]
        else:
            for i in range(N):
                tb_dict[f'gate_low_t{i}'] = low_counts[i]
                tb_dict[f'gate_low_p{i}'] = low_ratios[i]
        
        return distill_loss, tb_dict
    
    def forward(self, data_dict):
        # (원본 그대로 유지)
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

        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4
        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x
        
        x_conv5 = data_dict['radar_multi_scale_2d_features']['x_conv5']
        
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        data_dict['radar_spatial_features_2d_8x'] = ups[-1]

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        
        data_dict['radar_spatial_features_2d'] = x
        return data_dict