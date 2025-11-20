# ============================== 
# 원본run_ver3.py  (Ver3 게이팅 적용본)
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

# [CHANGED] Ver2 → Ver3 게이터로 교체
# 기존: from pcdet.models.backbones_2d.sweep_gater import SweepGater
# 경로는 프로젝트 구조에 따라 조정하세요. (예: 같은 디렉토리에 ver3.py가 있다면 .ver3)
from .gating import SweepGaterV3


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid(), min=eps, max=1 - eps)
    return y
            

class Radar_Distill_Multi_Sweep_Teacher(BaseBEVBackboneV2):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.model_cfg = model_cfg
        
        # (원본 유지) encoder/decoder/agg
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

        # [CHANGED] Ver3 게이터 생성 - 동적 teacher 개수 지원
        # TEACHER_SWEEPS: [1, 5, 8, 10] → 4 teachers
        self.num_teachers = self.model_cfg.get('NUM_TEACHERS', 4)  # Default: 4 teachers
        
        self.gater_low  = SweepGaterV3(
            C=256, num_sweeps=self.num_teachers, temp=0.7, use_adapter=True,
            mode='soft', k=2, lb_coeff=0.0, ent_coeff=0.0, tv_coeff=0.0,
            learn_router=True
        )
        self.gater_high = SweepGaterV3(
            C=256, num_sweeps=self.num_teachers, temp=0.7, use_adapter=True,
            mode='soft', k=2, lb_coeff=0.0, ent_coeff=0.0, tv_coeff=0.0,
            learn_router=True
        )

    # ----------------[원본 유지] 기존 게이트용 마스크----------------
    def get_low_gate_mask(self, low_radar_bev, T_low, tau=1e-3):
        """
        low_radar_bev: [B,C,H8,W8]
        T_low:        [B,N,C,H8,W8]   # teacher 스택 (N=num_teachers)
        return:       [B,N,1,H8,W8]
        """
        radar_occ8 = (low_radar_bev.sum(1, keepdim=True) > 0).float()              # [B,1,H8,W8]
        teach_occ8 = (T_low.abs().max(dim=2, keepdim=True)[0] > tau).float()       # [B,N,1,H8,W8]
        mask_low   = teach_occ8 * radar_occ8.unsqueeze(1)                          # [B,N,1,H8,W8]
        return mask_low

    # ----------------[원본 유지] 기존 로스/마스크----------------
    def low_loss(self, lidar_bev, radar_bev, mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask):
        B, _, H, W = radar_bev.shape

        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        # Prevent division by zero
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / max(B, 1)
        
        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / max(B, 1)

        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        loss = nn.L1Loss()
        # Clamp sigmoid to prevent extreme values
        radar_sigmoid = torch.clamp(radar_mask.sigmoid(), min=1e-7, max=1-1e-7)
        lidar_clamped = torch.clamp(lidar_mask, min=1e-7, max=1-1e-7)
        mask_loss = loss(radar_sigmoid, lidar_clamped)

        return feature_loss, mask_loss

    def high_loss(self, radar_bev, radar_bev2, lidar_bev, lidar_bev2, high_mask):
        scaled_radar_bev = radar_bev.softmax(1)
        scaled_lidar_bev = lidar_bev.softmax(1)
        
        scaled_radar_bev2 = radar_bev2.softmax(1)
        scaled_lidar_bev2 = lidar_bev2.softmax(1)
        
        # Normalize by mask sum to prevent explosion when mask is very sparse
        mask_sum = high_mask.sum().clamp(min=1)
        
        high_loss = F.l1_loss(scaled_radar_bev, scaled_lidar_bev, reduction='none') * high_mask
        high_loss = high_loss.sum() / mask_sum
        
        high_8x_loss = F.l1_loss(scaled_radar_bev2, scaled_lidar_bev2, reduction='none') * high_mask
        high_8x_loss = high_8x_loss.sum() / mask_sum
        
        high_loss = 0.5 * (high_loss + high_8x_loss)
        return high_loss

    def get_high_mask(self, heatmaps, radar_preds):
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
        weight[radar_tp_mask + radar_fn_mask] = 5 /(radar_tp_mask + radar_fn_mask).sum().clamp(min=1)
        weight[radar_fp_mask] = 1 / (radar_fp_mask).sum().clamp(min=1)

        return weight

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

    # ----------------[NEW 유지] 집계 teacher 기준 마스크----------------
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

    
    # ===== 핵심: Option A가 적용된 get_loss =====
    def get_loss(self, batch_dict):
        # Radar (student)
        low_radar_bev     = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']
        low_radar_de_8x   = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']
        high_radar_bev    = batch_dict['radar_spatial_features_2d']
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']
        radar_pred_dicts  = batch_dict['radar_pred_dicts']
        gt_heatmaps       = batch_dict['target_dicts']['heatmaps']

        # LiDAR teachers (N-sweep stacks, N=num_teachers)
        T_low   = batch_dict['lidar_teachers_low']   # [B,N,C,H8,W8]
        T_high  = batch_dict['lidar_teachers_high']  # [B,N,C,H, W]
        T_high8 = batch_dict['lidar_teachers_high_8x']
        N = T_low.shape[1]  # Dynamic teacher count

        # 1) EMA 업데이트는 마스크 사용 (원본과 동일한 통계 경로)
        with torch.no_grad():
            mask_low_gate  = self.get_low_gate_mask(low_radar_bev, T_low)
            mask_high_gate = self.get_high_mask(gt_heatmaps, radar_pred_dicts).unsqueeze(1).expand(-1, N, -1, -1, -1)
            self.gater_low.update_ema(T_low,  mask_low_gate)
            self.gater_high.update_ema(T_high, mask_high_gate)

        # 2) 게이트 계산에는 mask=None (Option A 핵심)
        alpha_low,  aux_low,  stats_low  = self.gater_low (S=low_radar_bev,  T=T_low,  mask=None, return_aux=True)
        alpha_high, aux_high, stats_high = self.gater_high(S=high_radar_bev, T=T_high, mask=None, return_aux=True)

        # (선택) 스케줄 사용 시: epoch가 배치에 있으면 auto_step
        if 'epoch' in batch_dict:
            epoch = int(batch_dict['epoch'])
            self.gater_low.auto_step (epoch, weights=alpha_low,  avg_weights=stats_low['avg_weights'])
            self.gater_high.auto_step(epoch, weights=alpha_high, avg_weights=stats_high['avg_weights'])

        # α를 8x 해상도로 보간 (동적 teacher 개수)
        B = alpha_high.shape[0]
        H8, W8 = T_high8.shape[-2], T_high8.shape[-1]
        ah4 = alpha_high.view(B*N, 1, alpha_high.shape[-2], alpha_high.shape[-1])
        ah8 = F.interpolate(ah4, size=(H8, W8), mode='bilinear', align_corners=False)
        alpha_high_8x = ah8.view(B, N, 1, H8, W8)

        # 3) 집계된 LiDAR feature(teacher)
        low_lidar_bev      = (alpha_low     * T_low  ).sum(dim=1)
        high_lidar_bev     = (alpha_high    * T_high ).sum(dim=1)
        high_lidar_bev_8x  = (alpha_high_8x * T_high8).sum(dim=1)

        # 4) 집계 teacher 기준 마스크 재계산 (원본 유지)
        mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask = \
            self.get_low_mask_from_agg(low_radar_bev, low_lidar_bev)
        high_mask = self.get_high_mask_from_agg(high_lidar_bev, radar_pred_dicts)

        # 5) 디버깅 로그 (동적 teacher 개수)
        choice_low  = alpha_low.argmax(dim=1)
        choice_high = alpha_high.argmax(dim=1)
        low_counts_t  = torch.stack([(choice_low  == i).sum() for i in range(N)]).float()
        high_counts_t = torch.stack([(choice_high == i).sum() for i in range(N)]).float()
        low_total   = int(choice_low.numel());   high_total  = int(choice_high.numel())
        eps = 1e-6
        low_ratios  = (low_counts_t  / (float(low_total)  + eps)).tolist()
        high_ratios = (high_counts_t / (float(high_total) + eps)).tolist()
        low_counts  = [int(x.item()) for x in low_counts_t]
        high_counts = [int(x.item()) for x in high_counts_t]

        # 6) distill losses (원본 유지)
        feature_loss,        mask_loss        = self.low_loss(low_lidar_bev, low_radar_bev,
                                                              mask_radar_lidar, mask_radar_de_lidar,
                                                              radar_mask, lidar_mask)
        de_8x_feature_loss,  de_8x_mask_loss  = self.low_loss(low_lidar_bev, low_radar_de_8x,
                                                              mask_radar_lidar, mask_radar_de_lidar,
                                                              radar_mask, lidar_mask)
        low_distill_loss = (0.5*(feature_loss + de_8x_feature_loss) + 0.5*(mask_loss + de_8x_mask_loss)) * 5.0

        high_distill_loss = self.high_loss(
            high_radar_bev,  high_radar_bev_8x,
            high_lidar_bev,  high_lidar_bev_8x,
            high_mask
        ) * 25.0

        distill_loss = low_distill_loss + high_distill_loss

        # (참고) aux losses는 흐름 유지 위해 합산하지 않음. 필요시 distill_loss += sum(aux_*.*)

        # Dynamic tensorboard dict generation
        tb_dict = {
            'low_feature_loss': low_distill_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
            'distll_loss': distill_loss.item(),
            'gate_low_total': low_total,
            'gate_high_total': high_total,
        }
        
        # Add per-teacher counts and ratios with sweep labels if available
        sweeps = batch_dict.get('teacher_sweeps_order', None)
        if isinstance(sweeps, (list, tuple)) and len(sweeps) == N:
            for i, sw in enumerate(sweeps):
                # Label with actual sweep numbers (e.g., s1/s5/s10)
                tb_dict[f'gate_low_s{sw}'] = low_counts[i]
                tb_dict[f'gate_high_s{sw}'] = high_counts[i]
                tb_dict[f'gate_low_ps{sw}'] = low_ratios[i]
                tb_dict[f'gate_high_ps{sw}'] = high_ratios[i]
                # Keep index-based keys for backward compatibility
                tb_dict[f'gate_low_t{i}'] = low_counts[i]
                tb_dict[f'gate_high_t{i}'] = high_counts[i]
                tb_dict[f'gate_low_p{i}'] = low_ratios[i]
                tb_dict[f'gate_high_p{i}'] = high_ratios[i]
        else:
            for i in range(N):
                tb_dict[f'gate_low_t{i}'] = low_counts[i]
                tb_dict[f'gate_high_t{i}'] = high_counts[i]
                tb_dict[f'gate_low_p{i}'] = low_ratios[i]
                tb_dict[f'gate_high_p{i}'] = high_ratios[i]
        return distill_loss, tb_dict
    
    def forward(self, data_dict):
        """
        (원본 유지) Radar BEV 생성 파이프라인
        """
        spatial_features = data_dict['radar_multi_scale_2d_features']['x_conv4']
        ups = []
        ret_dict = {}
        
        en_16x = self.encoder_1(spatial_features) #(B, 256, 90, 90)
        de_8x = torch.cat((self.decoder_1(en_16x), spatial_features), dim=1)#(B,512,180,180)
        de_8x = self.agg_1(de_8x)#(B,256,180,180)
        
        en_32x = self.encoder_2(en_16x)#(B,256,45,45)
        de_16x = torch.cat((self.decoder_2(en_32x), self.encoder_3(de_8x)), dim=1)#(B,512,90,90)
        de_16x = self.agg_2(de_16x)#(B,256,90,90)

        x = torch.cat((self.decoder_3(de_16x), de_8x), dim=1)#(B, 512, 180, 180)
        x_conv4 = self.agg_3(x)

        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2'] = x_conv4 # low_radar_bev
        data_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1'] = de_8x # low_radar_de_8x

        
        x_conv5 = data_dict['radar_multi_scale_2d_features']['x_conv5']
        
        ups = [x_conv4]
        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[0](x))
        data_dict['radar_spatial_features_2d_8x'] = ups[-1] # high_radar_bev_8x

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        data_dict['radar_spatial_features_2d'] = x # high_radar_bev

        return data_dict
