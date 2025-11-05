import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pcdet.utils.box_utils import center_to_corner_box2d
from ...ops.basicblock.modules.Basicblock_convn import ConvNeXtBlock
from functools import partial
import cv2
from .base_bev_backbone import BaseBEVBackboneV2
from pcdet.models.backbones_2d.sweep_gater import SweepGater


def clip_sigmoid(x, eps=1e-4):
    """Sigmoid function for input feature.

    Args:
        x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
        eps (float): Lower bound of the range to be clamped to. Defaults
            to 1e-4.

    Returns:
        torch.Tensor: Feature map after sigmoid.
    """
    # FIXME change back!
    # y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    y = torch.clamp(x.sigmoid(), min=eps, max=1 - eps)
    return y
            

class Radar_Distill_Multi_Sweep_Teacher(BaseBEVBackboneV2):
    def __init__(self, model_cfg, **kwargs):
        super().__init__(model_cfg, **kwargs)
        self.model_cfg = model_cfg
        
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

        self.sweep_gater_low  = SweepGater(C=256, num_sweeps=3, temp=0.7, use_adapter=True)
        self.sweep_gater_high = SweepGater(C=256, num_sweeps=3, temp=0.7, use_adapter=True)

    # ----------------[원본 유지] 기존 게이트용 마스크----------------
    def get_low_gate_mask(self, low_radar_bev, T_low, tau=1e-3):
        """
        low_radar_bev: [B,C,H8,W8]
        T_low:        [B,3,C,H8,W8]   # teacher 스택
        return:       [B,3,1,H8,W8]
        """
        radar_occ8 = (low_radar_bev.sum(1, keepdim=True) > 0).float()              # [B,1,H8,W8]
        teach_occ8 = (T_low.abs().max(dim=2, keepdim=True)[0] > tau).float()       # [B,3,1,H8,W8]
        mask_low   = teach_occ8 * radar_occ8.unsqueeze(1)                          # [B,3,1,H8,W8]
        return mask_low

    # ----------------[원본 유지] 기존 로스/마스크----------------
    def low_loss(self, lidar_bev, radar_bev, mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask):
        B, _, H, W = radar_bev.shape

        loss_radar_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_lidar = torch.sum(loss_radar_lidar * mask_radar_lidar) / B
        
        loss_radar_de_lidar = F.mse_loss(radar_bev, lidar_bev, reduction='none')
        loss_radar_de_lidar = torch.sum(loss_radar_de_lidar * mask_radar_de_lidar) / B

        feature_loss = 3e-4 * loss_radar_lidar + 5e-5 * loss_radar_de_lidar
        loss = nn.L1Loss()
        mask_loss = loss(radar_mask.sigmoid(), lidar_mask)

        return feature_loss, mask_loss

    def high_loss(self, radar_bev, radar_bev2, lidar_bev, lidar_bev2, high_mask):
        scaled_radar_bev = radar_bev.softmax(1)
        scaled_lidar_bev = lidar_bev.softmax(1)
        
        scaled_radar_bev2 = radar_bev2.softmax(1)
        scaled_lidar_bev2 = lidar_bev2.softmax(1)
        
        high_loss = F.l1_loss(scaled_radar_bev, scaled_lidar_bev, reduction='none') * high_mask
        high_loss = high_loss.sum()
        high_8x_loss = F.l1_loss(scaled_radar_bev2, scaled_lidar_bev2, reduction='none') * high_mask
        high_8x_loss = high_8x_loss.sum()
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

    # ----------------[NEW] 집계 teacher 기준 마스크----------------
    def get_low_mask_from_agg(self, radar_bev, lidar_bev_agg):
        """
        radar_bev:     [B,C,H8,W8]   (student)
        lidar_bev_agg: [B,C,H8,W8]   (alpha로 집계된 teacher)
        """
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

    # def get_high_mask_from_agg(self, high_lidar_bev_agg, radar_preds, th_obj=0.1, th_pred=0.1):
    #     """
    #     high_lidar_bev_agg: [B,C,H,W]  (alpha로 집계된 teacher @ 1x)
    #     radar_preds: list of dicts with 'hm'
    #     return: weight [B,1,H,W]
    #     """
    #     t_hm = high_lidar_bev_agg.softmax(1).max(dim=1, keepdim=True)[0]   # [B,1,H,W]
    #     s_hm = torch.cat([clip_sigmoid(d['hm']) for d in radar_preds], dim=1).max(dim=1, keepdim=True)[0]  # [B,1,H,W]

    #     radar_tp_mask = torch.logical_and(t_hm > th_obj, s_hm > th_pred)
    #     radar_fn_mask = torch.logical_and(t_hm > th_obj, s_hm < th_pred)
    #     radar_fp_mask = torch.logical_and(t_hm < th_obj, s_hm > th_pred)

    #     weight = torch.zeros_like(s_hm)
    #     denom_tpfn = (radar_tp_mask | radar_fn_mask).sum().clamp(min=1)
    #     denom_fp   = (radar_fp_mask).sum().clamp(min=1)
    #     weight[radar_tp_mask | radar_fn_mask] = 5.0 / denom_tpfn
    #     weight[radar_fp_mask] = 1.0 / denom_fp
    #     return weight

    def get_high_mask_from_agg(self, high_lidar_bev, radar_preds, thr_teacher=1e-3, thr_radar=0.1):
        """
        high_lidar_bev: [B,C,H,W]  (게이팅으로 집계된 teacher)
        radar_preds   : list of dicts with 'hm' -> [B,?,H,W]
        return        : weight [B,1,H,W]
        """
        # radar heatmap max over classes: [B,1,H,W]
        radar_batch_hm = torch.cat([clip_sigmoid(r['hm']) for r in radar_preds], dim=1)
        radar_max = radar_batch_hm.max(dim=1, keepdim=True)[0]

        # teacher occupancy from aggregated teacher: [B,1,H,W]
        teach_max = high_lidar_bev.abs().max(dim=1, keepdim=True)[0]
        gt_mask   = (teach_max > thr_teacher)

        # radar pos/neg
        radar_pos = (radar_max > thr_radar)
        radar_neg = ~radar_pos

        # regions
        tp = gt_mask & radar_pos
        fn = gt_mask & radar_neg
        fp = (~gt_mask) & radar_pos

        # weights with correct batch shape
        weight = torch.zeros_like(radar_max, dtype=radar_max.dtype)

        denom_tpfn = (tp | fn).sum().clamp_min(1)
        denom_fp   = fp.sum().clamp_min(1)

        weight[tp | fn] = 5.0 / denom_tpfn
        weight[fp]      = 1.0 / denom_fp
        return weight


    def get_loss(self, batch_dict):
        # ---------- Radar (student) ----------
        low_radar_bev     = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_2']  # [B,C,H8,W8]
        low_radar_de_8x   = batch_dict['radar_multi_scale_2d_features']['radar_spatial_features_8x_1']  # [B,C,H8,W8]
        high_radar_bev    = batch_dict['radar_spatial_features_2d']                                     # [B,C,H,W]
        high_radar_bev_8x = batch_dict['radar_spatial_features_2d_8x']                                  # [B,C,H8,W8]

        radar_pred_dicts  = batch_dict['radar_pred_dicts']
        gt_heatmaps       = batch_dict['target_dicts']['heatmaps']
        orig_gt_boxes     = batch_dict['orig_gt_boxes']

        # ---------- LiDAR teacher stacks ----------
        # [B, 3, C, H8, W8], [B, 3, C, H, W], [B, 3, C, H8, W8]
        T_low   = batch_dict['lidar_teachers_low']
        T_high  = batch_dict['lidar_teachers_high']
        T_high8 = batch_dict['lidar_teachers_high_8x']

        # ---------- gating (no-grad) ----------
        with torch.no_grad():
            # (원본 유지) 게이트 통계는 스택 기준으로 업데이트
            mask_low_gate  = self.get_low_gate_mask(low_radar_bev, T_low)
            mask_high_gate = self.get_high_mask(gt_heatmaps, radar_pred_dicts).unsqueeze(1).expand(-1, 3, -1, -1, -1)
            self.sweep_gater_low.update_ema(T_low,  mask_low_gate)
            self.sweep_gater_high.update_ema(T_high, mask_high_gate)

        # α 계산
        alpha_low  = self.sweep_gater_low (S=low_radar_bev,  T=T_low)   # [B,3,1,H8,W8]
        alpha_high = self.sweep_gater_high(S=high_radar_bev, T=T_high)  # [B,3,1,H, W]

        # α를 8x 해상도로 보간
        B = alpha_high.shape[0]
        H8, W8 = T_high8.shape[-2], T_high8.shape[-1]
        ah4 = alpha_high.view(B*3, 1, alpha_high.shape[-2], alpha_high.shape[-1])
        ah8 = F.interpolate(ah4, size=(H8, W8), mode='bilinear', align_corners=False)
        alpha_high_8x = ah8.view(B, 3, 1, H8, W8)

        # ---------- 집계된 LiDAR feature(teacher) ----------
        low_lidar_bev      = (alpha_low     * T_low  ).sum(dim=1)  # [B,C,H8,W8]
        high_lidar_bev     = (alpha_high    * T_high ).sum(dim=1)  # [B,C,H, W]
        high_lidar_bev_8x  = (alpha_high_8x * T_high8).sum(dim=1)  # [B,C,H8,W8]

        # ---------- (집계 teacher 기준) 마스크 재계산 ----------
        mask_radar_lidar, mask_radar_de_lidar, radar_mask, lidar_mask = \
            self.get_low_mask_from_agg(low_radar_bev, low_lidar_bev)

        high_mask = self.get_high_mask_from_agg(high_lidar_bev, radar_pred_dicts)

        # (선택) 디버깅 로그
        choice_low  = alpha_low.argmax(dim=1)
        choice_high = alpha_high.argmax(dim=1)
        low_counts_t  = torch.stack([(choice_low  == i).sum() for i in range(3)]).float()
        high_counts_t = torch.stack([(choice_high == i).sum() for i in range(3)]).float()
        low_total   = int(choice_low.numel());   high_total  = int(choice_high.numel())
        eps = 1e-6
        low_ratios  = (low_counts_t  / (float(low_total)  + eps)).tolist()
        high_ratios = (high_counts_t / (float(high_total) + eps)).tolist()
        low_counts  = [int(x.item()) for x in low_counts_t]
        high_counts = [int(x.item()) for x in high_counts_t]

        # ---------- distill losses ----------
        # low (두 갈래 평균)
        feature_loss,        mask_loss        = self.low_loss(low_lidar_bev, low_radar_bev,
                                                              mask_radar_lidar, mask_radar_de_lidar,
                                                              radar_mask, lidar_mask)
        de_8x_feature_loss,  de_8x_mask_loss  = self.low_loss(low_lidar_bev, low_radar_de_8x,
                                                              mask_radar_lidar, mask_radar_de_lidar,
                                                              radar_mask, lidar_mask)
        low_distill_loss = (0.5*(feature_loss + de_8x_feature_loss) + 0.5*(mask_loss + de_8x_mask_loss)) * 5.0

        # high (1x/8x)
        high_distill_loss = self.high_loss(
            high_radar_bev,  high_radar_bev_8x,
            high_lidar_bev,  high_lidar_bev_8x,
            high_mask
        ) * 25.0

        # (선택) SSKD/채널-wise 등 추가 항은 필요 시 플러그인
        distill_loss = low_distill_loss + high_distill_loss

        tb_dict = {
            'low_feature_loss': low_distill_loss.item(),
            'high_distill_loss': high_distill_loss.item(),
            'distll_loss': distill_loss.item(),
            'low_distill_de_8x_loss': de_8x_feature_loss.item(),
            'low_distill_loss': feature_loss.item(),
            'mask_loss': mask_loss.item(),
            'mask_de_8x_loss': de_8x_mask_loss.item(),

            # gating debug (원본 이름 유지)
            'gate_low_s8':  low_counts[0],  'gate_low_s9':  low_counts[1],  'gate_low_s10': low_counts[2],
            'gate_low_total': low_total,
            'gate_high_s8': high_counts[0], 'gate_high_s9': high_counts[1], 'gate_high_s10': high_counts[2],
            'gate_high_total': high_total,
            'gate_low_p8':  low_ratios[0],  'gate_low_p9':  low_ratios[1],  'gate_low_p10': low_ratios[2],
            'gate_high_p8': high_ratios[0], 'gate_high_p9': high_ratios[1], 'gate_high_p10': high_ratios[2],
        }
        return distill_loss, tb_dict

    
    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
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
