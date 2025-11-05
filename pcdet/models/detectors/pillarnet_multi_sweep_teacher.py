from .detector3d_template import Detector3DTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import FocalLoss
import torch
import torch.nn as nn
from ...utils import loss_utils
import time
from pathlib import Path
import os
import numpy as np


class PillarNet_Multi_Sweep_Teacher(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_cfg = model_cfg

        # [NEW] teacher / student 모듈 인덱스 설정 (네 구조 기준)
        self.teacher_feat_mod_idx = [0, 2, 4]
        self.teacher_skip_idx     = [0, 2, 4, 6]

        if self.model_cfg.get('FREEZE_PIPELINE', None) is not None:
            self.no_grad_module = model_cfg['FREEZE_PIPELINE']
            for i, cur_module in enumerate(self.module_list):
                cur_name = cur_module.__class__.__name__
                if cur_name in self.no_grad_module:
                    for param in cur_module.parameters():
                        param.requires_grad = False
        else:
            self.no_grad_module = []

    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------- [NEW] helpers ------------------------------- #
    def _collate_teacher_points(self, batch_dict, key: str):
        """
        batch_dict[key]: list of (Ni, C) np/torch
        return: torch[(sum Ni), C+1]  (맨 앞열에 batch_idx 추가), 모델 디바이스로
        """
        pts_cat = []
        for b, pts in enumerate(batch_dict[key]):  # len == B
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()
            if pts.size == 0:
                continue
            pts = pts.astype(np.float32, copy=False)
            bi = np.full((pts.shape[0], 1), b, dtype=np.float32)
            pts_cat.append(np.concatenate([bi, pts], axis=1))
        if not pts_cat:
            return torch.zeros((0, 5), dtype=torch.float32, device=self.device)
        out = torch.from_numpy(np.concatenate(pts_cat, axis=0)).to(self.device)
        return out

    @torch.no_grad()
    def _forward_teacher_once(self, batch_dict, points_key: str):
        tbd = {
            'batch_size': batch_dict['batch_size'],
            'frame_id':   batch_dict['frame_id'],
            'metadata':   batch_dict.get('metadata', None),
            'points':     self._collate_teacher_points(batch_dict, points_key),
        }
        for idx in self.teacher_feat_mod_idx:  # teacher VFE/백본/넥/헤드
            tbd = self.module_list[idx](tbd)
        return tbd

    def _build_teacher_stacks_inplace(self, batch_dict):
        """teacher_points_s8/s9/s10로부터 [B,3,C,H,W] 스택 생성 후 batch_dict에 저장"""
        keys = ['teacher_points_s8', 'teacher_points_s9', 'teacher_points_s10']
        lows, highs, highs8 = [], [], []
        for k in keys:
            tbd = self._forward_teacher_once(batch_dict, k)
            lows.append(  tbd['multi_scale_2d_features']['x_conv4'] )   # [B,C,H8,W8]
            highs.append( tbd['spatial_features_2d'] )                   # [B,C,H,W]
            highs8.append(tbd['spatial_features_2d_8x'] )                # [B,C,H8,W8]
        batch_dict['lidar_teachers_low']      = torch.stack(lows,  dim=1)   # [B,3,C,H8,W8]
        batch_dict['lidar_teachers_high']     = torch.stack(highs, dim=1)   # [B,3,C,H,W]
        batch_dict['lidar_teachers_high_8x']  = torch.stack(highs8,dim=1)   # [B,3,C,H8,W8]
    # ----------------------------------------------------------------------------- #

    def forward(self, batch_dict):
        # [NEW] distillation 훈련 시: 먼저 teacher BEV 스택을 만들어 batch_dict에 주입
        if self.training and self.model_cfg.get('DISTILL', False):
            self._build_teacher_stacks_inplace(batch_dict)

        # 메인 파이프라인: 교사 모듈(0,2,4)은 스킵하고 학생 + teacher head만 실행
        for idx, cur_module in enumerate(self.module_list):
            if self.training and self.model_cfg.get('DISTILL', False) and idx in self.teacher_skip_idx:
                continue
            cur_name = cur_module.__class__.__name__
            if cur_name in self.no_grad_module:
                cur_module.eval()
            batch_dict = cur_module(batch_dict)

        if self.training:
            if self.model_cfg.get('DISTILL', None) is None:
                loss, tb_dict, disp_dict = self.get_training_loss()
            elif self.model_cfg.get('DISTILL', None):
                loss, tb_dict, disp_dict = self.get_training_distll_loss(batch_dict)
            else:
                loss, tb_dict, disp_dict = self.get_training_wo_distll_loss(batch_dict)

            ret_dict = {'loss': loss}
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {'loss_rpn': loss_rpn.item(), **tb_dict}
        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def get_training_distll_loss(self, batch_dict):
        disp_dict = {}
        loss_feature, tb_dict = self.radar_backbone_2d.get_loss(batch_dict)
        loss_rpn, _tb_dict = self.radar_dense_head.get_loss()
        tb_dict.update(_tb_dict)
        loss = loss_feature + loss_rpn
        return loss, tb_dict, disp_dict

    def get_training_wo_distll_loss(self, batch_dict):
        disp_dict = {}
        loss_rpn, tb_dict = self.radar_dense_head.get_loss()
        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']
            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )
        return final_pred_dict, recall_dict