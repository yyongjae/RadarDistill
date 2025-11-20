from functools import partial
import torch.nn as nn
import torch
from ...utils.spconv_utils import replace_feature, spconv
from .spconv_backbone_2d import post_act_block, SparseBasicBlock, post_act_block_dense, BasicBlock

class Radar_PillarRes18BackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.sparse_shape = grid_size[[1, 0]]
        
        block = post_act_block
        dense_block = post_act_block_dense
        
        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408] <- [800, 704]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704] <- [400, 352]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352] <- [200, 176]
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res4'),
        )
        
        norm_fn = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.01)
        self.conv5 = nn.Sequential(
            # [200, 176] <- [100, 88]
            dense_block(256, 256, 3, norm_fn=norm_fn, stride=2, padding=1),
            BasicBlock(256, 256, norm_fn=norm_fn),
            BasicBlock(256, 256, norm_fn=norm_fn),
        )

        self.num_point_features = 256
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 256,
            'x_conv5': 256
        }

    def forward(self, batch_dict):
        pillar_features, pillar_coords = batch_dict['radar_pillar_features'], batch_dict['radar_pillar_coords']
        batch_size = batch_dict['batch_size']
        
        # DEBUG: Check pillar_features and coords before sparse tensor creation
        if torch.isnan(pillar_features).any() or torch.isinf(pillar_features).any():
            print(f"[BACKBONE DEBUG] NaN/Inf in pillar_features before SparseConvTensor creation")
            print(f"  Shape: {pillar_features.shape}")
            print(f"  NaN count: {torch.isnan(pillar_features).sum().item()}")
            print(f"  Inf count: {torch.isinf(pillar_features).sum().item()}")
            print(f"  Stats: min={pillar_features.min().item()}, max={pillar_features.max().item()}")
            raise RuntimeError("NaN/Inf in pillar_features before sparse tensor creation")
        
        # Check coords validity
        if pillar_coords.min() < 0:
            print(f"[BACKBONE DEBUG] Negative coords detected: min={pillar_coords.min().item()}")
        if pillar_coords[:, 1].max() >= self.sparse_shape[0] or pillar_coords[:, 2].max() >= self.sparse_shape[1]:
            print(f"[BACKBONE DEBUG] Out-of-bounds coords: max_x={pillar_coords[:, 1].max().item()}, max_y={pillar_coords[:, 2].max().item()}")
            print(f"  sparse_shape: {self.sparse_shape}")
        
        input_sp_tensor = spconv.SparseConvTensor(
            features=pillar_features,
            indices=pillar_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        
        # Detailed debugging for conv1
        x = input_sp_tensor
        if torch.isnan(x.features).any():
            print(f"[BACKBONE DEBUG] NaN in input_sp_tensor.features IMMEDIATELY after SparseConvTensor creation")
            # Stop execution if input is already NaN
            batch_dict.update({'radar_multi_scale_2d_features': {'x_conv1': x}})
            return batch_dict

        for i, layer in enumerate(self.conv1):
            x = layer(x)
            if torch.isnan(x.features).any():
                print(f"[BACKBONE DEBUG] NaN in x_conv1.features after layer {i} ({layer.__class__.__name__})")
                # Stop execution to prevent further errors
                batch_dict.update({'radar_multi_scale_2d_features': {'x_conv1': x}})
                return batch_dict

        x_conv1 = x
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        
        x_conv4_dense = x_conv4.dense()
        x_conv5 = self.conv5(x_conv4_dense)

        # Use the original variable name for consistency
        x_conv4 = x_conv4_dense

        batch_dict.update({
            'radar_multi_scale_2d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
                'x_conv5': x_conv5,
            }
        })
        batch_dict.update({
            'radar_multi_scale_2d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
                'x_conv5': 16,
            }
        })
        
        return batch_dict
