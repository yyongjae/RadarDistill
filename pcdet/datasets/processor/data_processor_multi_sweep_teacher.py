from functools import partial

import numpy as np
from skimage import transform
import torch
import torchvision
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor_Multi_Sweep_Teacher(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features, radar_num_point_features=6):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.radar_num_point_features = radar_num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []
        self.voxel_generator = None
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    @staticmethod
    def _get_teacher_keys(data_dict):
        """Dynamically detect teacher point keys from data_dict"""
        return [k for k in data_dict.keys() if k.startswith('teacher_points_s')]

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        # teacher 포인트 동적 범위 마스크
        for k in self._get_teacher_keys(data_dict):
            if data_dict.get(k, None) is not None:
                m = common_utils.mask_points_by_range(data_dict[k], self.point_cloud_range)
                data_dict[k] = data_dict[k][m]

        # radar 유지
        if data_dict.get('radar_points', None) is not None:
            m = common_utils.mask_points_by_range(data_dict['radar_points'], self.point_cloud_range)
            data_dict['radar_points'] = data_dict['radar_points'][m]

        # GT 박스 클립 (기존과 동일)
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range,
                min_num_corners=config.get('min_num_corners', 1),
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            for k in self._get_teacher_keys(data_dict):
                if data_dict.get(k, None) is not None and len(data_dict[k]) > 0:
                    idx = np.random.permutation(len(data_dict[k]))
                    data_dict[k] = data_dict[k][idx]
            if data_dict.get('radar_points', None) is not None and len(data_dict['radar_points']) > 0:
                r_idx = np.random.permutation(len(data_dict['radar_points']))
                data_dict['radar_points'] = data_dict['radar_points'][r_idx]
        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        return data_dict

    def double_flip(self, points):
        pts_y = points.copy(); pts_y[:, 1] = -pts_y[:, 1]
        pts_x = points.copy(); pts_x[:, 0] = -pts_x[:, 0]
        pts_xy = points.copy(); pts_xy[:, 0] = -pts_xy[:, 0]; pts_xy[:, 1] = -pts_xy[:, 1]
        return pts_y, pts_x, pts_xy

    def _voxelize_once(self, points, drop_xyz, config):
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output
        if not drop_xyz:
            return voxels, coordinates, num_points
        return voxels[..., 3:], coordinates, num_points  # xyz 제거 옵션

    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        drop_xyz = not data_dict.get('use_lead_xyz', True)

        # -------- teacher 세트 각각 보xel화 (동적 teacher keys) --------
        for k in self._get_teacher_keys(data_dict):
            if data_dict.get(k, None) is None:
                continue
            pts = data_dict[k]
            # Extract suffix: 'teacher_points_s1' -> 's1', 'teacher_points_s10' -> 's10'
            suffix = k.split('_s')[-1] if '_s' in k else k[-2:]
            suffix = 's' + suffix  # Add 's' prefix back
            
            if pts is None or pts.size == 0:
                # 빈 케이스도 키는 만들어 둠
                data_dict[f'teacher_voxels_{suffix}'] = np.zeros((0, config.MAX_POINTS_PER_VOXEL, self.num_point_features), dtype=pts.dtype)
                data_dict[f'teacher_voxel_coords_{suffix}'] = np.zeros((0, 3), dtype=np.int32)
                data_dict[f'teacher_voxel_num_points_{suffix}'] = np.zeros((0,), dtype=np.int32)
                continue

            if config.get('DOUBLE_FLIP', False):
                vox_list, coord_list, num_list = [], [], []
                base_sets = [pts] + list(self.double_flip(pts))
                for p in base_sets:
                    v, c, n = self._voxelize_once(p, drop_xyz, config)
                    vox_list.append(v); coord_list.append(c); num_list.append(n)
                data_dict[f'teacher_voxels_{suffix}'] = vox_list
                data_dict[f'teacher_voxel_coords_{suffix}'] = coord_list
                data_dict[f'teacher_voxel_num_points_{suffix}'] = num_list
            else:
                v, c, n = self._voxelize_once(pts, drop_xyz, config)
                data_dict[f'teacher_voxels_{suffix}'] = v
                data_dict[f'teacher_voxel_coords_{suffix}'] = c
                data_dict[f'teacher_voxel_num_points_{suffix}'] = n

        # -------- radar 보xel화(그대로) --------
        if data_dict.get('radar_points', None) is not None:
            self.radar_voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.radar_num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )
            r_pts = data_dict['radar_points']
            r_voxels, r_coords, r_num_pts = self.radar_voxel_generator.generate(r_pts)
            if not data_dict.get('use_lead_xyz', True):
                r_voxels = r_voxels[..., 3:]
            if config.get('DOUBLE_FLIP', False):
                rv_list, rc_list, rn_list = [r_voxels], [r_coords], [r_num_pts]
                ry, rx, rxy = self.double_flip(r_pts)
                for rp in (ry, rx, rxy):
                    v, c, n = self.radar_voxel_generator.generate(rp)
                    if not data_dict.get('use_lead_xyz', True):
                        v = v[..., 3:]
                    rv_list.append(v); rc_list.append(c); rn_list.append(n)
                data_dict['radar_voxels'] = rv_list
                data_dict['radar_voxel_coords'] = rc_list
                data_dict['radar_voxel_num_points'] = rn_list
            else:
                data_dict['radar_voxels'] = r_voxels
                data_dict['radar_voxel_coords'] = r_coords
                data_dict['radar_voxel_num_points'] = r_num_pts

        return data_dict

    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:
            return data_dict

        # teacher 각각에 샘플링 적용
        def _sample_one(arr):
            if arr is None or len(arr) == 0:
                return arr
            if num_points < len(arr):
                depth = np.linalg.norm(arr[:, :3], axis=1)
                far = np.where(depth >= 40.0)[0]
                near = np.where(depth < 40.0)[0]
                if num_points > len(far):
                    near_choice = np.random.choice(near, num_points - len(far), replace=False) if len(near) > 0 else np.array([], dtype=np.int32)
                    choice = np.concatenate((near_choice, far), 0) if len(far) > 0 else near_choice
                else:
                    choice = np.random.choice(np.arange(len(arr), dtype=np.int32), num_points, replace=False)
            else:
                choice = np.arange(len(arr), dtype=np.int32)
                if num_points > len(arr):
                    extra = np.random.choice(choice, num_points - len(arr), replace=False)
                    choice = np.concatenate((choice, extra), 0)
            np.random.shuffle(choice)
            return arr[choice]

        for k in self.TEACHER_KEYS:
            if k in data_dict:
                data_dict[k] = _sample_one(data_dict[k])

        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    # 나머지 이미지 관련 함수, downsample 등은 그대로
    def forward(self, data_dict):
        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)
        return data_dict
