import numpy as np


class PointFeatureEncoder(object):
    def __init__(self, config, point_cloud_range=None,is_radar=False):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list
        self.point_cloud_range = point_cloud_range
        self.is_radar = is_radar

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
            data_dict:
                points: (N, 3 + C_out),
                use_lead_xyz: whether to use xyz as point-wise features
                ...
        """
        data_dict['points'], use_lead_xyz = getattr(self, self.point_encoding_config.encoding_type)(
            data_dict['points']
        )
        data_dict['use_lead_xyz'] = use_lead_xyz
        
       
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            max_sweeps = self.point_encoding_config.max_sweeps
            idx = self.src_feature_list.index('timestamp')
            dt = np.round(data_dict['points'][:, idx], 2)
            max_dt = sorted(np.unique(dt))[min(len(np.unique(dt))-1, max_sweeps-1)]
            data_dict['points'] = data_dict['points'][dt <= max_dt]
        
        return data_dict

    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
    def radar_absolute_coordinates_encoding(self, points=None):
        if points is None:
            num_output_features = len(self.used_feature_list)
            return num_output_features

        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        
        return point_features, True
    
class PointFeatureEncoder_Distill_Multi_Sweep_Teacher(object):
    def __init__(self, config, point_cloud_range=None):
        super().__init__()
        self.point_encoding_config = config
        assert list(self.point_encoding_config.src_feature_list[0:3]) == ['x', 'y', 'z']
        self.used_feature_list = self.point_encoding_config.used_feature_list
        self.src_feature_list = self.point_encoding_config.src_feature_list

        # radar 전용 설정
        self.radar_used_feature_list = self.point_encoding_config.radar_used_feature_list
        self.radar_src_feature_list = self.point_encoding_config.radar_src_feature_list

        self.point_cloud_range = point_cloud_range

    @property
    def num_point_features(self):
        return getattr(self, self.point_encoding_config.encoding_type)(points=None)

    @property
    def radar_num_point_features(self):
        return getattr(self, self.point_encoding_config.radar_encoding_type)(points=None)

    def forward(self, data_dict):
        """
        points 키는 사용하지 않음.
        - teacher_points_sX (dynamic): 동일 인코딩 적용 (필요시 timestamp로 sweep filter)
        - radar_points: radar 인코딩 별도 적용
        """
        # 1) (옵션) teacher sweep 필터링: timestamp 기준
        teacher_keys = [k for k in data_dict.keys() if k.startswith('teacher_points_s')]
        if self.point_encoding_config.get('filter_sweeps', False) and 'timestamp' in self.src_feature_list:
            ts_idx = self.src_feature_list.index('timestamp')
            max_sweeps = self.point_encoding_config.max_sweeps
            for k in teacher_keys:
                if k not in data_dict:
                    continue
                pts = data_dict[k]
                if pts is None or pts.size == 0:
                    continue
                dt = np.round(pts[:, ts_idx], 2)
                uniq = np.unique(dt)
                max_dt = sorted(uniq)[min(len(uniq) - 1, max_sweeps - 1)]
                data_dict[k] = pts[dt <= max_dt].astype(np.float32)

        # 2) teacher 인코딩
        use_lead_xyz_teacher = True
        for k in teacher_keys:
            if k not in data_dict:
                continue
            pts = data_dict[k]
            if pts is None or pts.size == 0:
                data_dict[k] = pts
                continue
            enc, use_lead_xyz_teacher = getattr(self, self.point_encoding_config.encoding_type)(pts)
            data_dict[k] = enc

        # 호환성: 일부 다운스트림에서 참조 가능하도록 동일 flag도 넣어줌
        data_dict['use_lead_xyz_teacher'] = use_lead_xyz_teacher
        data_dict['use_lead_xyz'] = use_lead_xyz_teacher

        # 3) radar 인코딩
        if 'radar_points' in data_dict and data_dict['radar_points'] is not None and data_dict['radar_points'].size > 0:
            data_dict['radar_points'] = getattr(
                self, self.point_encoding_config.radar_encoding_type
            )(data_dict['radar_points'])

        return data_dict

    # ---------- encoding funcs ----------
    def absolute_coordinates_encoding(self, points=None):
        if points is None:
            return len(self.used_feature_list)
        point_feature_list = [points[:, 0:3]]
        for x in self.used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features, True

    def radar_absolute_coordinates_encoding(self, points=None):
        if points is None:
            return len(self.radar_used_feature_list)
        point_feature_list = [points[:, 0:3]]
        for x in self.radar_used_feature_list:
            if x in ['x', 'y', 'z']:
                continue
            idx = self.radar_src_feature_list.index(x)
            point_feature_list.append(points[:, idx:idx+1])
        point_features = np.concatenate(point_feature_list, axis=1)
        return point_features
