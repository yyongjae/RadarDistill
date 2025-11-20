import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils


class DataBaseSampler_Distill_Multi_Sweep_Teacher(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        
        # Debug statistics
        self._debug_stats = {
            'total_objects': 0,
            'total_scenes': 0,
            'teacher_filtering': {}  # Will store per-teacher stats
        }
        self._debug_log_interval = 50  # Log every N scenes

        for class_name in class_names:
            self.db_infos[class_name] = []
            
        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
        
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]
        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)
        
        self.gt_database_data_key = self.load_db_to_shared_memory() if self.use_shared_memory else None

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)

        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __del__(self):
        if self.use_shared_memory:
            self.logger.info('Deleting GT database from shared memory')
            cur_rank, num_gpus = common_utils.get_dist_info()
            sa_key = self.sampler_cfg.DB_DATA_PATH[0]
            if cur_rank % num_gpus == 0 and os.path.exists(f"/dev/shm/{sa_key}"):
                SharedArray.delete(f"shm://{sa_key}")

            if num_gpus > 1:
                dist.barrier()
            self.logger.info('GT database has been removed from shared memory')

    def load_db_to_shared_memory(self):
        self.logger.info('Loading GT database to shared memory')
        cur_rank, world_size, num_gpus = common_utils.get_dist_info(return_gpu_per_machine=True)

        assert self.sampler_cfg.DB_DATA_PATH.__len__() == 1, 'Current only support single DB_DATA'
        db_data_path = self.root_path.resolve() / self.sampler_cfg.DB_DATA_PATH[0]
        sa_key = self.sampler_cfg.DB_DATA_PATH[0]

        if cur_rank % num_gpus == 0 and not os.path.exists(f"/dev/shm/{sa_key}"):
            gt_database_data = np.load(db_data_path)
            common_utils.sa_create(f"shm://{sa_key}", gt_database_data)
            
        if num_gpus > 1:
            dist.barrier()
        self.logger.info('GT database has been saved to shared memory')
        return sa_key

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num and info['num_radar_points_in_gt'] >= 1:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def _keep_first_k_sweeps(self, arr, k, ts_idx=4):
        """
        timestamp(=time_lag) 기준으로 가까운 순 'K 스윕'만 남긴다.
        arr: (N, F) with arr[:, ts_idx] = time_lag
        """
        if arr.shape[1] <= ts_idx:
            return arr  # timestamp 없으면 그대로(최소한의 fallback)
        dt = np.round(arr[:, ts_idx], 2)
        uniq = np.sort(np.unique(dt))
        thr = uniq[min(k - 1, len(uniq) - 1)]
        return arr[dt <= thr]

    def _parse_k_from_key(self, key):
        # 'teacher_points_s1' / '..._s5' / '..._s8' / '..._s10' -> 1/5/8/10
        return int(key.split('_s')[-1])

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]

        # Dynamically collect teacher point keys (e.g., teacher_points_s1, teacher_points_s3, etc.)
        teacher_keys = [k for k in data_dict.keys() if k.startswith('teacher_points_s')]
        teacher_pts = {k: data_dict[k] for k in teacher_keys}

        radar_points = data_dict['radar_points']

        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib'); data_dict.pop('road_plane')

        # Initialize lists for each teacher separately (efficient: filter only once per teacher)
        teacher_obj_points = {k: [] for k in teacher_keys}
        obj_radar_points_list = []
        ts_idx = self.sampler_cfg.get('TIMESTAMP_IDX', 4)
        
        # Debug: Initialize per-teacher stats for this scene
        scene_stats = {}
        for k in teacher_keys:
            if k not in self._debug_stats['teacher_filtering']:
                self._debug_stats['teacher_filtering'][k] = {
                    'total_points_before': 0,
                    'total_points_after': 0,
                    'total_objects': 0,
                    'unique_sweeps_before': [],
                    'unique_sweeps_after': []
                }
            scene_stats[k] = {'before': 0, 'after': 0, 'sweeps_before': set(), 'sweeps_after': set()}
        
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None

        for info in total_valid_sampled_dict:
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
                obj_radar_points = None
            else:
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(-1, self.sampler_cfg.NUM_POINT_FEATURES)
                radar_file_path = self.root_path / info['radar_path']
                obj_radar_points = np.fromfile(str(radar_file_path), dtype=np.float32).reshape(-1, 6)

            # Position adjustment first (before filtering)
            obj_points[:, :3] += info['box3d_lidar'][:3]
            if obj_radar_points is not None:
                obj_radar_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                dv = mv_height
                obj_points[:, 2] -= dv
                if obj_radar_points is not None:
                    obj_radar_points[:, 2] -= dv

            # Filter once per teacher (efficient: no redundant filtering)
            for k in teacher_keys:
                K = self._parse_k_from_key(k)  # Extract sweep count: s1→1, s5→5, s8→8, s10→10
                
                # Debug: Track before filtering
                pts_before = len(obj_points)
                if obj_points.shape[1] > ts_idx:
                    ts_before = np.unique(np.round(obj_points[:, ts_idx], 2))
                    scene_stats[k]['sweeps_before'].update(ts_before)
                scene_stats[k]['before'] += pts_before
                
                # Perform filtering
                obj_pts_for_k = self._keep_first_k_sweeps(obj_points, k=K, ts_idx=ts_idx)
                
                # Debug: Track after filtering
                pts_after = len(obj_pts_for_k)
                if obj_pts_for_k.shape[1] > ts_idx:
                    ts_after = np.unique(np.round(obj_pts_for_k[:, ts_idx], 2))
                    scene_stats[k]['sweeps_after'].update(ts_after)
                scene_stats[k]['after'] += pts_after
                
                # Accumulate global stats
                self._debug_stats['teacher_filtering'][k]['total_points_before'] += pts_before
                self._debug_stats['teacher_filtering'][k]['total_points_after'] += pts_after
                self._debug_stats['teacher_filtering'][k]['total_objects'] += 1
                
                teacher_obj_points[k].append(obj_pts_for_k)

            self._debug_stats['total_objects'] += 1

            if obj_radar_points is not None:
                obj_radar_points_list.append(obj_radar_points)

        # Concatenate per teacher
        obj_radar_points = np.concatenate(obj_radar_points_list, axis=0) if obj_radar_points_list else np.zeros((0, 6), dtype=np.float32)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        # Remove collision boxes
        large_boxes = box_utils.enlarge_box3d(sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH)

        # Merge augmentation objects with scene points for each teacher
        for k in teacher_keys:
            pts_scene = box_utils.remove_points_in_boxes3d(teacher_pts[k], large_boxes)
            obj_pts = np.concatenate(teacher_obj_points[k], axis=0) if teacher_obj_points[k] else np.zeros((0, 5), dtype=np.float32)
            data_dict[k] = np.concatenate([obj_pts, pts_scene], axis=0)

        # 레이더 포인트
        radar_points = box_utils.remove_points_in_boxes3d(radar_points, large_boxes)
        data_dict['radar_points'] = np.concatenate([obj_radar_points, radar_points], axis=0)

        # GT 갱신
        data_dict['gt_names'] = np.concatenate([gt_names, sampled_gt_names], axis=0)
        data_dict['gt_boxes'] = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        
        # Debug: Log statistics periodically
        self._debug_stats['total_scenes'] += 1
        if self.logger and self._debug_stats['total_scenes'] % self._debug_log_interval == 0:
            self.logger.info(f"\n{'='*80}")
            self.logger.info(f"[MULTI_SWEEP_FILTER_DEBUG] Scene #{self._debug_stats['total_scenes']} Statistics")
            self.logger.info(f"{'='*80}")
            self.logger.info(f"Total augmentation objects processed: {self._debug_stats['total_objects']}")
            self.logger.info(f"Total augmentation objects in this scene: {len(total_valid_sampled_dict)}")
            
            for k in sorted(teacher_keys):
                stats = self._debug_stats['teacher_filtering'][k]
                K = self._parse_k_from_key(k)
                pts_before = stats['total_points_before']
                pts_after = stats['total_points_after']
                retention_rate = (pts_after / pts_before * 100) if pts_before > 0 else 0
                
                sweeps_before = sorted(scene_stats[k]['sweeps_before'])
                sweeps_after = sorted(scene_stats[k]['sweeps_after'])
                
                self.logger.info(f"\n  [{k}] Target: {K} sweeps")
                self.logger.info(f"    - This scene: {scene_stats[k]['before']} pts → {scene_stats[k]['after']} pts")
                self.logger.info(f"    - This scene sweeps: {len(sweeps_before)} → {len(sweeps_after)}")
                self.logger.info(f"    - This scene timestamps before: {sweeps_before[:10]}{'...' if len(sweeps_before) > 10 else ''}")
                self.logger.info(f"    - This scene timestamps after:  {sweeps_after}")
                self.logger.info(f"    - Cumulative: {pts_before:,} pts → {pts_after:,} pts ({retention_rate:.1f}% retained)")
                self.logger.info(f"    - Cumulative objects: {stats['total_objects']}")
            
            self.logger.info(f"{'='*80}\n")
        
        return data_dict


    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []
        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict.pop('gt_boxes_mask')
        return data_dict