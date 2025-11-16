import pickle

import os
import copy
import numpy as np
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils


class DataBaseSampler_Distill(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        self.min_points_per_class = {}
        self.total_removals = {class_name: 0 for class_name in self.class_names}
        self._ts_ana_log_count = 0

        # Add counters for t=0.0 analysis
        self._stats = {'total': 0, 'no_t0': 0}

        for class_name in self.class_names:
            self.db_infos[class_name] = []
            
        self.use_shared_memory = sampler_cfg.get('USE_SHARED_MEMORY', False)
            
        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path / db_info_path
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
            if name in db_infos.keys():
                self.min_points_per_class[name] = min_num

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

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        radar_points = data_dict['radar_points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        obj_radar_points_list = []
        skipped_count = 0  # Track how many objects we skip due to filtering issues
        
        if self.use_shared_memory:
            gt_database_data = SharedArray.attach(f"shm://{self.gt_database_data_key}")
            gt_database_data.setflags(write=0)
        else:
            gt_database_data = None 

        for idx, info in enumerate(total_valid_sampled_dict):
            if self.use_shared_memory:
                start_offset, end_offset = info['global_data_offset']
                obj_points = copy.deepcopy(gt_database_data[start_offset:end_offset])
            else:
                file_path = self.root_path / info['path']
                obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                    [-1, self.sampler_cfg.NUM_POINT_FEATURES])
                # breakpoint()
                radar_file_path = self.root_path / info['radar_path']
                obj_radar_points = np.fromfile(str(radar_file_path), dtype=np.float32).reshape(
                    [-1, 6])
            

            # sweep 몇 개 가져갈건지 - ENHANCED FILTERING
            keep_k = self.sampler_cfg.get('SAMPLE_N_SWEEPS', None)
            filtering_success = False
            
            if keep_k is not None and obj_points.shape[1] >= 5:
                try:
                    k = int(keep_k)
                    ts_idx = self.sampler_cfg.get('TIMESTAMP_IDX', -1)  # default: last column
                    decimals = 2

                    # Extract timestamps
                    try:
                        times = obj_points[:, ts_idx]
                    except Exception as e:
                        times = obj_points[:, -1]
                    times_r = np.round(times, decimals)
                    unique_times = np.sort(np.unique(times_r))

                    if unique_times.size > 0:
                        # Apply timestamp threshold filtering
                        thr = unique_times[min(k - 1, unique_times.size - 1)]
                        time_mask = times_r <= thr
                        obj_points = obj_points[time_mask]

                        # CRITICAL: Verify filtering worked correctly
                        times_after = np.round(obj_points[:, -1], decimals)
                        unique_times_after = np.unique(times_after)

                        if len(unique_times_after) <= k:
                            filtering_success = True
                        # else:
                        #     # Log failure case for debugging
                        #     if self.logger and getattr(self, '_ts_ana_log_count', 0) < 50:
                        #         uniq_before, counts_before = np.unique(times_r, return_counts=True)
                        #         hist_before = [(float(t), int(c)) for t, c in zip(uniq_before, counts_before)]
                        #         uniq_after, counts_after = np.unique(times_after, return_counts=True)
                        #         hist_after = [(float(t), int(c)) for t, c in zip(uniq_after, counts_after)]
                        #         self.logger.warning(
                        #             f"[FILTER FAIL] Class={info['name']}, k={k}, thr={thr:.2f}\n"
                        #             f"  - Before: unique={len(hist_before)}, hist={hist_before}\n"
                        #             f"  - After: unique={len(hist_after)}, hist={hist_after}"
                        #         )
                    else:
                        filtering_success = True # No timestamps to filter, so it's a success

                    # === Analysis for t=0.0 existence ===
                    self._stats['total'] += 1
                    if 0.0 not in unique_times:
                        self._stats['no_t0'] += 1

                    # if self.logger and self._stats['total'] > 0 and self._stats['total'] % 100 == 0:
                    #     ratio = (self._stats['no_t0'] / self._stats['total']) * 100
                    #     self.logger.info(
                    #         f"[T0_STATS] Objects w/o t=0.0: {self._stats['no_t0']} / {self._stats['total']} ({ratio:.2f}%)"
                    #     )

                    # Log filtering info for debugging
                    # if self.logger and getattr(self, '_ts_ana_log_count', 0) < 50:
                    #     uniq_before, counts_before = np.unique(times_r, return_counts=True)
                    #     hist_before = [(float(t), int(c)) for t, c in zip(uniq_before, counts_before)]

                    #     times_after_final = np.round(obj_points[:, ts_idx], decimals)
                    #     uniq_after, counts_after = np.unique(times_after_final, return_counts=True)
                    #     hist_after = [(float(t), int(c)) for t, c in zip(uniq_after, counts_after)]

                    #     log_func = self.logger.info if filtering_success else self.logger.warning
                    #     log_func(
                    #         f"[FILTER] Success={filtering_success}, Class={info['name']}, k={k}, thr={thr:.2f}\n"
                    #         f"  - Before: unique={len(hist_before)}, hist={hist_before}\n"
                    #         f"  - After: unique={len(hist_after)}, hist={hist_after}"
                    #     )

                except Exception as e:
                    # Any exception during filtering = skip this object
                    if self.logger:
                        self.logger.error(f"Error during filtering object {info['name']}: {e}")
                    skipped_count += 1
                    continue  # Skip this problematic object
            
            # If SAMPLE_N_SWEEPS is set but filtering didn't succeed, skip the object
            if keep_k is not None and not filtering_success:
                skipped_count += 1
                continue

            # Optional: filter by explicit time range if provided (fallback behavior)
            if self.sampler_cfg.get('FILTER_OBJ_POINTS_BY_TIMESTAMP', False) and obj_points.shape[1] >= 5:
                time_range = self.sampler_cfg.get('TIME_RANGE', None)
                if isinstance(time_range, (list, tuple)) and len(time_range) == 2:
                    min_time, max_time = float(time_range[0]), float(time_range[1])
                    times = obj_points[:, -1]
                    time_mask = (times <= max_time + 1e-6) & (times >= min_time - 1e-6)
                    obj_points = obj_points[time_mask]

            obj_points[:, :3] += info['box3d_lidar'][:3]
            obj_radar_points[:, :3] += info['box3d_lidar'][:3]


            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]
                obj_radar_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)
            obj_radar_points_list.append(obj_radar_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        obj_radar_points = np.concatenate(obj_radar_points_list, axis=0)
        
        # TIMESTAMP ANALYSIS (per-sample, after concatenation)
        try:
            try:
                ts_idx = self.sampler_cfg.get('TIMESTAMP_IDX', -1)
                times_all = obj_points[:, ts_idx]
            except Exception:
                times_all = obj_points[:, -1]
            times_r = np.round(times_all, 2)
            total_ts_count = int(times_r.size)
            uniq_ts, counts = np.unique(times_r, return_counts=True)
            unique_ts_count = int(uniq_ts.size)
            # if self.logger is not None and getattr(self, '_ts_ana_log_count', 0) < 50:
            #     self.logger.info(f'TS ANALYSIS: total={total_ts_count}, unique={unique_ts_count}')
            #     self.logger.info(f'TS ANALYSIS: unique_values={uniq_ts.tolist()}')
            #     hist = [(float(t), int(c)) for t, c in zip(uniq_ts, counts)]
            #     self.logger.info(f'TS ANALYSIS: histogram={hist}')
            #     self._ts_ana_log_count = getattr(self, "_ts_ana_log_count", 0) + 1
        except Exception:
            pass
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        radar_points = box_utils.remove_points_in_boxes3d(radar_points, large_sampled_gt_boxes)
        radar_points = np.concatenate([obj_radar_points, radar_points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        data_dict['radar_points'] = radar_points
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