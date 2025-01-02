## lidar 모델
 PORT=1234 OMP_NUM_THREADS=6  ./dist_train.sh 4 --cfg_file cfgs/distillation_models/cbgs_voxel0075_res3d_centerpoint.yaml --ckpt_save_interval 5 --extra_tag interval7_cbgs_gt_aug

## pointpillars weight 만들기
 PORT=1233 OMP_NUM_THREADS=6 ./dist_train.sh 4 --cfg_file cfgs/distillation_models/cbgs_voxel0075_res3d_centerpoint_radar.yaml --ckpt_save_interval 5 --extra_tag interval7_cbgs_gt_aug
