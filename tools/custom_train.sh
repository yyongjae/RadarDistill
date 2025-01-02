
PORT=1234 OMP_NUM_THREADS=6 bash scripts/dist_train.sh 4 --cfg_file cfgs/nuscenes_models/pillarnet_radar_CMA.yaml --extra_tag cma_cbgs_w_gt_aug

