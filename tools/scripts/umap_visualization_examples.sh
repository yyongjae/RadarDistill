#!/bin/bash

# BEV Feature UMAP 시각화 예제 스크립트
# 이 스크립트는 evaluation 시 BEV feature의 UMAP 시각화를 생성하는 방법을 보여줍니다.

# 설정
NGPUS=1
CFG_FILE="cfgs/radar_distill/radar_distill_train_multi_sweep.yaml"
CKPT="/path/to/your/checkpoint.pth"  # 실제 checkpoint 경로로 변경하세요
EXTRA_TAG="umap_demo"

# 기본 UMAP 시각화 (2D)
echo "=== 예제 1: 기본 2D UMAP 시각화 ==="
bash scripts/dist_test.sh ${NGPUS} \
  --cfg_file ${CFG_FILE} \
  --ckpt ${CKPT} \
  --extra_tag ${EXTRA_TAG}_2d \
  --save_umap_visualization \
  --features_to_analyze low_radar_bev,high_radar_bev \
  --model_type student

# 3D UMAP 시각화
echo "=== 예제 2: 3D UMAP 시각화 ==="
bash scripts/dist_test.sh ${NGPUS} \
  --cfg_file ${CFG_FILE} \
  --ckpt ${CKPT} \
  --extra_tag ${EXTRA_TAG}_3d \
  --save_umap_visualization \
  --features_to_analyze high_radar_bev \
  --model_type student \
  --umap_n_components 3

# 여러 feature 동시 분석
echo "=== 예제 3: 여러 Feature 동시 분석 ==="
bash scripts/dist_test.sh ${NGPUS} \
  --cfg_file ${CFG_FILE} \
  --ckpt ${CKPT} \
  --extra_tag ${EXTRA_TAG}_multi \
  --save_umap_visualization \
  --features_to_analyze low_radar_bev,high_radar_bev \
  --model_type student

# UMAP 파라미터 커스터마이징
echo "=== 예제 4: UMAP 파라미터 커스터마이징 ==="
bash scripts/dist_test.sh ${NGPUS} \
  --cfg_file ${CFG_FILE} \
  --ckpt ${CKPT} \
  --extra_tag ${EXTRA_TAG}_custom \
  --save_umap_visualization \
  --features_to_analyze high_radar_bev \
  --model_type student \
  --umap_n_neighbors 30 \
  --umap_min_dist 0.5 \
  --umap_max_samples 5000

# Similarity map과 함께 사용
echo "=== 예제 5: UMAP + Similarity Map ==="
bash scripts/dist_test.sh ${NGPUS} \
  --cfg_file ${CFG_FILE} \
  --ckpt ${CKPT} \
  --extra_tag ${EXTRA_TAG}_full \
  --save_umap_visualization \
  --save_class_similarity \
  --save_scene_instance_similarity \
  --features_to_analyze high_radar_bev \
  --model_type student \
  --max_scene_instance_plots 100

echo "=== 완료! ==="
echo "결과는 output/${EXTRA_TAG}_*/eval/umap/ 디렉토리에서 확인하세요."
