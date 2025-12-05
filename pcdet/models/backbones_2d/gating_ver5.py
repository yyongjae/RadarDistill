import torch
import torch.nn as nn
import torch.nn.functional as F

class SweepGaterV5(nn.Module):
    """
    [Gater Ver5 - Object-wise RKG]
    GT Box 영역별로 Relative Knowledge Gain을 계산하여,
    객체마다 다른 최적의 Sweep을 할당하는 모듈.
    """
    def __init__(self, num_sweeps=3, C=256, temp=0.1, **kwargs):
        super().__init__()
        self.num_sweeps = num_sweeps
        self.temp = temp
        # 별도의 학습 파라미터(MLP) 없음. 
        # 오직 RKG 규칙에 따라 가중치를 생성하여 Distillation을 가이드함.

    def get_patch_similarity(self, feat1_patch, feat2_patch):
        """
        Box 영역(Patch) 내의 Feature 벡터 유사도 계산
        feat_patch: [C, H_box, W_box] or [N, C, H_box, W_box]
        """
        # 벡터화 (Channel, H, W -> Vector) - contiguous for stability
        if feat1_patch.dim() == 3: # [C, H, W]
            f1 = feat1_patch.contiguous().view(-1)
            # feat2가 [N, C, H, W]인 경우 처리
            if feat2_patch.dim() == 4:
                N = feat2_patch.shape[0]
                f2 = feat2_patch.contiguous().view(N, -1)
                f1 = f1.unsqueeze(0).expand(N, -1)
                return F.cosine_similarity(f1, f2, dim=1) # [N]
            else:
                f2 = feat2_patch.contiguous().view(-1)
                return F.cosine_similarity(f1, f2, dim=0) # Scalar
        return 0.0

    def forward(self, S_curr, S_base, T, gt_boxes_info):
        """
        Args:
            S_curr: [B, C, H, W]
            S_base: [B, C, H, W]
            T: [B, N, C, H, W]
            gt_boxes_info: (gt_boxes, grid_config) 정보를 담은 딕셔너리 or 튜플
                           gt_boxes: [B, M, 8]
        Returns:
            weights: [B, N, 1, H, W] (Object별로 다른 가중치가 칠해진 맵)
            stats: 로깅용 통계
        """
        B, N, C, H, W = T.shape
        gt_boxes = gt_boxes_info['boxes']
        pc_range = gt_boxes_info['pc_range']
        
        # 1. 빈 Weight Map 생성 (기본값: 배경은 학습 안하므로 0 처리하거나, s1으로 채움)
        # Distill Loss에서 어차피 Masking 되므로 0으로 초기화해도 무방함.
        # [B, N, H, W]
        weight_map = torch.zeros((B, N, H, W), device=S_curr.device, dtype=S_curr.dtype)
        
        # 통계용 - 모든 teacher 추적
        gain_lists = [[] for _ in range(N)]  # N개 teacher별 gain 리스트
        prob_lists = [[] for _ in range(N)]  # N개 teacher별 prob 리스트

        stride_x = (pc_range[3] - pc_range[0]) / W
        stride_y = (pc_range[4] - pc_range[1]) / H

        # 2. 배치 내 각 샘플(프레임) 순회
        for b in range(B):
            # 유효한 박스 추출
            valid_mask = (gt_boxes[b, :, 3] > 0)
            boxes = gt_boxes[b][valid_mask]

            # Fallback: 박스가 하나도 없으면 균등 가중치 사용 (s1 편향 방지)
            if len(boxes) == 0:
                weight_map[b, :, :, :] = 1.0 / N
                continue

            # 좌표 변환 (World -> Grid)
            cx = (boxes[:, 0] - pc_range[0]) / stride_x
            cy = (boxes[:, 1] - pc_range[1]) / stride_y
            w  = boxes[:, 3] / stride_x
            l  = boxes[:, 4] / stride_y

            # AABB 영역 계산
            x1 = (cx - w/2).floor().long().clamp(0, W-1)
            x2 = (cx + w/2).ceil().long().clamp(0, W-1)
            y1 = (cy - l/2).floor().long().clamp(0, H-1)
            y2 = (cy + l/2).ceil().long().clamp(0, H-1)

            # 3. 객체별 Gain 계산 및 Weight 할당
            for k in range(len(boxes)):
                # 영역이 유효하지 않으면 패스
                if x2[k] <= x1[k] or y2[k] <= y1[k]:
                    continue
                
                # Patch 추출 (Slicing)
                # s_curr_patch: [C, H_box, W_box]
                s_curr_patch = S_curr[b, :, y1[k]:y2[k]+1, x1[k]:x2[k]+1]
                s_base_patch = S_base[b, :, y1[k]:y2[k]+1, x1[k]:x2[k]+1]
                # t_patch: [N, C, H_box, W_box]
                t_patch = T[b, :, :, y1[k]:y2[k]+1, x1[k]:x2[k]+1]

                with torch.no_grad():
                    # A. Similarity 계산 [N]
                    sim_base = self.get_patch_similarity(s_base_patch, t_patch)
                    
                    # Gain 계산 시에는 S_curr 쪽 Gradient 끊기 (Target 생성 목적)
                    # (단, Distill Loss에서는 Gradient가 흘러야 하므로 여기서만 detach)
                    s_curr_patch_no_grad = s_curr_patch.detach()
                    sim_curr = self.get_patch_similarity(s_curr_patch_no_grad, t_patch)
                    
                    # B. Gain [N]
                    gain = sim_curr - sim_base
                    
                    # C. Softmax (Local Decision)
                    probs = F.softmax(gain / self.temp, dim=0) # [N]

                    # 통계 수집 - 모든 teacher
                    for i in range(N):
                        gain_lists[i].append(gain[i].item())
                        prob_lists[i].append(probs[i].item())

                # 4. Weight Map에 칠하기 (Broadcasting)
                # probs: [N] -> [N, 1, 1]
                # target region: [N, H_box, W_box]
                weight_map[b, :, y1[k]:y2[k]+1, x1[k]:x2[k]+1] = probs.view(N, 1, 1)

        # 5. 차원 확장 및 리턴
        # weights: [B, N, 1, H, W]
        weights = weight_map.unsqueeze(2)
        
        # 통계 정리 - 모든 teacher의 평균 gain/prob
        stats = {}
        for i in range(N):
            avg_gain = sum(gain_lists[i]) / len(gain_lists[i]) if gain_lists[i] else 0.0
            avg_prob = sum(prob_lists[i]) / len(prob_lists[i]) if prob_lists[i] else 0.0
            stats[f'gain_t{i}'] = avg_gain
            stats[f'prob_t{i}'] = avg_prob
        
        # Router Loss는 별도로 없음 (Gain이 곧 Weight이므로)
        # 호환성을 위해 0.0 리턴
        loss_router = torch.tensor(0.0, device=S_curr.device)
        
        return weights, loss_router, stats