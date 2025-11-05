import torch
import torch.nn as nn
import torch.nn.functional as F

class SweepGater(nn.Module):
    """
    스윕 선택 게이터:
      - (옵션) 1x1 conv 어댑터로 스윕별 특성 정렬
      - EMA(μ, σ) 정규화 후 학생-교사 차이로 proxy 손실 산출
      - 점수 = 개선량 우위 + 난이도 밴드패스 (+ UCB 탐험)
      - soft/hard 방식으로 α 계산, 마스크 지원
    """
    def __init__(
        self,
        C: int,
        num_sweeps: int = 3,
        temp: float = 0.7,
        use_adapter: bool = True,
        ema_momentum: float = 0.99,
        alpha: float = 1.0,             # advantage 가중
        beta: float  = 0.5,             # band-pass 가중
        gamma: float = 0.0,             # UCB 가중(탐험)
        band_L: float = 0.05,           # 난이도 하한(프록시 스케일에 맞게 튠)
        band_H: float = 0.20            # 난이도 상한
    ):
        super().__init__()
        self.temp = temp
        self.use_adapter = use_adapter
        self.ema_momentum = ema_momentum

        # scoring 하이퍼
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.band_L = band_L
        self.band_H = band_H

        # (옵션) 스윕별 1x1 어댑터
        self.adapters = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(num_sweeps)]) if use_adapter else None

        # 통계/정규화용 버퍼
        # 정규화 파라미터: [1, S, C, 1, 1]
        self.register_buffer('ema_mu',    torch.zeros(1, num_sweeps, C, 1, 1), persistent=False)
        self.register_buffer('ema_sigma', torch.ones (1, num_sweeps, C, 1, 1), persistent=False)

        # 프록시(스윕 스칼라) EMA: [1, S, 1, 1, 1]
        self.register_buffer('ema_proxy', torch.zeros(1, num_sweeps, 1, 1, 1), persistent=False)

        # 선택 카운트 & 스텝: UCB용
        self.register_buffer('sel_count', torch.zeros(1, num_sweeps, 1, 1, 1), persistent=False)
        self.total_steps = 0

    # -------------------- 통계 업데이트 -------------------- #
    @torch.no_grad()
    def update_ema(self, T: torch.Tensor, mask: torch.Tensor = None, momentum: float = None):
        """
        정규화용 μ, σ EMA 업데이트
        T:    [B, S, C, H, W]
        mask: [B, S, 1, H, W] or None (유효/가중)
        """
        assert T.dim() == 5, f"Expected [B,S,C,H,W], got {T.shape}"
        if momentum is None:
            momentum = self.ema_momentum

        if mask is None:
            mu  = T.mean(dim=(0, 3, 4), keepdim=True)  # [1,S,C,1,1]
            var = T.var (dim=(0, 3, 4), keepdim=True, unbiased=False)
        else:
            W     = mask.clamp(min=0.0)                  # [B,S,1,H,W]
            Wc    = W.expand_as(T)                       # [B,S,C,H,W]
            denom = Wc.sum(dim=(0, 3, 4), keepdim=True).clamp_min(1e-6)
            mu    = (T * Wc).sum(dim=(0, 3, 4), keepdim=True) / denom
            var   = ((T - mu) ** 2 * Wc).sum(dim=(0, 3, 4), keepdim=True) / denom

        sigma = var.clamp_min(0).sqrt().clamp_min(1e-6)

        # 스윕 수 변동 대응
        if self.ema_mu.shape[1] != mu.shape[1]:
            self.ema_mu    = mu.clone()
            self.ema_sigma = sigma.clone()
        else:
            self.ema_mu    = momentum * self.ema_mu    + (1 - momentum) * mu
            self.ema_sigma = momentum * self.ema_sigma + (1 - momentum) * sigma

    @torch.no_grad()
    def _update_proxy_ema(self, proxy_map: torch.Tensor, mask: torch.Tensor = None):
        """
        proxy_map: [B, S, 1, H, W]  (채널축 평균된 프록시 맵)
        mask     : [B, S, 1, H, W] or None
        ema_proxy 업데이트 (스윕 스칼라)
        """
        cur = self._scalar_per_sweep(proxy_map, mask)  # [1,S,1,1,1]
        self.ema_proxy = self.ema_momentum * self.ema_proxy + (1 - self.ema_momentum) * cur

    # -------------------- 내부 유틸 -------------------- #
    def _apply_adapters(self, S: torch.Tensor, T: torch.Tensor):
        """
        S: [B,C,H,W], T: [B,S,C,H,W] -> (S_rep, T_adapt)
        """
        B, S_cnt, C, H, W = T.shape
        if self.adapters is not None:
            assert len(self.adapters) == S_cnt, f"adapters:{len(self.adapters)} != sweeps:{S_cnt}"
            T_adapt  = torch.stack([self.adapters[i](T[:, i]) for i in range(S_cnt)], dim=1)  # [B,S,C,H,W]
            S_rep    = torch.stack([self.adapters[i](S)       for i in range(S_cnt)], dim=1)  # [B,S,C,H,W]
        else:
            T_adapt = T
            S_rep   = S.unsqueeze(1).expand(-1, S_cnt, -1, -1, -1)
        return S_rep, T_adapt

    def _adapt_and_norm(self, S: torch.Tensor, T: torch.Tensor):
        """
        정렬 + 정규화
        return: Sz, Tz (둘 다 [B,S,C,H,W])
        """
        S_rep, T_adapt = self._apply_adapters(S, T)
        Tz = (T_adapt - self.ema_mu) / self.ema_sigma
        Sz = (S_rep   - self.ema_mu) / self.ema_sigma
        return Sz, Tz

    def _masked_channel_l2(self, delta: torch.Tensor, mask: torch.Tensor = None):
        """
        delta: [B,S,C,H,W]  ->  [B,S,1,H,W]  (채널축 L2, 마스크 평균)
        """
        if mask is None:
            return (delta ** 2).mean(dim=2, keepdim=True)
        # 채널에 동일 마스크(스칼라)라면 평균과 동일
        num = (delta ** 2).sum(dim=2, keepdim=True)          # [B,S,1,H,W]
        den = mask.sum(dim=2, keepdim=True).clamp_min(1e-6)  # [B,S,1,H,W]
        return num / den

    def _scalar_per_sweep(self, proxy_map: torch.Tensor, mask: torch.Tensor = None):
        """
        proxy_map: [B,S,1,H,W] -> [1,S,1,1,1]  (배치/공간 마스킹 평균)
        """
        if mask is None:
            return proxy_map.mean(dim=(0, 3, 4), keepdim=True)
        w   = mask.clamp(min=0.0)
        num = (proxy_map * w).sum(dim=(0, 3, 4), keepdim=True)
        den = w.sum(dim=(0, 3, 4), keepdim=True).clamp_min(1e-6)
        return num / den

    # -------------------- 메인 포워드 -------------------- #
    def forward(
        self,
        S: torch.Tensor,               # [B,C,H,W]   (student)
        T: torch.Tensor,               # [B,S,C,H,W] (teacher stack)
        mask: torch.Tensor = None,     # [B,S,1,H,W] (유효/가중) or None
        hard: bool = False
    ):
        # 1) 정렬+정규화, 프록시 맵 산출
        Sz, Tz = self._adapt_and_norm(S, T)                  # [B,S,C,H,W]
        delta  = Tz - Sz                                     # [B,S,C,H,W]
        proxy_map = self._masked_channel_l2(delta, mask)     # [B,S,1,H,W]

        # 2) 스윕 스칼라 손실: 현재(cur) & EMA(prev)
        cur  = self._scalar_per_sweep(proxy_map, mask)       # [1,S,1,1,1]
        prev = self.ema_proxy                                # [1,S,1,1,1]

        # 3) 개선량 우위(advantage)
        impr = (prev - cur).detach()
        adv  = impr - impr.mean(dim=1, keepdim=True)         # [1,S,1,1,1]

        # 4) 난이도 밴드패스 (너무 쉬움/어려움 페널티)
        below = (self.band_L - cur).clamp(min=0.0)
        above = (cur - self.band_H).clamp(min=0.0)
        band  = -(below * below + above * above)             # [1,S,1,1,1]

        # 5) UCB 탐험(옵션)
        self.total_steps += 1
        if self.gamma > 0:
            # log term 은 상수 텐서로
            log_t = torch.log(torch.tensor(float(self.total_steps + 1), device=cur.device))
            ucb = torch.sqrt((2.0 * log_t) / (self.sel_count + 1.0))
        else:
            ucb = torch.zeros_like(cur)

        # 6) 점수 맵으로 확장 & 마스킹
        score = (self.alpha * adv + self.beta * band + self.gamma * ucb).expand_as(proxy_map)  # [B,S,1,H,W]
        if mask is not None:
            score = score.masked_fill(mask <= 0, float('-inf'))

        # 7) soft/hard 게이팅
        if hard:
            choice = score.argmax(dim=1, keepdim=True)      # [B,1,1,H,W]
            alpha  = torch.zeros_like(score).scatter_(1, choice, 1.0)
            if mask is not None:
                alpha = alpha * (mask > 0).float()
            # 선택 카운트는 공간평균으로 1step당 1증가 정도만 누적(가벼운 근사)
            with torch.no_grad():
                self.sel_count += torch.ones_like(self.sel_count)
        else:
            alpha = F.softmax(score / self.temp, dim=1)     # [B,S,1,H,W]
            if mask is not None:
                alpha = alpha * (mask > 0).float()
                z = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
                alpha = alpha / z

        # 8) proxy EMA 업데이트
        with torch.no_grad():
            self._update_proxy_ema(proxy_map, mask)

        return alpha  # [B,S,1,H,W]

    # -------------------- 유틸 -------------------- #
    @torch.no_grad()
    def reset_stats(self):
        """통계/카운터 초기화"""
        self.ema_mu.zero_()
        self.ema_sigma.fill_(1.0)
        self.ema_proxy.zero_()
        self.sel_count.zero_()
        self.total_steps = 0
