import torch
import torch.nn as nn
import torch.nn.functional as F

class SweepGater(nn.Module):
    def __init__(self, C, num_sweeps=3, temp=0.7, use_adapter=True):
        super().__init__()
        self.temp = temp
        self.use_adapter = use_adapter
        self.adapters = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(num_sweeps)]) if use_adapter else None
        # [1, S, C, 1, 1]
        self.register_buffer('ema_mu',    torch.zeros(1, num_sweeps, C, 1, 1), persistent=False)
        self.register_buffer('ema_sigma', torch.ones (1, num_sweeps, C, 1, 1), persistent=False)
        self.ema_momentum = 0.99

    @torch.no_grad()
    def update_ema(self, T: torch.Tensor, momentum: float = None, mask: torch.Tensor = None):
        """
        T:    [B, S, C, H, W]
        mask: [B, S, 1, H, W] or None  (0=무효, >0=가중)
        """
        assert T.dim() == 5, f"Expected [B,S,C,H,W], got {T.shape}"
        if momentum is None: momentum = self.ema_momentum

        if mask is None:
            mu    = T.mean(dim=(0,3,4), keepdim=True)                          # [1,S,C,1,1]
            var   = T.var (dim=(0,3,4), keepdim=True, unbiased=False)
        else:
            W     = mask.clamp(min=0.0)                                        # [B,S,1,H,W]
            Wc    = W.expand_as(T)                                             # [B,S,C,H,W]
            denom = Wc.sum(dim=(0,3,4), keepdim=True).clamp_min(1e-6)
            mu    = (T * Wc).sum(dim=(0,3,4), keepdim=True) / denom
            var   = ((T - mu) ** 2 * Wc).sum(dim=(0,3,4), keepdim=True) / denom

        sigma = var.clamp_min(0).sqrt().clamp_min(1e-6)

        if self.ema_mu.shape[1] != mu.shape[1]:
            self.ema_mu, self.ema_sigma = mu.clone(), sigma.clone()
        else:
            self.ema_mu    = momentum * self.ema_mu    + (1 - momentum) * mu
            self.ema_sigma = momentum * self.ema_sigma + (1 - momentum) * sigma


    def _adapt_and_norm(self, s_feat: torch.Tensor, T: torch.Tensor):
        """s_feat:[B,C,H,W], T:[B,S,C,H,W] -> 정규화된 Sz,Tz 반환"""
        B, n_sweeps, C, H, W = T.shape
        if self.adapters is not None:
            assert len(self.adapters) == n_sweeps, f"adapters:{len(self.adapters)} != sweeps:{n_sweeps}"
            T = torch.stack([self.adapters[i](T[:, i])   for i in range(n_sweeps)], dim=1)  # [B,S,C,H,W]
            S_rep = torch.stack([self.adapters[i](s_feat) for i in range(n_sweeps)], dim=1)  # [B,S,C,H,W]
        else:
            S_rep = s_feat.unsqueeze(1).expand(-1, n_sweeps, -1, -1, -1)

        Tz = (T     - self.ema_mu) / self.ema_sigma       # [B,S,C,H,W]
        Sz = (S_rep - self.ema_mu) / self.ema_sigma       # [B,S,C,H,W]
        return Sz, Tz


    def forward(self, S, T, hard: bool = False, mask: torch.Tensor = None):
        """
        S:    [B,C,H,W]
        T:    [B,S,C,H,W]
        mask: [B,S,1,H,W] or None (0=무효, >0=가중치)
        return alpha: [B,S,1,H,W]
        """
        Sz, Tz = self._adapt_and_norm(S, T)                 # [B,S,C,H,W]
        delta  = Tz - Sz

        # 0) 마스크 준비
        if mask is None:
            w = torch.ones_like(delta[:, :, :1])            # [B,S,1,H,W]
        else:
            w = mask.clamp(min=0.0)                         # same shape

        # 1) 채널 L2를 '마스킹 평균'으로
        l2_num = w * (delta ** 2).sum(dim=2, keepdim=True)  # [B,S,1,H,W]
        l2_den = w.sum(dim=2, keepdim=True).clamp_min(1e-6)
        l2     = l2_num / l2_den

        # 2) 스윕 바이어스 제거(l2_m)도 마스킹 평균으로 (배치/공간 평균)
        l2m_num = (l2 * w).sum(dim=(0,3,4), keepdim=True)   # [1,S,1,1,1]
        l2m_den = w.sum(dim=(0,3,4), keepdim=True).clamp_min(1e-6)
        l2_m    = l2m_num / l2m_den

        # 3) 점수 & 무효 영역 제외
        denom  = (delta.norm(dim=2, keepdim=True) + 1e-4)
        score  = -(l2 - l2_m) / denom                       # [B,S,1,H,W]
        score  = score.masked_fill(w <= 0, float('-inf'))   # 무효 sweep은 경쟁 제외

        # 4) soft/hard 게이팅
        if hard:
            choice = score.argmax(dim=1, keepdim=True)      # [B,1,1,H,W]
            alpha  = torch.zeros_like(score).scatter_(1, choice, 1.0)
            alpha  = alpha * (w > 0).float()                # 무효 위치는 0
        else:
            alpha  = F.softmax(score / self.temp, dim=1)
            alpha  = alpha * (w > 0).float()                # 무효 위치 제거
            z = alpha.sum(dim=1, keepdim=True).clamp_min(1e-6)
            alpha = alpha / z                                # 유효 sweep 합=1로 재정규화

        return alpha