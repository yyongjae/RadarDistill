import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SweepGaterV3(nn.Module):
    """
    Sweep gating (Temporal MoE) with:
      - Heuristic score (advantage/band/UCB) ⊕ Learned router (1x1 conv)
      - Soft / Top-k-soft / Hard-ST / Hard-Gumbel gating
      - Capacity & Load-balance loss, Entropy & TV smoothing regularizers
    """
    def __init__(
        self,
        C: int,
        num_sweeps: int,
        temp: float = 0.7,
        use_adapter: bool = True,
        ema_momentum: float = 0.99,
        # heuristic weights
        alpha_adv: float = 1.0,
        beta_band: float = 0.5,
        gamma_ucb: float = 0.0,
        band_L: float = 0.05,
        band_H: float = 0.20,
        # learned router
        learn_router: bool = True,
        router_hidden: int = 64,
        w_heur: float = 0.5,
        w_lear: float = 0.5,
        # gating mode
        mode: str = 'soft',            # default soft; you can schedule later
        k: int = 2,
        # capacity & losses
        capacity_factor: float = 1.25,
        lb_coeff: float = 0.0,
        ent_coeff: float = 0.0,
        tv_coeff: float = 0.0,
    ):
        super().__init__()
        S = num_sweeps
        self.temp = float(temp)
        self.use_adapter = use_adapter
        self.ema_momentum = float(ema_momentum)

        # heuristic params
        self.alpha_adv = float(alpha_adv)
        self.beta_band = float(beta_band)
        self.gamma_ucb = float(gamma_ucb)
        self.band_L = float(band_L)
        self.band_H = float(band_H)

        # adapters per sweep (align channels)
        self.adapters = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(S)]) if use_adapter else None

        # EMA buffers
        self.register_buffer('ema_mu',    torch.zeros(1, S, C, 1, 1), persistent=False)
        self.register_buffer('ema_sigma', torch.ones (1, S, C, 1, 1), persistent=False)
        self.register_buffer('ema_proxy', torch.zeros(1, S, 1, 1, 1), persistent=False)
        self.register_buffer('sel_count', torch.zeros(1, S, 1, 1, 1), persistent=False)
        self.total_steps = 0

        # learned router head (small conv): inputs = [Sz, Tz, delta]
        self.learn_router = learn_router
        if learn_router:
            in_ch = S*C*3
            self.router = nn.Sequential(
                nn.Conv2d(in_ch, router_hidden, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(router_hidden, S, 1)
            )
        self.w_heur = float(w_heur)
        self.w_lear = float(w_lear)

        # gating mode
        assert mode in ['soft','topk_soft','hard_st','hard_gumbel']
        self.mode = mode
        self.k = int(max(1, min(k, S)))

        # capacity & losses
        self.capacity_factor = float(capacity_factor)
        self.lb_coeff = float(lb_coeff)
        self.ent_coeff = float(ent_coeff)
        self.tv_coeff = float(tv_coeff)

    # ---------- utils ----------
    @staticmethod
    def _topk_mask(logits, k):
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=1)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0).bool()
        return mask, topk_idx, topk_vals

    @staticmethod
    def _gumbel_like(x, eps=1e-9):
        u = torch.empty_like(x).uniform_(0,1)
        return -torch.log(-torch.log(u + eps) + eps)

    @staticmethod
    def _tv2d(x):
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw

    # ---------- stats update ----------
    @torch.no_grad()
    def update_ema(self, T, mask=None, momentum=None):
        assert T.dim() == 5
        m = self.ema_momentum if momentum is None else momentum
        if mask is None:
            mu  = T.mean(dim=(0,3,4), keepdim=True)
            var = T.var (dim=(0,3,4), keepdim=True, unbiased=False)
        else:
            W  = mask.clamp(min=0.0)
            Wc = W.expand_as(T)
            denom = Wc.sum(dim=(0,3,4), keepdim=True).clamp_min(1e-6)
            mu  = (T*Wc).sum(dim=(0,3,4), keepdim=True)/denom
            var = ((T-mu).pow(2)*Wc).sum(dim=(0,3,4), keepdim=True)/denom
        sigma = var.clamp_min(0).sqrt().clamp_min(1e-6)

        if self.ema_mu.shape[1] != mu.shape[1]:
            self.ema_mu, self.ema_sigma = mu.clone(), sigma.clone()
        else:
            self.ema_mu    = m*self.ema_mu    + (1-m)*mu
            self.ema_sigma = m*self.ema_sigma + (1-m)*sigma

    @torch.no_grad()
    def _update_proxy_ema(self, proxy_map, mask=None):
        cur = self._scalar_per_sweep(proxy_map, mask)  # [1,S,1,1,1]
        self.ema_proxy = self.ema_momentum*self.ema_proxy + (1-self.ema_momentum)*cur

    # ---------- align & norm ----------
    def _apply_adapters(self, S, T):
        B, S_cnt, C, H, W = T.shape
        if self.adapters is not None:
            assert len(self.adapters) == S_cnt
            T_adapt = torch.stack([self.adapters[i](T[:,i]) for i in range(S_cnt)], dim=1)
            S_rep   = torch.stack([self.adapters[i](S)      for i in range(S_cnt)], dim=1)
        else:
            T_adapt = T
            S_rep   = S.unsqueeze(1).expand(-1, S_cnt, -1, -1, -1)
        return S_rep, T_adapt

    def _adapt_and_norm(self, S, T):
        S_rep, T_adapt = self._apply_adapters(S, T)
        Tz = (T_adapt - self.ema_mu) / self.ema_sigma
        Sz = (S_rep   - self.ema_mu) / self.ema_sigma
        return Sz, Tz

    def _masked_channel_l2(self, delta, mask=None):
        if mask is None:
            return (delta ** 2).mean(dim=2, keepdim=True)
        num = (delta ** 2).sum(dim=2, keepdim=True)
        den = mask.sum(dim=2, keepdim=True).clamp_min(1e-6)
        return num / den

    def _scalar_per_sweep(self, proxy_map, mask=None):
        if mask is None:
            return proxy_map.mean(dim=(0,3,4), keepdim=True)
        w   = mask.clamp(min=0.0)
        num = (proxy_map*w).sum(dim=(0,3,4), keepdim=True)
        den = w.sum(dim=(0,3,4), keepdim=True).clamp_min(1e-6)
        return num/den

    # ---------- gating core ----------
    def _heuristic_score(self, cur, prev, proxy_map, mask):
        impr = (prev - cur).detach()
        adv  = impr - impr.mean(dim=1, keepdim=True)
        below = (self.band_L - cur).clamp(min=0.0)
        above = (cur - self.band_H).clamp(min=0.0)
        band  = -(below*below + above*above)
        self.total_steps += 1
        if self.gamma_ucb > 0:
            log_t = torch.log(torch.tensor(float(self.total_steps+1), device=cur.device))
            ucb = torch.sqrt((2.0*log_t) / (self.sel_count + 1.0))
        else:
            ucb = torch.zeros_like(cur)
        score = (self.alpha_adv*adv + self.beta_band*band + self.gamma_ucb*ucb).expand_as(proxy_map)
        if mask is not None:
            score = score.masked_fill(mask <= 0, float('-inf'))
        return score

    def _learned_score(self, Sz, Tz, delta):
        # Sz/Tz/delta: [B,S,C,H,W] -> concat on channel dim=2 => [B,S,3C,H,W]
        if not self.learn_router:
            return None
        x = torch.cat([Sz, Tz, delta], dim=2)  # [B,S,3C,H,W]
        B, S, C3, H, W = x.shape               # C3 = 3*C
        # ✅ 스윕 S까지 채널로 펴서 in_ch = S*3C로 맞춘다
        x = x.reshape(B, S * C3, H, W)         # [B, S*3C, H, W]
        return self.router(x)                  # [B,S,H,W]

    def _capacity_mask(self, logits, k, capacity_factor):
        return self._topk_mask(logits, k)[0]
    
    @torch.no_grad()
    def auto_step(
        self,
        epoch: int,
        weights: torch.Tensor = None,      # [B,S,1,H,W] or None
        avg_weights: torch.Tensor = None,  # [S] on CPU or None
        schedule: dict = None              # 세부 스케줄 커스터마이즈 용
    ):
        """
        에폭 기반 자동 전환:
          0 ~ E1-1:  soft,   tau 선형 감소
          E1 ~ E2-1: topk_soft, tau 더 낮춤
          E2 ~      : hard_st (원하면 hard_gumbel로 변경)
        또한 avg_weights 분산이 너무 낮으면(top-k가 과도하게 한 스윕만 고르면)
        k를 1→2→3 범위에서 미세 조정.
        """
        # ------ 기본 스케줄 ------
        default = dict(
            E1=2,        # soft -> topk_soft(시작)
            E2=6,        # topk_soft -> hard(시작)
            tau0=1.0,    # 초기 온도
            tau1=0.7,    # E1에서의 온도
            tau2=0.4,    # E2에서의 온도
            hard_mode='hard_st',  # or 'hard_gumbel'
            k_min=1, k_max=self.k  # k 조정 범위(초기 self.k 유지)
        )
        if schedule is None:
            schedule = default
        else:
            for k in default:
                schedule.setdefault(k, default[k])

        E1, E2 = schedule['E1'], schedule['E2']
        tau0, tau1, tau2 = float(schedule['tau0']), float(schedule['tau1']), float(schedule['tau2'])
        hard_mode = schedule['hard_mode']
        k_min, k_max = int(schedule['k_min']), int(schedule['k_max'])

        # ------ 모드 전환 ------
        if epoch < E1:
            self.mode = 'soft'
            # 선형 감소: tau = tau0 -> tau1
            if E1 > 0:
                t = max(0.0, min(1.0, epoch / max(E1 - 1, 1)))
                self.temp = tau0 + (tau1 - tau0) * t
            else:
                self.temp = tau1
        elif epoch < E2:
            self.mode = 'topk_soft'
            # 선형 감소: tau = tau1 -> tau2
            span = max(E2 - E1, 1)
            t = max(0.0, min(1.0, (epoch - E1) / span))
            self.temp = tau1 + (tau2 - tau1) * t
        else:
            self.mode = hard_mode
            self.temp = tau2

        # ------ k 미세 조정(선택): 라우팅 편향 완화 ------
        # avg_weights: [S] (각 스윕 평균 중요도). 분산이 너무 낮으면 k를 살짝 늘려 다양성 확보
        try:
            if avg_weights is not None and torch.is_tensor(avg_weights):
                var = float(avg_weights.var().item())
                # 임계치들은 경험칙, 필요시 조정
                if var < 1e-5 and self.k < k_max:
                    self.k += 1
                elif var > 1e-3 and self.k > k_min:
                    self.k -= 1
                self.k = int(max(k_min, min(self.k, k_max)))
        except Exception:
            pass  # 안전 가드

    # ---------- forward ----------
    def forward(self, S, T, mask=None, return_aux=False, temperature=None, k=None):
        B, S_cnt, C, H, W = T.shape
        assert S_cnt == self.ema_mu.shape[1], "Call update_ema() after changing sweep count."

        Sz, Tz = self._adapt_and_norm(S, T)
        delta  = Tz - Sz
        proxy_map = self._masked_channel_l2(delta, mask)         # [B,S,1,H,W]
        cur  = self._scalar_per_sweep(proxy_map, mask)           # [1,S,1,1,1]
        prev = self.ema_proxy

        score_heur = self._heuristic_score(cur, prev, proxy_map, mask).squeeze(2)
        score_learn = self._learned_score(Sz, Tz, delta)
        logits = score_heur if score_learn is None else (self.w_heur*score_heur + self.w_lear*score_learn)

        tau = self.temp if temperature is None else temperature
        kk  = self.k if k is None else min(max(1,k), S_cnt)

        if self.mode == 'soft':
            weights = F.softmax(logits / tau, dim=1)
        elif self.mode == 'topk_soft':
            w_full = F.softmax(logits / tau, dim=1)
            mask_k = self._capacity_mask(logits, kk, self.capacity_factor)
            weights = w_full * mask_k
            weights = weights / (weights.sum(dim=1, keepdim=True).clamp_min(1e-9))
        elif self.mode == 'hard_st':
            mask_k = self._capacity_mask(logits, kk, self.capacity_factor).float()
            soft = F.softmax(logits / tau, dim=1)
            weights = mask_k + (soft - soft.detach())
            weights = (weights * mask_k) / (weights*mask_k).sum(dim=1, keepdim=True).clamp_min(1e-9)
        elif self.mode == 'hard_gumbel':
            g = self._gumbel_like(logits)
            y = F.softmax((logits + g)/tau, dim=1)
            mask_k = self._capacity_mask(logits + g, kk, self.capacity_factor).float()
            hard = (y * mask_k) / (y*mask_k).sum(dim=1, keepdim=True).clamp_min(1e-9)
            weights = hard + (y - y.detach())
        else:
            raise ValueError

        if mask is not None:
            weights = weights * (mask.squeeze(2) > 0).float()
            z = weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
            weights = weights / z

        with torch.no_grad():
            _, topk_idx, _ = self._topk_mask(logits, kk)
            sel_mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)  # [B,S,H,W]
            inc = sel_mask.mean(dim=(0,2,3), keepdim=True)                 # [1,S,1,1]
            self.sel_count += inc.unsqueeze(2)           
            self._update_proxy_ema(proxy_map, mask)

        aux_losses = {}
        if self.lb_coeff > 0:
            imp = weights.mean(dim=(0,2,3))
            aux_losses['L_lb'] = self.lb_coeff * imp.var()
        if self.ent_coeff > 0:
            H_ = -(weights.clamp_min(1e-9)*weights.clamp_min(1e-9).log()).sum(dim=1).mean()
            aux_losses['L_ent'] = self.ent_coeff * H_
        if self.tv_coeff > 0:
            aux_losses['L_tv'] = self.tv_coeff * self._tv2d(weights)

        if return_aux:
            stats = {
                'cur': cur.detach().cpu(),
                'prev': prev.detach().cpu(),
                'avg_weights': weights.mean(dim=(0,2,3)).detach().cpu(),
                'mode': self.mode, 'k': kk, 'tau': tau
            }
            return weights.unsqueeze(2), aux_losses, stats
        return weights.unsqueeze(2)