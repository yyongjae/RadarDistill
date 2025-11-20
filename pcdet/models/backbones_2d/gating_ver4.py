# sweep_gater_v4.py (grouped-conv router, no per-sweep loops)
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

def _assert_finite(name, t):
    """Check NaN only (±inf는 masked logits에서 허용)."""
    if t is None:
        return
    if torch.isnan(t).any():
        bad = torch.isnan(t).float().mean().item() * 100.0
        raise RuntimeError(f"[NaNGuard] {name} has NaN ({bad:.4f}%)")

class SweepGaterV4(nn.Module):
    """
    Sweep gating (Temporal MoE) with:
      - Heuristic score (advantage / band / UCB) ⊕ Learned router (grouped 1x1 encoder → cross-sweep 1x1 mixer)
      - Positional (Δt/index/sin) embedding per sweep
      - Optional teacher confidence / entropy channels
      - Soft / Top-k-soft / Hard-ST / Hard-Gumbel gating
      - Capacity factor reflected in effective-k
      - Logits masking by validity mask (+ all -inf guard)
      - EMA warmup & moments (FP32)
      - Aux losses: Load-balance / Entropy / TV smoothing
    """

    def __init__(
        self,
        C: int,
        num_sweeps: int,
        # temps & schedules
        temp: float = 0.7,
        # adapters & EMA
        use_adapter: bool = True,
        ema_momentum: float = 0.99,
        # memory knobs
        gating_downsample: int = 1,  # 1/2/4/8
        # heuristic weights
        alpha_adv: float = 1.0,
        beta_band: float = 0.5,
        gamma_ucb: float = 0.0,
        band_L: float = 0.05,
        band_H: float = 0.20,
        # learned router
        learn_router: bool = True,
        router_hidden: int = 64,
        router_use_spatial: bool = False,   # Depthwise 3x3 → Pointwise 1x1 (per-sweep)
        w_heur: float = 0.5,
        w_lear: float = 0.5,
        # extra channels
        use_pos: bool = True,
        pos_mode: str = "sin",              # 'sin' | 'index' | 'delta'
        pos_channels: int = 2,              # sin/cos=2, index/delta=1
        use_conf: bool = False,
        use_entropy: bool = False,
        # gating mode
        mode: str = 'soft',                 # 'soft'|'topk_soft'|'hard_st'|'hard_gumbel'
        k: int = 2,
        # capacity & losses
        capacity_factor: float = 1.25,
        lb_coeff: float = 0.0,
        ent_coeff: float = 0.0,
        tv_coeff: float = 0.0,
        # misc
        persistent_ema: bool = False,
        use_runtime_assert: bool = False,
        # compression
        compress_ratio: int = 8,            # per-sweep compression ratio (grouped 1x1 출력 채널 축소비)
    ):
        super().__init__()
        S = int(num_sweeps)
        self.S_cnt = S
        self.temp = float(temp)
        self.use_adapter = bool(use_adapter)
        self.ema_momentum = float(ema_momentum)
        self.gating_downsample = int(gating_downsample)
        self.compress_ratio = int(max(1, compress_ratio))

        # heuristic params
        self.alpha_adv = float(alpha_adv)
        self.beta_band = float(beta_band)
        self.gamma_ucb = float(gamma_ucb)
        self.band_L = float(band_L)
        self.band_H = float(band_H)

        # adapters per sweep
        self.adapters = nn.ModuleList([nn.Conv2d(C, C, 1) for _ in range(S)]) if use_adapter else None

        # EMA buffers (FP32 유지)
        self.register_buffer('ema_mu',    torch.zeros(1, S, C, 1, 1, dtype=torch.float32), persistent=persistent_ema)
        self.register_buffer('ema_sigma', torch.ones (1, S, C, 1, 1, dtype=torch.float32), persistent=persistent_ema)
        self.register_buffer('ema_proxy', torch.zeros(1, S, 1, 1, 1, dtype=torch.float32), persistent=persistent_ema)
        self.register_buffer('sel_count', torch.zeros(1, S, 1, 1, 1, dtype=torch.float32), persistent=persistent_ema)
        self.total_steps = 0

        # learned router head (grouped encoder → (optional) DW3x3/PW1x1 → cross-sweep mixer)
        self.learn_router = bool(learn_router)
        self.use_pos = bool(use_pos)
        self.pos_mode = str(pos_mode)
        self.pos_channels = int(pos_channels) if self.use_pos else 0
        self.use_conf = bool(use_conf)
        self.use_entropy = bool(use_entropy)
        self.router_use_spatial = bool(router_use_spatial)
        self.use_runtime_assert = bool(use_runtime_assert)

        if self.learn_router:
            extra_per_sweep = (self.pos_channels if self.use_pos else 0) \
                              + (1 if self.use_conf else 0) \
                              + (1 if self.use_entropy else 0)
            # per-sweep 입력 채널 (Sz,Tz,Δ,[pos/conf/entropy])
            self.ci_per_sweep = 3 * C + extra_per_sweep
            # per-sweep 압축 출력 채널
            self.co_per_sweep = max(1, self.ci_per_sweep // self.compress_ratio)

            in_ch = S * self.ci_per_sweep
            enc_ch = S * self.co_per_sweep

            # 1) per-sweep grouped 1x1 encoder (스윕별 독립 압축)
            self.encoder_g1x1 = nn.Conv2d(in_ch, enc_ch, kernel_size=1, groups=S, bias=False)

            # 2) (옵션) per-sweep DW 3x3 + per-sweep PW 1x1 (여전히 한 번의 호출)
            if self.router_use_spatial:
                # depthwise: 채널별 컨볼루션(스윕/채널 완전 독립)
                self.router_dw = nn.Conv2d(enc_ch, enc_ch, kernel_size=3, padding=1, groups=enc_ch, bias=False)
                # per-sweep pointwise: 스윕별 1x1(그룹=S)
                self.router_pw = nn.Conv2d(enc_ch, enc_ch, kernel_size=1, groups=S, bias=False)
                self.router_gn1 = nn.GroupNorm(num_groups=min(32, enc_ch), num_channels=enc_ch)
                self.router_gn2 = nn.GroupNorm(num_groups=min(32, enc_ch), num_channels=enc_ch)
            else:
                self.router_dw = None
                self.router_pw = None
                self.router_gn1 = None
                self.router_gn2 = None

            # 3) cross-sweep mixer: 전 스윕을 한 번에 섞어 S개의 로짓 출력
            self.mixer = nn.Sequential(
                nn.GroupNorm(num_groups=min(32, enc_ch), num_channels=enc_ch),
                nn.GELU(),
                nn.Conv2d(enc_ch, router_hidden, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=min(32, router_hidden), num_channels=router_hidden),
                nn.GELU(),
                nn.Conv2d(router_hidden, S, kernel_size=1, bias=True),
            )
        else:
            self.ci_per_sweep = None
            self.co_per_sweep = None
            self.encoder_g1x1 = None
            self.router_dw = None
            self.router_pw = None
            self.router_gn1 = None
            self.router_gn2 = None
            self.mixer = None

        self.w_heur = float(w_heur)
        self.w_lear = float(w_lear)

        assert mode in ['soft', 'topk_soft', 'hard_st', 'hard_gumbel']
        self.mode = mode
        self.k = int(max(1, min(k, S)))

        self.capacity_factor = float(capacity_factor)
        self.lb_coeff = float(lb_coeff)
        self.ent_coeff = float(ent_coeff)
        self.tv_coeff = float(tv_coeff)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------- utils ----------
    @staticmethod
    def _topk_mask(logits, k):
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=1)
        mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0).bool()
        return mask, topk_idx, topk_vals

    @staticmethod
    def _gumbel_like(x, eps=1e-9):
        u = torch.empty_like(x).uniform_(0, 1)
        return -torch.log(-torch.log(u + eps) + eps)

    @staticmethod
    def _tv2d(x):
        dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return dh + dw

    # ---------- positional embedding ----------
    def _make_pos(self, B, S, H, W, device, sweep_times=None):
        if not self.use_pos or self.pos_channels == 0:
            return None
        if self.pos_mode == 'sin':
            idx = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1, 1)
            pos = torch.cat([torch.sin(2 * math.pi * idx), torch.cos(2 * math.pi * idx)], dim=2)
        elif self.pos_mode == 'index':
            pos = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1, 1)
        elif self.pos_mode == 'delta':
            if sweep_times is None:
                pos = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1, 1)
            else:
                t = sweep_times.to(device).view(1, S, 1, 1, 1).float()
                t0 = t.max(dim=1, keepdim=True)[0]
                dt = (t0 - t).clamp_min(0.0)
                pos = dt / (dt.max().clamp_min(1e-6))
        else:
            pos = torch.linspace(0, 1, S, device=device).view(1, S, 1, 1, 1)
        return pos.expand(B, -1, -1, H, W)

    # ---------- stats update (FP32) ----------
    @torch.no_grad()
    def update_ema(self, T, mask=None, momentum=None):
        assert T.dim() == 5
        m = self.ema_momentum if momentum is None else float(momentum)
        T32 = T.float()
        if mask is None:
            mu = T32.mean(dim=(0, 3, 4), keepdim=True)
            var = T32.var(dim=(0, 3, 4), keepdim=True, unbiased=False)
        else:
            W = mask.clamp(min=0.0).float()
            Wc = W.expand_as(T32)
            denom = Wc.sum(dim=(0, 3, 4), keepdim=True).clamp_min(1e-6)
            mu = (T32 * Wc).sum(dim=(0, 3, 4), keepdim=True) / denom
            var = ((T32 - mu).pow(2) * Wc).sum(dim=(0, 3, 4), keepdim=True) / denom
        sigma = var.clamp_min(0).sqrt().clamp_min(1e-6)

        if self.ema_mu.shape[1] != mu.shape[1]:
            self.ema_mu, self.ema_sigma = mu.detach(), sigma.detach()
        else:
            self.ema_mu = m * self.ema_mu + (1 - m) * mu
            self.ema_sigma = m * self.ema_sigma + (1 - m) * sigma

    @torch.no_grad()
    def warmup_ema(self, T, steps: int = 10, mask=None):
        for _ in range(max(1, steps)):
            self.update_ema(T, mask=mask)

    @torch.no_grad()
    def _update_proxy_ema(self, proxy_map, mask=None):
        cur = self._scalar_per_sweep(proxy_map, mask).float()
        self.ema_proxy = self.ema_momentum * self.ema_proxy + (1 - self.ema_momentum) * cur

    # ---------- align & norm ----------
    def _apply_adapters(self, S, T):
        B, S_cnt, C, H, W = T.shape
        if self.adapters is not None:
            assert len(self.adapters) == S_cnt
            T_adapt = torch.stack([self.adapters[i](T[:, i]) for i in range(S_cnt)], dim=1)
            S_rep = torch.stack([self.adapters[i](S) for i in range(S_cnt)], dim=1)
        else:
            T_adapt = T
            S_rep = S.unsqueeze(1).expand(-1, S_cnt, -1, -1, -1)
        return S_rep, T_adapt

    def _adapt_and_norm(self, S, T):
        S_rep, T_adapt = self._apply_adapters(S, T)
        sigma = torch.nan_to_num(self.ema_sigma, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(1e-3)
        Tz32 = (T_adapt.float() - self.ema_mu).div(sigma)
        Sz32 = (S_rep.float()  - self.ema_mu).div(sigma)
        Tz32 = torch.nan_to_num(Tz32, nan=0.0, posinf=0.0, neginf=0.0)
        Sz32 = torch.nan_to_num(Sz32, nan=0.0, posinf=0.0, neginf=0.0)
        dtype = S.dtype
        return Sz32.to(dtype), Tz32.to(dtype)

    def _masked_channel_l2(self, delta, mask=None):
        if mask is None:
            result = (delta ** 2).mean(dim=2, keepdim=True)
        else:
            num = (delta ** 2).sum(dim=2, keepdim=True)
            den = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
            result = num / den
        return torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    def _scalar_per_sweep(self, proxy_map, mask=None):
        if mask is None:
            return proxy_map.mean(dim=(0, 3, 4), keepdim=True)
        w = mask.clamp(min=0.0).float()
        num = (proxy_map * w).sum(dim=(0, 3, 4), keepdim=True)
        den = w.sum(dim=(0, 3, 4), keepdim=True).clamp_min(1e-6)
        return num / den

    # ---------- gating core ----------
    def _heuristic_score(self, cur, prev, proxy_map, mask):
        if self.use_runtime_assert:
            _assert_finite("heur_cur", cur); _assert_finite("heur_prev", prev)
            _assert_finite("heur_proxy_map", proxy_map)

        impr = (prev - cur).detach()
        if self.use_runtime_assert:
            _assert_finite("heur_impr", impr)

        adv = impr - impr.mean(dim=1, keepdim=True)
        if self.use_runtime_assert:
            _assert_finite("heur_adv", adv)

        below = (self.band_L - cur).clamp(min=0.0)
        above = (cur - self.band_H).clamp(min=0.0)
        band = -(below * below + above * above)
        if self.use_runtime_assert:
            _assert_finite("heur_band", band)

        self.total_steps += 1
        if self.gamma_ucb > 0:
            log_t = math.log1p(float(self.total_steps))
            log_t = torch.tensor(log_t, device=cur.device, dtype=cur.dtype)
            ucb = torch.sqrt((2.0 * log_t) / (self.sel_count + 1.0))
            if self.use_runtime_assert:
                _assert_finite("heur_ucb", ucb)
        else:
            ucb = torch.zeros_like(cur)

        score_compact = (self.alpha_adv * adv + self.beta_band * band + self.gamma_ucb * ucb)
        if self.use_runtime_assert:
            _assert_finite("heur_score_compact", score_compact)

        score = score_compact.expand_as(proxy_map)
        if self.use_runtime_assert:
            _assert_finite("heur_score_before_mask", score)

        if mask is not None:
            score = score.masked_fill(mask <= 0, float('-inf'))
            if self.use_runtime_assert:
                _assert_finite("heur_score_after_mask", score)
        return score  # [B,S,1,H,W]

    def _learned_score(self, Sz, Tz, delta, pos=None, conf=None, entropy=None):
        if not self.learn_router:
            return None

        xs = [Sz, Tz, delta]  # [B,S,C,H,W] each
        if pos is not None:
            xs.append(pos)  # [B,S,Ep,H,W]
        if self.use_conf:
            xs.append(conf if conf is not None else Sz.new_zeros(Sz.size(0), Sz.size(1), 1, Sz.size(3), Sz.size(4)))
        if self.use_entropy:
            xs.append(entropy if entropy is not None else Sz.new_zeros(Sz.size(0), Sz.size(1), 1, Sz.size(3), Sz.size(4)))

        x = torch.cat(xs, dim=2)  # [B,S,(3C+extra),H,W]
        B, S, Cx, H, W = x.shape
        assert Cx == self.ci_per_sweep, f"router in-per-sweep mismatch: got {Cx}, expect {self.ci_per_sweep}"

        # 라우터 경로만 AMP
        with autocast():
            # 0) downsample 한 번에 (루프 없음)
            x_flat = x.reshape(B, S * Cx, H, W)  # [B, S·Cx, H, W]
            if self.gating_downsample > 1:
                x_flat = F.avg_pool2d(
                    x_flat, kernel_size=self.gating_downsample, stride=self.gating_downsample
                )
            _, _, Hd, Wd = x_flat.shape

            # 1) per-sweep grouped 1x1 encoder: [B, S·Cx, Hd, Wd] → [B, S·Co, Hd, Wd]
            enc = self.encoder_g1x1(x_flat)

            # 2) (옵션) per-sweep DW3x3 → per-sweep PW1x1 (여전히 한 번에)
            if self.router_use_spatial:
                enc = self.router_gn1(enc)
                enc = F.gelu(self.router_dw(enc))
                enc = self.router_gn2(enc)
                enc = F.gelu(self.router_pw(enc))

            # 3) cross-sweep mixer → [B, S, Hd, Wd]
            logits = self.mixer(enc)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)

            # 업샘플 (필요 시)
            if self.gating_downsample > 1:
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits  # [B,S,H,W]

    def _capacity_mask(self, logits, k, capacity_factor):
        k_eff = max(1, min(logits.size(1), int(round(k * max(1e-6, capacity_factor)))))
        return self._topk_mask(logits, k_eff)[0]

    @torch.no_grad()
    def auto_step(self, epoch: int, weights: torch.Tensor = None, avg_weights: torch.Tensor = None, schedule: dict = None):
        default = dict(
            E1=2, E2=6,
            tau0=1.0, tau1=0.7, tau2=0.4,
            hard_mode='hard_st',
            k_min=1, k_max=self.k,
            bandL0=0.00, bandH0=0.50,
            bandL1=0.05, bandH1=0.25,
            bandL2=0.08, bandH2=0.18,
        )
        if schedule is None:
            schedule = default
        else:
            for kk in default:
                schedule.setdefault(kk, default[kk])

        E1, E2 = schedule['E1'], schedule['E2']
        tau0, tau1, tau2 = float(schedule['tau0']), float(schedule['tau1']), float(schedule['tau2'])
        hard_mode = schedule['hard_mode']
        k_min, k_max = int(schedule['k_min']), int(schedule['k_max'])

        if epoch < E1:
            self.mode = 'soft'
            t = 0.0 if E1 <= 1 else max(0.0, min(1.0, epoch / (E1 - 1)))
            self.temp = tau0 + (tau1 - tau0) * t
            self.band_L = schedule['bandL0'] + (schedule['bandL1'] - schedule['bandL0']) * t
            self.band_H = schedule['bandH0'] + (schedule['bandH1'] - schedule['bandH0']) * t
        elif epoch < E2:
            self.mode = 'topk_soft'
            span = max(E2 - E1, 1)
            t = max(0.0, min(1.0, (epoch - E1) / span))
            self.temp = tau1 + (tau2 - tau1) * t
            self.band_L = schedule['bandL1'] + (schedule['bandL2'] - schedule['bandL1']) * t
            self.band_H = schedule['bandH1'] + (schedule['bandH2'] - schedule['bandH1']) * t
        else:
            self.mode = hard_mode
            self.temp = tau2
            self.band_L = schedule['bandL2']
            self.band_H = schedule['bandH2']

        try:
            if avg_weights is not None and torch.is_tensor(avg_weights):
                var = float(avg_weights.var().item())
                if var < 1e-5 and self.k < k_max:
                    self.k += 1
                elif var > 1e-3 and self.k > k_min:
                    self.k -= 1
                self.k = int(max(k_min, min(self.k, k_max)))
        except Exception:
            pass

    # ---------- forward ----------
    def forward(
        self,
        S, T, mask=None, return_aux=False, temperature=None, k=None,
        conf=None, entropy=None, sweep_times=None,
    ):
        """
        S: student feat [B,C,H,W]
        T: teacher feat [B,S,C,H,W]
        mask: validity [B,S,1,H,W]
        conf/entropy: optional [B,S,1,H,W]
        sweep_times: [B,S] or [S]
        """
        B, S_cnt, C, H, W = T.shape
        assert S_cnt == self.ema_mu.shape[1], "Call update_ema()/warmup_ema() after changing sweep count."

        Sz, Tz = self._adapt_and_norm(S, T)
        delta = Tz - Sz

        proxy_map = self._masked_channel_l2(delta, mask)               # [B,S,1,H,W]
        cur = self._scalar_per_sweep(proxy_map, mask)                   # [1,S,1,1,1]
        prev = self.ema_proxy                                          # [1,S,1,1,1] (FP32)

        score_heur_raw = self._heuristic_score(cur, prev, proxy_map, mask)  # [B,S,1,H,W]
        if self.use_runtime_assert:
            _assert_finite("score_heur_before_squeeze", score_heur_raw)

        score_heur = score_heur_raw.squeeze(2)  # [B,S,H,W]
        if self.use_runtime_assert:
            _assert_finite("score_heur", score_heur)

        pos = self._make_pos(B, S_cnt, H, W, device=T.device, sweep_times=sweep_times) if self.use_pos else None
        if self.use_runtime_assert and pos is not None:
            _assert_finite("pos", pos)

        score_learn = self._learned_score(Sz, Tz, delta, pos=pos, conf=conf, entropy=entropy)
        if self.use_runtime_assert and score_learn is not None:
            _assert_finite("score_learn", score_learn)

        logits = score_heur if score_learn is None else (self.w_heur * score_heur + self.w_lear * score_learn)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=-30.0, max=30.0)

        # invalid mask 적용 (softmax 전) + all-invalid guard
        if mask is not None:
            invalid = (mask.squeeze(2) <= 0)                # [B,S,H,W]
            logits = logits.masked_fill(invalid, float('-inf'))

            # 모든 sweep이 invalid인 픽셀은 softmax 전에 0으로 되돌려 NaN 방지
            all_invalid = invalid.all(dim=1, keepdim=True)  # [B,1,H,W]
            if all_invalid.any():
                logits = torch.where(all_invalid, torch.zeros_like(logits), logits)

        # ---- 소프트맥스/TopK는 FP32에서 안전하게
        if logits.dtype != torch.float32:
            logits = logits.float()
        tau = float(self.temp if temperature is None else temperature)
        kk = self.k if k is None else min(max(1, k), S_cnt)

        # ---- gating
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
            weights = (weights * mask_k) / (weights * mask_k).sum(dim=1, keepdim=True).clamp_min(1e-9)
        elif self.mode == 'hard_gumbel':
            g = self._gumbel_like(logits)
            y = F.softmax((logits + g) / tau, dim=1)
            mask_k = self._capacity_mask(logits + g, kk, self.capacity_factor).float()
            hard = (y * mask_k) / (y * mask_k).sum(dim=1, keepdim=True).clamp_min(1e-9)
            weights = hard + (y - y.detach())
        else:
            raise ValueError

        # ---- dead-pixel guard (모든 sweep invalid인 위치 정규화)
        if mask is not None:
            valid = (mask.squeeze(2) > 0).float()
            weights = weights * valid
            z = weights.sum(dim=1, keepdim=True)
            dead = (z <= 1e-12)
            if dead.any():
                fill = torch.full_like(weights, 1.0 / weights.size(1))
                weights = torch.where(dead, fill, weights / z.clamp_min(1e-9))
            else:
                weights = weights / z.clamp_min(1e-9)

        if self.use_runtime_assert:
            _assert_finite("Sz", Sz); _assert_finite("Tz", Tz)
            _assert_finite("delta", delta); _assert_finite("proxy_map", proxy_map)
            _assert_finite("logits", logits); _assert_finite("weights", weights)

        # ---- stats update (FP32 누적)
        with torch.no_grad():
            _, topk_idx, _ = self._topk_mask(logits, max(1, min(S_cnt, int(round(kk * max(1e-6, self.capacity_factor))))))
            sel_mask = torch.zeros_like(logits).scatter(1, topk_idx, 1.0)     # [B,S,H,W]
            inc = sel_mask.mean(dim=(0, 2, 3), keepdim=True).float()          # [1,S,1,1]
            self.sel_count += inc.unsqueeze(2)                                # [1,S,1,1,1]
            self._update_proxy_ema(proxy_map, mask)

        # ---- aux losses
        aux_losses = {}
        if self.lb_coeff > 0:
            imp = weights.mean(dim=(0, 2, 3))
            aux_losses['L_lb'] = self.lb_coeff * imp.var()
        if self.ent_coeff > 0:
            H_ = -(weights.clamp_min(1e-9) * weights.clamp_min(1e-9).log()).sum(dim=1).mean()
            aux_losses['L_ent'] = self.ent_coeff * H_
        if self.tv_coeff > 0:
            aux_losses['L_tv'] = self.tv_coeff * self._tv2d(weights)

        if return_aux:
            stats = {
                'cur': cur.detach().cpu(),
                'prev': prev.detach().cpu(),
                'avg_weights': weights.mean(dim=(0, 2, 3)).detach().cpu(),
                'mode': self.mode, 'k': kk, 'tau': tau,
                'band_L': float(self.band_L), 'band_H': float(self.band_H),
            }
            return weights.unsqueeze(2), aux_losses, stats
        return weights.unsqueeze(2)
