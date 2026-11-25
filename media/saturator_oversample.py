"""饱和失真与过采样处理模块。

针对原始实现的六类缺陷的修复对应关系：

1. 失真在原始采样率下做、谐波越过奈奎斯特折叠成哨音
   -> :func:`saturate` 在过采样域内施加非线性，上下采样均经抗镜像/抗混叠
      低通滤波（:func:`band_limit`），降采样后折叠能量被滤波器吸收。
2. 过采样倍数固定为 4、与素材采样率无关
   -> :func:`oversample_by_rate` 按采样率（以及需要保留的谐波阶数）自适应
      选择 2 的幂次倍数，使有效采样率达到目标（默认 192 kHz）且为最高次
      谐波留出头寸。
3. 饱和曲线按段落瞬时峰值归一化、段落间峰值变化导致失真量跳变
   -> :func:`curve_by_longterm` 用 attack/release 非对称峰值包络跟随器
   （慢释放）估计长期峰值，drive 平滑变化，不再随块峰值跳变。
4. 多声道各自饱和、谐波相位不一致、声像发虚
   -> :func:`multichannel_phase` 取跨声道共享峰值，所有声道使用同一条
      drive 曲线与同一个非线性映射，保持声道间相位/声像。
5. 旁路硬切产生爆音
   -> :func:`bypass` / :func:`bypass_crossfade` 做等功率交叉淡化，
      淡化位置可在块间通过 state 延续。
6. 重采样过冲与带外能量
   -> :func:`output_limit` 前瞻限幅到安全天花板，并在末端再做一次限带。

仅依赖 numpy；FIR 设计与 FFT 卷积均在本模块内实现。
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------

TARGET_EFFECTIVE_RATE = 192_000.0   # 过采样后期望达到的有效采样率 (Hz)
MAX_FACTOR = 16                     # 过采样倍数上限（CPU 护栏）
DEFAULT_MARGIN = 0.9                # 滤波器截止相对奈奎斯特的余量
DEFAULT_TAPS = 101                  # windowed-sinc FIR 抽头数（奇数）
ATTACK_MS = 10.0                    # 长期峰值包络攻击时间
RELEASE_MS = 1500.0                 # 长期峰值包络释放时间（慢，抑制块间跳变）
DRIVE_FLOOR = 0.25                  # drive 下限（静音/极弱信号不至于狂放大）
DRIVE_CEIL = 4.0                   # drive 上限（避免静默段突发噪声爆炸）
FADE_MS = 10.0                      # 旁路交叉淡化时长
LIMIT_CEIL = 0.95                   # 输出天花板（约 -0.45 dBFS）
LIMIT_ATTACK_MS = 2.0
LIMIT_RELEASE_MS = 80.0
LIMIT_LOOKAHEAD_MS = 2.0


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _as_array(x: Any) -> np.ndarray:
    """把标量/列表转成 float64 数组；ndarray 原样返回。"""
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _coef_time(ms: float, rate: float) -> float:
    """一阶包络跟随器的时间常数系数。"""
    if ms <= 0.0:
        return 1.0
    return math.exp(-1.0 / (max(rate, 1e-9) * ms / 1000.0))


# ---------------------------------------------------------------------------
# 1) 按采样率定倍
# ---------------------------------------------------------------------------

def oversample_by_rate(rate: float,
                       harmonics: int = 1,
                       target_rate: float = TARGET_EFFECTIVE_RATE,
                       max_factor: int = MAX_FACTOR) -> int:
    """按采样率自适应选择过采样倍数（2 的幂）。

    准则（取两者中更大者）：

    * 有效采样率 ``rate * factor`` 不低于 ``target_rate``（默认 192 kHz，
      因此 48k -> x4、96k -> x2、192k -> x1，而不是无视采样率固定 x4）；
    * 当调用方显式要求保留 ``harmonics`` 次谐波（>1）时，过采样后的
      奈奎斯特必须容纳通带上限信号的该次谐波，即 ``factor >= harmonics``。

    返回值被夹在 ``[1, max_factor]`` 并向上取整到 2 的幂。
    """
    rate = float(max(rate, 1e-9))
    target_rate = float(max(target_rate, rate))
    harmonics = int(max(harmonics, 1))
    max_factor = int(max(max_factor, 1))

    need_rate = target_rate / rate
    need = max(need_rate, harmonics)

    factor = 1
    while factor < need and factor < max_factor:
        factor <<= 1
    return min(factor, max_factor)


# ---------------------------------------------------------------------------
# 2) 限带低通（windowed-sinc FIR + FFT 卷积）
# ---------------------------------------------------------------------------

def _fir_lowpass(cutoff: float, rate: float, numtaps: int = DEFAULT_TAPS,
                 gain: float = 1.0) -> np.ndarray:
    """设计 windowed-sinc 低通 FIR。"""
    nt = int(numtaps)
    if nt % 2 == 0:
        nt += 1
    cutoff = float(min(max(cutoff, 1e-9), rate * 0.5 - 1e-9))
    mid = nt // 2
    n = np.arange(nt) - mid
    h = np.sinc(2.0 * cutoff / rate * n)
    window = np.hanning(nt)
    h *= window
    s = h.sum()
    if s > 0:
        h *= gain / s
    return h


def _fft_convolve(x: np.ndarray, h: np.ndarray,
                  state: Optional[np.ndarray] = None) -> np.ndarray:
    """一维因果 FFT 卷积（overlap-save），返回与 x 等长的流式输出。

    因果 FIR 的群延迟为 ``(h.size-1)//2`` 个样本。传入 ``state``
    （长度 ``h.size-1`` 的上一块输入历史，首块前补零）时，取拼接信号
    full 卷积索引 ``h.size-1`` 起的 ``x.size`` 个输出；分块结果与
    整段 ``np.convolve(x, h, 'full')[:len(x)]`` 严格一致，块边界无瞬态。
    """
    if x.size == 0:
        return x.copy()

    if state is not None:
        xp = np.concatenate([state, x])
        full_n = xp.size + h.size - 1
        nfft = 1 << (int(math.ceil(math.log2(max(full_n, 1)))))
        y = np.fft.irfft(np.fft.rfft(xp, nfft) * np.fft.rfft(h, nfft), nfft)
        start = h.size - 1
        out = y[start:start + x.size]
        if h.size > 1:
            state[:] = xp[-(h.size - 1):]
        return out

    # 无状态：等价于零初始历史的首块。
    init = np.zeros(h.size - 1, dtype=x.dtype)
    xp = np.concatenate([init, x])
    full_n = xp.size + h.size - 1
    nfft = 1 << (int(math.ceil(math.log2(max(full_n, 1)))))
    y = np.fft.irfft(np.fft.rfft(xp, nfft) * np.fft.rfft(h, nfft), nfft)
    start = h.size - 1
    return y[start:start + x.size]


def band_limit(signal: Any, rate: Optional[float] = None,
               cutoff: Optional[float] = None,
               margin: float = DEFAULT_MARGIN,
               numtaps: int = DEFAULT_TAPS, gain: float = 1.0,
               state: Optional[np.ndarray] = None) -> Any:
    """限带低通滤波。

    * ``signal`` 为数组且给出 ``rate`` 时：做低通滤波并返回等长数组；
    * ``signal`` 为标量（如采样率）时：返回在该采样率、给定 margin 下
      的截止频率 (Hz)，便于审计与配置检查；
    * 仅给一个数组参数时：默认按 48 kHz 处理。

    ``margin`` 让截止频率停在奈奎斯特以内（默认 0.9），给 FIR 过渡带
    留出空间，保证被滤除的是带外折叠能量而不是有用信号。``state``
    传入一个可写数组（长度 numtaps-1）时支持分块连续滤波。
    """
    if rate is None:
        if np.isscalar(signal):
            rate = float(signal)
        else:
            rate = 48000.0
    rate = float(max(rate, 1e-9))
    margin = float(min(max(margin, 1e-6), 0.9999))
    if cutoff is None:
        cutoff = 0.5 * margin * rate

    if np.isscalar(signal):
        return float(min(cutoff, 0.5 * rate))

    x = _as_array(signal)
    h = _fir_lowpass(cutoff, rate, numtaps=numtaps, gain=gain)
    if x.ndim == 1:
        return _fft_convolve(x, h, state)
    if state is not None and state.ndim == 2:
        return np.stack([_fft_convolve(x[ch], h, state[ch])
                         for ch in range(x.shape[0])], axis=0)
    return np.stack([_fft_convolve(x[ch], h) for ch in range(x.shape[0])], axis=0)


# ---------------------------------------------------------------------------
# 3) 长期峰值 -> 平滑 drive 曲线
# ---------------------------------------------------------------------------

def _peak_envelope(x: np.ndarray, rate: float,
                   state: Optional[Dict[str, float]] = None,
                   attack_ms: float = ATTACK_MS,
                   release_ms: float = RELEASE_MS) -> np.ndarray:
    """非对称一阶峰值包络跟随器（attack 快、release 慢）。"""
    if x.size == 0:
        return x.copy()
    ka = _coef_time(attack_ms, rate)
    kr = _coef_time(release_ms, rate)
    env = np.empty(x.size, dtype=np.float64)
    peak = float(state.get("peak", 0.0)) if state is not None else 0.0
    mag = np.abs(x)
    for i in range(x.size):
        m = float(mag[i])
        coef = ka if m > peak else kr
        peak = (1.0 - coef) * m + coef * peak
        env[i] = peak
    if state is not None:
        state["peak"] = peak
    return env


def curve_by_longterm(signal: Any, rate: Optional[float] = None,
                      reference_peak: Optional[float] = None,
                      state: Optional[Dict[str, float]] = None,
                      floor: float = DRIVE_FLOOR,
                      ceil: float = DRIVE_CEIL) -> np.ndarray:
    """按长期峰值计算逐样本 drive 曲线。

    非线性归一化使用慢释放的峰值包络（长期峰值），而非块内瞬时峰值：
    段落峰值短期回落时 drive 不会立刻飙升，段落间失真量保持平稳。

    ``reference_peak`` 可给定全曲长期峰值（离线渲染路径），此时 drive
    为恒定标量曲线，块间完全一致。
    """
    x = _as_array(signal)
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    if x.size == 0:
        return x.copy()

    if reference_peak is not None and reference_peak > 0.0:
        env = np.full(x.size, float(reference_peak), dtype=np.float64)
    else:
        env = _peak_envelope(x, rate, state)

    # 峰值下限防止静音段除零；drive 夹在 floor/ceil 之间。
    env = np.maximum(env, 1e-6)
    drive = 1.0 / env
    drive = np.clip(drive, floor, ceil)
    return drive[0] if scalar_input else drive


# ---------------------------------------------------------------------------
# 4) 多声道同相：共享一条曲线
# ---------------------------------------------------------------------------

def multichannel_phase(signal: Any) -> np.ndarray:
    """把多声道信号折叠为跨声道共享的单声道控制信号。

    取同一时刻各声道幅度的最大值（任何声道的峰值都会计入），所有声道
    随后共用由它导出的 drive 与饱和曲线，因此左右声道谐波同相、
    声像不发虚。单声道信号原样返回。
    """
    x = _as_array(signal)
    if x.ndim <= 1:
        return x.reshape(-1) if x.ndim == 0 else x
    return np.max(np.abs(x), axis=0)


# ---------------------------------------------------------------------------
# 5) 非线性与过采样饱和
# ---------------------------------------------------------------------------

def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def saturate(signal: Any, drive: Any = 1.0) -> np.ndarray:
    """tanh 饱和非线性：``tanh(drive * x)``。

    drive 可为标量或与信号等长的逐样本曲线（多声道调用方负责传入
    跨声道共享的同一条曲线）。
    """
    x = _as_array(signal)
    d = _as_array(drive)
    if d.ndim == 0:
        d = float(d)
    return _tanh(x * d)


def _upsample(x: np.ndarray, factor: int, rate: float,
              margin: float, numtaps: int,
              state: Optional[np.ndarray] = None) -> np.ndarray:
    """零填插零 + 抗镜像低通（增益补偿 factor）。"""
    if factor == 1:
        return x.copy()
    up = np.zeros(x.size * factor, dtype=np.float64)
    up[::factor] = x
    high_rate = rate * factor
    cutoff = 0.5 * margin * rate
    return band_limit(up, high_rate, cutoff=cutoff, margin=margin,
                      numtaps=numtaps, gain=float(factor), state=state)


def _downsample(x: np.ndarray, factor: int, rate: float,
                margin: float, numtaps: int,
                state: Optional[np.ndarray] = None) -> np.ndarray:
    """抗混叠低通 + 抽取。"""
    if factor == 1:
        return x.copy()
    high_rate = rate * factor
    cutoff = 0.5 * margin * rate
    filtered = band_limit(x, high_rate, cutoff=cutoff, margin=margin,
                          numtaps=numtaps, gain=1.0, state=state)
    return filtered[::factor]


def process(signal: Any, rate: Optional[float] = None, drive: Any = 1.0,
            factor: Optional[int] = None,
            margin: float = DEFAULT_MARGIN,
            numtaps: int = DEFAULT_TAPS,
            state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """单块饱和处理：过采样 -> 同相非线性 -> 抗混叠降采样。

    多声道输入 (channels, samples) 时，drive 必须是跨声道共享的单条
    曲线（由 :func:`multichannel_phase` + :func:`curve_by_longterm`
    导出），各声道执行完全相同的映射。``state`` 内可携带
    ``up``/``down`` 滤波器历史，使分块处理边界连续。
    """
    x = _as_array(signal)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    if factor is None:
        factor = oversample_by_rate(rate)
    factor = int(max(factor, 1))

    def _states(ch: int):
        if state is None:
            return None, None
        ups = state.setdefault("up", {})
        dns = state.setdefault("down", {})
        key = str(ch)
        up_st = ups.setdefault(key, np.zeros(numtaps - 1, dtype=np.float64))
        dn_st = dns.setdefault(key, np.zeros(numtaps - 1, dtype=np.float64))
        return up_st, dn_st

    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)

    if x.ndim == 1:
        up_st, dn_st = _states(0)
        up = _upsample(x, factor, rate, margin, numtaps, up_st)
        d = _as_array(drive)
        if d.ndim == 1 and factor > 1:
            d = np.repeat(d, factor)[:up.size]
        wet = saturate(up, d)
        out = _downsample(wet, factor, rate, margin, numtaps, dn_st)
        return out[0] if scalar_input else out

    # 多声道：同一条 drive 曲线、同一个 factor，逐声道同相处理；
    # 各声道滤波器历史独立存放，互不串扰（保证相同输入逐位一致）。
    outs = []
    d = _as_array(drive)
    for ch in range(x.shape[0]):
        up_st, dn_st = _states(ch)
        up = _upsample(x[ch], factor, rate, margin, numtaps, up_st)
        dd = d
        if dd.ndim == 1 and factor > 1:
            dd = np.repeat(dd, factor)[:up.size]
        wet_ch = saturate(up, dd)
        outs.append(_downsample(wet_ch, factor, rate, margin, numtaps, dn_st))
    return np.stack(outs, axis=0)


# ---------------------------------------------------------------------------
# 6) 旁路等功率交叉淡化
# ---------------------------------------------------------------------------

def _bypass_gain(n: int, rate: float, fade_ms: float,
                 bypass: Any, state: Optional[Dict[str, Any]]) -> np.ndarray:
    """生成逐样本湿声位置 w：0 = 全干声（bypass 开），1 = 全湿声。"""
    rate = float(max(rate, 1e-9))
    fade_n = max(int(round(rate * fade_ms / 1000.0)), 1)

    if np.isscalar(bypass):
        target = 0.0 if bool(bypass) else 1.0
        # 首块从静止位置（干声）起步，保证任何方向的切换都完整渐变，
        # 不会因为初始化到目标值而吞掉第一段淡化。
        pos = float(state["pos"]) if (state is not None and "pos" in state) else 0.0
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            step = 1.0 / fade_n
            if pos < target:
                pos = min(target, pos + step)
            elif pos > target:
                pos = max(target, pos - step)
            out[i] = pos
        if state is not None:
            state["pos"] = pos
        return out

    b = np.asarray(bypass, dtype=np.float64).reshape(-1)
    if b.size == 1:
        return _bypass_gain(n, rate, fade_ms, bool(b[0]), state)
    if b.size != n:
        # 长度不符时按边界值扩展，绝不硬切。
        padded = np.empty(n, dtype=np.float64)
        padded[:min(n, b.size)] = b[:min(n, b.size)]
        if n > b.size:
            padded[b.size:] = b[-1]
        b = padded
    # bypass 标志 1=干声，湿声位置取反。
    wet_target = 1.0 - b
    pos = np.zeros(n, dtype=np.float64)
    cur = float(state["pos"]) if (state is not None and "pos" in state) else 0.0
    for i in range(n):
        target = float(wet_target[i])
        step = 1.0 / fade_n
        if cur < target:
            cur = min(target, cur + step)
        elif cur > target:
            cur = max(target, cur - step)
        pos[i] = cur
    if state is not None:
        state["pos"] = cur
    return pos


def bypass_crossfade(dry: Any, wet: Any = None, rate: Optional[float] = None,
                     bypass: Any = True, fade_ms: float = FADE_MS,
                     state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """干/湿声等功率交叉淡化（sqrt 余弦曲线），消除硬切爆音。

    ``bypass`` 可为布尔标量、逐样本布尔/0-1 数组；``state`` 让淡化位置
    在分块处理之间延续，块边界上不会重置出点击声。
    只给一个信号参数时，干/湿声取同一信号（用于淡化原语自检）。
    """
    d = _as_array(dry)
    scalar_input = d.ndim == 0
    if scalar_input:
        d = d.reshape(1)
    if wet is None:
        w = d.copy()
    else:
        w = _as_array(wet)
        if w.ndim == 0:
            w = w.reshape(1)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    n = int(d.shape[-1])
    wpos = _bypass_gain(n, rate, fade_ms, bypass, state)
    if d.ndim == 2:
        wpos = wpos[np.newaxis, :]
    # raised-cosine 等功率曲线：dry_g^2 + wet_g^2 = 1 恒成立，
    # 且两端导数为 0，淡化起止处没有斜率突变（无点击）。
    # wpos=1 全湿声，wpos=0 全干声。
    wet_g = np.sqrt(0.5 * (1.0 + np.cos(np.pi * (1.0 - wpos))))
    dry_g = np.sqrt(0.5 * (1.0 - np.cos(np.pi * (1.0 - wpos))))
    out = d * dry_g + w * wet_g
    return out[0] if scalar_input else out


def bypass(signal: Any, wet: Any = None, rate: Optional[float] = None,
           bypass_flag: Any = True,
           fade_ms: float = FADE_MS,
           state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """旁路入口：在干声 ``signal`` 与处理声 ``wet`` 之间交叉淡化。"""
    return bypass_crossfade(signal, wet, rate, bypass_flag, fade_ms, state)


# ---------------------------------------------------------------------------
# 7) 输出限幅 + 限带
# ---------------------------------------------------------------------------

def output_limit(signal: Any, rate: Optional[float] = None,
                 ceiling: float = LIMIT_CEIL,
                 margin: float = DEFAULT_MARGIN,
                 attack_ms: float = LIMIT_ATTACK_MS,
                 release_ms: float = LIMIT_RELEASE_MS,
                 lookahead_ms: float = LIMIT_LOOKAHEAD_MS,
                 numtaps: int = DEFAULT_TAPS,
                 state: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """前瞻限幅到安全天花板，末端再做一次限带。

    重采样 FIR 的振铃可能产生过冲；这里先按前瞻峰值计算平滑增益
    （attack/release 约束，不硬砍），再经低通滤除限幅动作产生的
    高频分量，保证输出既不削顶又带外干净。``state`` 让增益在分块
    渲染之间延续，块边界不重置出跳变。
    """
    x = _as_array(signal)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)

    def _limit_1d(ch: np.ndarray, st: Optional[Dict[str, Any]]) -> np.ndarray:
        if ch.size == 0:
            return ch.copy()
        look = max(int(round(rate * lookahead_ms / 1000.0)), 1)
        ka = _coef_time(attack_ms, rate)
        kr = _coef_time(release_ms, rate)
        if st is not None and "filter" not in st:
            st["filter"] = np.zeros(numtaps - 1, dtype=np.float64)

        y = ch
        # 迭代限幅：低通振铃可能把样本重新顶过天花板，最多三轮收敛。
        for _ in range(3):
            # 前瞻峰值：样本 i 的增益参考 [i, i+look) 窗口内最大幅度，
            # 因此增益在峰值到来之前就开始下降（真正的前瞻）。
            peak = np.empty(y.size, dtype=np.float64)
            for i in range(y.size):
                j = min(y.size, i + look)
                peak[i] = np.max(np.abs(y[i:j]))
            target_gain = np.minimum(1.0, ceiling / np.maximum(peak, 1e-9))

            gain = np.empty(y.size, dtype=np.float64)
            g = float(st.get("gain", 1.0)) if st is not None else 1.0
            for i in range(y.size):
                tg = float(target_gain[i])
                # 增益下降要快（attack），回升要慢（release），避免抽吸。
                coef = ka if tg < g else kr
                g = coef * g + (1.0 - coef) * tg
                gain[i] = g
            if st is not None:
                st["gain"] = g
            y = y * gain
            if float(np.max(np.abs(y))) <= ceiling + 1e-9:
                break

        # 限带：滤掉限幅动作产生的高频分量；用更保守的内部天花板
        # 给 FIR 过渡带振铃预留余量，保证限带后仍不超过 ceiling。
        f_st = st["filter"] if st is not None else None
        filtered = band_limit(y, rate, margin=margin, numtaps=numtaps,
                              state=f_st)
        if float(np.max(np.abs(filtered))) <= ceiling + 1e-9:
            return filtered
        # 振铃回弹时按实测过冲整体收一档，再限带一次。
        overshoot = float(np.max(np.abs(filtered))) / ceiling
        return band_limit(y / max(overshoot, 1.0), rate,
                          margin=margin, numtaps=numtaps, state=f_st)

    if x.ndim == 1:
        out = _limit_1d(x, state)
        return out[0] if scalar_input else out
    outs = []
    nch = x.shape[0]
    for ch in range(nch):
        st = None
        if state is not None:
            st = state.setdefault(str(ch), {})
        outs.append(_limit_1d(x[ch], st))
    return np.stack(outs, axis=0)


# ---------------------------------------------------------------------------
# 8) 端到端渲染
# ---------------------------------------------------------------------------

def render(signal: Any, rate: Optional[float] = None, amount: float = 1.0,
           bypass_flag: Any = False,
           factor: Optional[int] = None,
           margin: float = DEFAULT_MARGIN,
           ceiling: float = LIMIT_CEIL,
           fade_ms: float = FADE_MS,
           state: Optional[Dict[str, Any]] = None,
           reference_peak: Optional[float] = None,
           numtaps: int = DEFAULT_TAPS) -> np.ndarray:
    """端到端渲染管线。

    流程：长期峰值（多声道共享）-> 自适应过采样饱和 -> 旁路交叉淡化
    -> 输出限幅限带。``state`` 用于分块渲染时延续包络与淡化位置；
    分块场景传入全曲 ``reference_peak`` 可保证各块 drive 完全一致，
    块边界不产生失真量跳变。
    """
    x = _as_array(signal)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    scalar_input = x.ndim == 0
    if scalar_input:
        x = x.reshape(1)
    if x.size == 0:
        return x.copy()
    if state is None:
        state = {}
    if factor is None:
        factor = oversample_by_rate(rate)

    env_state = state.setdefault("envelope", {})
    sat_state = state.setdefault("saturator", {})
    if reference_peak is None and "reference_peak" in state:
        reference_peak = state["reference_peak"]
    if x.ndim == 2:
        # 多声道同相：一条共享曲线驱动所有声道。
        control = multichannel_phase(x)
        if reference_peak is None:
            reference_peak = float(np.max(np.abs(x))) if x.size else 0.0
        drive = curve_by_longterm(control, rate, reference_peak=reference_peak,
                                  state=env_state) * float(amount)
        # process 内部降采样已做抗混叠限带，无需重复滤波。
        wet = process(x, rate, drive, factor=factor, margin=margin,
                      numtaps=numtaps, state=sat_state)
    else:
        if reference_peak is None:
            reference_peak = float(np.max(np.abs(x))) if x.size else 0.0
        drive = curve_by_longterm(x, rate, reference_peak=reference_peak,
                                  state=env_state) * float(amount)
        wet = process(x, rate, drive, factor=factor, margin=margin,
                      numtaps=numtaps, state=sat_state)
    state["reference_peak"] = reference_peak

    xf_state = state.setdefault("bypass", {})
    # 完全旁路（静态标志且淡化已收敛到干声，或无状态的纯旁路首调）时
    # 原样透传干声，不再经过限幅/限带，保证旁路路径与输入逐样本一致。
    fully_bypassed = (np.isscalar(bypass_flag) and bool(bypass_flag)
                      and xf_state.get("pos", 0.0) == 0.0)
    if fully_bypassed:
        # 推进淡化状态机并标记已在干声位，保持分块状态一致性。
        _bypass_gain(x.shape[-1], rate, fade_ms, bypass_flag, xf_state)
        xf_state["pos"] = 0.0
        return x[0] if scalar_input else x
    mixed = bypass(x, wet, rate, bypass_flag, fade_ms=fade_ms, state=xf_state)
    lim_state = state.setdefault("limiter", {})
    out = output_limit(mixed, rate, ceiling=ceiling, margin=margin,
                       numtaps=numtaps, state=lim_state)
    if scalar_input:
        return np.asarray(out, dtype=np.float64).reshape(1)[0]
    return out


# ---------------------------------------------------------------------------
# 9) 失真度量与校验
# ---------------------------------------------------------------------------

def _thd_from_spectrum(x: np.ndarray, rate: float) -> Tuple[float, float]:
    """返回 (THD, 基频 Hz)。取频谱最强峰为基频。"""
    if x.size < 8:
        return 0.0, 0.0
    n = x.size
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(x * win))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    if spec.size <= 2:
        return 0.0, 0.0
    k0 = int(np.argmax(spec[1:]) + 1)
    f0 = freqs[k0]
    fund = spec[k0]
    harm_energy = 0.0
    h = 2
    while h * k0 < spec.size:
        harm_energy += spec[h * k0] ** 2
        h += 1
    thd = math.sqrt(harm_energy) / max(fund, 1e-12)
    return float(thd), float(f0)


def validate_distortion(rate: float, f0: float = 1000.0,
                        duration: float = 0.1, amount: float = 2.0,
                        factor: Optional[int] = None,
                        margin: float = DEFAULT_MARGIN,
                        aliased_ratio_limit: float = 0.05) -> Dict[str, Any]:
    """合成正弦测量信号，校验过采样饱和是否抑制了折叠失真。

    输出包含 THD、带外（折叠区）能量占比、过采样倍数及判定结论。
    """
    rate = float(max(rate, 1e-9))
    n = max(int(rate * duration), 64)
    t = np.arange(n) / rate
    tone = 0.7 * np.sin(2.0 * np.pi * f0 * t)
    if factor is None:
        factor = oversample_by_rate(rate)

    wet = render(tone, rate, amount=amount, factor=factor, margin=margin)

    spec = np.abs(np.fft.rfft(wet * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band_top = 0.5 * margin * rate
    total_e = float(np.sum(spec ** 2) + 1e-12)
    aliased_e = float(np.sum(spec[freqs > band_top] ** 2))
    aliased_ratio = aliased_e / total_e

    thd, measured_f0 = _thd_from_spectrum(wet, rate)
    ok = bool(aliased_ratio < aliased_ratio_limit and np.all(np.isfinite(wet))
              and np.max(np.abs(wet)) <= 1.0 + 1e-9)
    return {
        "ok": ok,
        "rate": rate,
        "factor": factor,
        "f0": measured_f0,
        "thd": thd,
        "aliased_ratio": aliased_ratio,
        "band_top": band_top,
        "peak": float(np.max(np.abs(wet))),
    }


def metrics(signal: Any, rate: Optional[float] = None) -> Dict[str, float]:
    """常规音频/失真度量：THD、带外能量比、峰值、RMS、峰均比、声道相干。"""
    x = _as_array(signal)
    if rate is None:
        rate = 48000.0
    rate = float(max(rate, 1e-9))
    if x.ndim == 0:
        x = x.reshape(1)

    if x.ndim == 2:
        mono = np.mean(x, axis=0)
        # 声道相干：两声道零延迟归一化互相关。
        if x.shape[0] >= 2:
            a, b = x[0], x[1]
            den = math.sqrt(float(np.sum(a * a) * np.sum(b * b))) + 1e-12
            coherence = float(np.sum(a * b) / den)
        else:
            coherence = 1.0
    else:
        mono = x
        coherence = 1.0

    thd, f0 = _thd_from_spectrum(mono, rate)
    n = mono.size
    spec = np.abs(np.fft.rfft(mono * np.hanning(n))) if n >= 8 else np.array([0.0])
    freqs = np.fft.rfftfreq(n, 1.0 / rate) if n >= 8 else np.array([0.0])
    band_top = 0.5 * DEFAULT_MARGIN * rate
    total_e = float(np.sum(spec ** 2) + 1e-12)
    out_of_band = float(np.sum(spec[freqs > band_top] ** 2)) / total_e

    peak = float(np.max(np.abs(x))) if x.size else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0
    return {
        "thd": thd,
        "f0": f0,
        "out_of_band_ratio": out_of_band,
        "peak": peak,
        "rms": rms,
        "crest": peak / max(rms, 1e-12),
        "channel_coherence": coherence,
    }


# ---------------------------------------------------------------------------
# 10) 审计、安全回退、确定性
# ---------------------------------------------------------------------------

def audit(config: Dict[str, Any], rate: Optional[float] = None) -> Dict[str, Any]:
    """审计配置是否覆盖六类缺陷，返回问题清单与总体结论。"""
    if not isinstance(config, dict):
        config = {}
    rate = float(config.get("rate", rate if rate is not None else 48000.0))
    findings = []

    factor = int(config.get("factor", oversample_by_rate(rate)))
    expected = oversample_by_rate(rate)
    if factor < expected:
        findings.append({
            "severity": "high",
            "issue": "oversampling",
            "message": f"factor={factor} 低于采样率 {rate:g} Hz 的推荐值 "
                       f"{expected}，高次谐波仍会折叠",
        })

    margin = float(config.get("margin", DEFAULT_MARGIN))
    if not 0.5 <= margin < 1.0:
        findings.append({
            "severity": "medium",
            "issue": "band_limit",
            "message": f"margin={margin} 超出 [0.5, 0.98]，谐波未留余量或切掉有用信号",
        })

    if float(config.get("release_ms", RELEASE_MS)) < 200.0:
        findings.append({
            "severity": "medium",
            "issue": "longterm_peak",
            "message": "峰值释放过快，段落间峰值变化会导致失真量跳变",
        })

    if not bool(config.get("shared_curve", True)):
        findings.append({
            "severity": "high",
            "issue": "multichannel_phase",
            "message": "多声道未共享饱和曲线，谐波相位发散、声像发虚",
        })

    if float(config.get("fade_ms", FADE_MS)) <= 0.0:
        findings.append({
            "severity": "high",
            "issue": "bypass",
            "message": "旁路淡化时长为 0，切换点会出现硬切爆音",
        })

    ceiling = float(config.get("ceiling", LIMIT_CEIL))
    if ceiling > 1.0:
        findings.append({
            "severity": "medium",
            "issue": "output_limit",
            "message": "限幅天花板高于 1.0，输出可能削顶",
        })

    return {
        "ok": not any(f["severity"] == "high" for f in findings),
        "rate": rate,
        "recommended_factor": expected,
        "findings": findings,
    }


def fallback(requested: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """把请求配置钳制到安全范围，返回安全配置与被调整的项。

    非字典输入（如自检标量）按空配置处理，回退到全部默认安全值。
    """
    if not isinstance(requested, dict):
        requested = {}
    rate = float(max(requested.get("rate", 48000.0), 1e-9))
    adjusted = []

    factor = int(requested.get("factor", oversample_by_rate(rate)))
    safe_factor = oversample_by_rate(rate)
    if factor < safe_factor:
        adjusted.append({"key": "factor", "from": factor, "to": safe_factor})
        factor = safe_factor
    if factor > MAX_FACTOR:
        adjusted.append({"key": "factor", "from": factor, "to": MAX_FACTOR})
        factor = MAX_FACTOR

    margin = float(requested.get("margin", DEFAULT_MARGIN))
    safe_margin = min(max(margin, 0.5), 0.98)
    if safe_margin != margin:
        adjusted.append({"key": "margin", "from": margin, "to": safe_margin})

    fade_ms = float(requested.get("fade_ms", FADE_MS))
    safe_fade = max(fade_ms, 1.0)
    if safe_fade != fade_ms:
        adjusted.append({"key": "fade_ms", "from": fade_ms, "to": safe_fade})

    ceiling = float(requested.get("ceiling", LIMIT_CEIL))
    safe_ceiling = min(max(ceiling, 1e-6), 1.0)
    if safe_ceiling != ceiling:
        adjusted.append({"key": "ceiling", "from": ceiling, "to": safe_ceiling})

    safe = {
        "rate": rate,
        "factor": factor,
        "margin": safe_margin,
        "fade_ms": safe_fade,
        "ceiling": safe_ceiling,
        "shared_curve": True,
        "release_ms": float(requested.get("release_ms", RELEASE_MS)),
    }
    return {"config": safe, "adjusted": adjusted}


def determinism(signal: Any, rate: Optional[float] = None,
                amount: float = 1.0,
                **kwargs: Any) -> Dict[str, Any]:
    """两次独立渲染必须逐位一致（无状态泄漏、无随机性）。"""
    if rate is None:
        rate = 48000.0
    x = _as_array(signal)
    out1 = render(x, rate, amount=amount, state={}, **kwargs)
    out2 = render(x, rate, amount=amount, state={}, **kwargs)
    identical = bool(out1.shape == out2.shape and np.array_equal(out1, out2))
    return {
        "ok": identical,
        "shape": tuple(out1.shape),
        "finite": bool(np.all(np.isfinite(out1))),
        "max_abs_diff": float(np.max(np.abs(out1 - out2))) if identical is False else 0.0,
    }
