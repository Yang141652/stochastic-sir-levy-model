#!/usr/bin/env python3

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import SIRTS1_ref_thresholds as T
import jia2026joint_S33_E7_drivers as D


HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "numerics", "S35_SIR_paths")
os.makedirs(OUT, exist_ok=True)


# ============================================================
# 参数
# ============================================================
LAM, MU, EPS = 8.0, 0.75, 0.5
G1 = (-0.3, 0.5, 0.2)

PHI = np.array([
    [3, 2, 1],
    [2, 3, 2],
    [1, 2, 4]
]) * 1e-3

LAW = dict(
    alpha=0.4,
    b=1.0,
    zeta=(0.01, 0.3, 0.01),
    n=6,
    om=0.5,
    c=(1, 5, 5, 5, 1, 1)
)

P = T.setup(**LAW)

# 只使用 Brownian + 六方向 TS
# 在 simulate_paths() 中，最终更新为 dE + dJ
DRIVERS = ("TS6",)

NAMES = {
    "TS6": "Brownian + Tempered stable"
}

# TS1 的方向参数虽然不会被使用，但 build 函数需要传入
B = D.build(
    LAM,
    MU,
    EPS,
    G1,
    PHI,
    P,
    ts1_dir=1
)


# ============================================================
# 两个 beta 情形
# ============================================================
e316 = D.summary(
    B["laws"]["TS6"],
    0.316,
    LAM,
    B["a"],
    PHI
)

REGIMES = {
    # 疾病灭绝情形
    "band": {
        "beta": 0.232,
        "x0": (5.551, 0.5, 0.1),
        "T": 600.0
    },

    # 疾病持续情形
    "per": {
        "beta": 0.316,
        "x0": (
            e316["Sbar"],
            e316["Ibar"],
            e316["Rbar"]
        ),
        "T": 80.0
    }
}


# ============================================================
# 模拟 Brownian + TS6 路径
# ============================================================
def simulate_paths(
    beta,
    T_end,
    dt,
    npath,
    x0,
    driver,
    seed,
    rec_dt=0.25
):
    """
    模拟包含 Brownian 噪声和六方向 TS 噪声的 SIR 系统。
    """

    D.SIM.rng = np.random.default_rng(seed)

    Gm = np.linalg.cholesky(PHI)

    # driver 为 TS6
    m1, m2, m3 = B["mu_x"][driver]

    nstep = int(round(T_end / dt))
    rec_every = int(round(rec_dt / dt))
    nrec = nstep // rec_every + 1

    # 对数坐标
    Y = np.tile(np.log(x0), (npath, 1))

    rec_t = np.zeros(nrec)

    rec = {
        "S": np.empty((nrec, npath)),
        "I": np.empty((nrec, npath)),
        "R": np.empty((nrec, npath))
    }

    # 初值
    rec["S"][0] = np.exp(Y[:, 0])
    rec["I"][0] = np.exp(Y[:, 1])
    rec["R"][0] = np.exp(Y[:, 2])

    jrec = 1

    for k in range(nstep):

        # Brownian 增量
        dE = (
            np.sqrt(dt)
            * D.SIM.rng.standard_normal((npath, 3))
            @ Gm.T
        )

        # 六方向 TS 增量
        dJ = B["jumps"][driver](dt, npath)

        S = np.exp(Y[:, 0])
        I = np.exp(Y[:, 1])
        R = np.exp(Y[:, 2])

        drift = np.stack([
            LAM / S - m1 - beta * I - PHI[0, 0] / 2,
            beta * S - m2 - PHI[1, 1] / 2,
            I / R - m3 - PHI[2, 2] / 2
        ], axis=1)

        # Brownian + TS 同时作用
        Y = Y + drift * dt + dE + dJ

        # 保存路径
        if (k + 1) % rec_every == 0:

            t = (k + 1) * dt
            rec_t[jrec] = t

            S = np.exp(Y[:, 0])
            I = np.exp(Y[:, 1])
            R = np.exp(Y[:, 2])

            rec["S"][jrec] = np.maximum(S, 1e-300)
            rec["I"][jrec] = np.maximum(I, 1e-300)
            rec["R"][jrec] = np.maximum(R, 1e-300)

            jrec += 1

    return rec_t, rec


# ============================================================
# 累计长时间平均
# ============================================================
def running_average(X, t):
    """
    X.shape = (n_time, n_path)

    返回每条路径的累计时间平均：
        1/t * integral_0^t X(s) ds
    """

    avg = np.empty_like(X, dtype=float)

    # t=0 时使用初始值
    avg[0] = X[0]

    dt = np.diff(t)

    integral = np.zeros_like(X, dtype=float)

    integral[1:] = np.cumsum(
        0.5 * (X[1:] + X[:-1]) * dt[:, None],
        axis=0
    )

    avg[1:] = integral[1:] / t[1:, None]

    return avg


def quantiles(X):
    """
    返回 5%、25%、50%、75%、95% 分位数。
    """
    return np.percentile(
        X,
        [5, 25, 50, 75, 95],
        axis=1
    )


# ============================================================
# 模拟并计算长时间平均
# ============================================================
DATA_AVG = {}

for reg, cfg in REGIMES.items():

    DATA_AVG[reg] = {}

    for j, driver in enumerate(DRIVERS):

        # 两个 beta 情形使用不同随机种子
        seed = 35100 + 100 * j + (
            0 if reg == "band" else 10
        )

        t, rec = simulate_paths(
            beta=cfg["beta"],
            T_end=cfg["T"],
            dt=0.005,
            npath=1000,
            x0=cfg["x0"],
            driver=driver,
            seed=seed,
            rec_dt=0.25
        )

        # 计算 S、I、R 的累计时间平均
        S_avg = running_average(rec["S"], t)
        I_avg = running_average(rec["I"], t)
        R_avg = running_average(rec["R"], t)

        DATA_AVG[reg][driver] = {
            "t": t,
            "S": quantiles(S_avg),
            "I": quantiles(I_avg),
            "R": quantiles(R_avg)
        }


# ============================================================
# 理论长期平均值
# ============================================================
THEORY = {}

for reg, cfg in REGIMES.items():

    THEORY[reg] = {}

    for driver in DRIVERS:

        if reg == "band":
            # 疾病灭绝态
            S_theory = LAM / B["a"][0]
            I_theory = 0.0
            R_theory = 0.0

        else:
            # 疾病持续态
            sm = D.summary(
                B["laws"][driver],
                cfg["beta"],
                LAM,
                B["a"],
                PHI
            )

            S_theory = sm["Sbar"]
            I_theory = sm["Ibar"]
            R_theory = sm["Rbar"]

        THEORY[reg][driver] = {
            "S": S_theory,
            "I": I_theory,
            "R": R_theory
        }


# ============================================================
# 绘图
# ============================================================
fig, ax = plt.subplots(
    3,
    2,
    figsize=(6.4, 6.0),
    sharex="col"
)

variables = [
    ("S", r"$\frac{1}{t}\int_0^t S(s)\,ds$"),
    ("I", r"$\frac{1}{t}\int_0^t I(s)\,ds$"),
    ("R", r"$\frac{1}{t}\int_0^t R(s)\,ds$")
]

reg_order = ["band", "per"]

# 只保留一条 Brownian + TS 曲线
color = "#0072B2"

for col, reg in enumerate(reg_order):

    beta = REGIMES[reg]["beta"]
    driver = "TS6"

    for row, (var, ylabel) in enumerate(variables):

        A = ax[row, col]

        t = DATA_AVG[reg][driver]["t"]
        q = DATA_AVG[reg][driver][var]

        # 90% 区间
        A.fill_between(
            t,
            q[0],
            q[4],
            color=color,
            alpha=0.12,
            linewidth=0
        )

        # 50% 区间
        A.fill_between(
            t,
            q[1],
            q[3],
            color=color,
            alpha=0.28,
            linewidth=0
        )

        # 中位数
        A.plot(
            t,
            q[2],
            color=color,
            linewidth=1.2,
            label=NAMES[driver]
        )

        # 理论地方病平均值
        theory = THEORY[reg][driver][var]

        if theory > 0:
            A.axhline(
                theory,
                color="black",
                linestyle="--",
                linewidth=0.8,
                alpha=0.9
            )

        # 左侧 I、R 图添加准灭绝水平线

        # 左列 t=0--600，右列 t=0--100
        if col == 0:
            A.set_xlim(0.0, 600.0)
        else:
            A.set_xlim(0.0, 80.0)

        # 纵坐标设置
        # 左侧 I、R 使用压缩后的对数坐标
        if col == 0 and var in ("I", "R"):

            A.set_yscale(
                "symlog",
                linthresh=1e-1,
                linscale=0.25,
                base=10
            )

            # 只保留少量刻度
            A.set_yticks([
                0.0,
                1e-1,
                1.0
            ])

            A.set_yticklabels([
                "0",
                r"$10^{-1}$",
                r"$10^{0}$"
            ])

            A.set_ylim(0.0, 1.0)

        else:
            A.set_yscale("log")
        A.grid(alpha=0.2, which="both")
        A.tick_params(labelsize=8)

        if var == "S":
            A.yaxis.get_offset_text().set_visible(False)

        if col == 0:
            A.set_ylabel(ylabel, fontsize=10)

        if row == 0:
            A.set_title(
                r"$\beta=%.3f$" % beta,
                fontsize=10
            )

        if row == 2:
            A.set_xlabel(r"$t$", fontsize=10)

            if col == 0:
                A.set_xticks([0, 100, 200, 300, 400, 500, 600])
            else:
                A.set_xticks([0, 20, 40, 60, 80])


# 只显示一个图例：Brownian + TS
handles, labels = ax[0, 0].get_legend_handles_labels()

fig.legend(
    handles[:1],
    labels[:1],
    loc="lower center",
    bbox_to_anchor=(0.5, 0.015),
    ncol=1,
    frameon=False,
    fontsize=8
)

fig.tight_layout(
    rect=(0, 0.10, 1, 1)
)

outfile = os.path.join(
    OUT,
    "jia2026joint_F5_long_time_averages_Brownian_TS.pdf"
)

fig.savefig(
    outfile,
    metadata={"CreationDate": None}
)

plt.close(fig)

print("written:", outfile)