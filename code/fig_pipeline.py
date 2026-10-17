"""Figure 1: workflow schematic (English and Chinese labels)."""
from __future__ import annotations

import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from bn5211_style import fs, FIG_DIR

LABELS = {
    "en": {
        "kin": "Kinematics\n76 channels, 30 Hz",
        "vid": "Endoscope video\nleft view, 30 fps",
        "ks": "K-simple (30)\nduration, path length,\nvelocity, gripper",
        "kt": "K-texture (10)\napproximate entropy,\njerk, spectral entropy",
        "va": "V-appear (1024)\nResNet-18, 1 Hz,\nmean + SD",
        "vd": "V-dyn (8)\nframe-to-frame\nembedding change",
        "vf": "V-flow (15)\nFarneback flow,\n2 Hz frame pairs",
        "clf": "Standardise (+PCA)\nL2 logistic regression\nC chosen by inner LOUO",
        "fuse": "Late fusion\nmean of class\nprobabilities",
        "out": "Skill label\nN / I / E",
        "louo": "Outer loop: 8 official leave-one-user-out folds; inner loop over the 7 training users.\n"
                "The held-out surgeon never enters scaling, PCA or model selection.",
        "imp": "focused improvement",
    },
    "zh": {
        "kin": "运动学数据\n76 通道, 30 Hz",
        "vid": "内窥镜视频\n左视图, 30 fps",
        "ks": "K-simple (30)\n时长、路径长度、\n速度、夹钳动作",
        "kt": "K-texture (10)\n近似熵、加加速度、\n频谱熵",
        "va": "V-appear (1024)\nResNet-18, 1 Hz,\n均值 + 标准差",
        "vd": "V-dyn (8)\n相邻帧嵌入\n变化统计",
        "vf": "V-flow (15)\nFarneback 光流,\n2 Hz 帧对",
        "clf": "标准化 (+PCA)\nL2 逻辑回归\nC 由内层 LOUO 选择",
        "fuse": "后期融合\n类别概率\n取平均",
        "out": "技能标签\nN / I / E",
        "louo": "外层: 8 个官方留一用户折; 内层: 在 7 位训练用户上再做留一用户。\n"
                "被留出的外科医生不参与标准化、PCA 和模型选择。",
        "imp": "针对性改进",
    },
}


def box(ax, x, y, w, h, text, color, fill=None, weight="normal", fontsize=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                       linewidth=0.8, edgecolor=color, facecolor=fill or "white",
                       transform=ax.transAxes, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=fontsize or fs.FS_SMALL, color=fs.INK, linespacing=1.15,
            fontweight=weight, zorder=3)
    return (x, y, w, h)


def arrow(ax, a, b, color=fs.AXIS, dashed=False):
    ax.annotate("", xy=b, xytext=a, xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=color, lw=0.7, shrinkA=0, shrinkB=0,
                                mutation_scale=6, linestyle=(0, (3, 2)) if dashed else "-"),
                zorder=1)


def draw(lang: str) -> None:
    L = LABELS[lang]
    fs.apply()
    if lang == "zh":
        plt.rcParams["font.family"] = ["IBM Plex Sans", "Microsoft YaHei"]
    fig = plt.figure(figsize=(fs.FIG_W_FULL, 2.55), layout="none")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    teal, terra, indigo, ochre = (fs.PALETTE[k] for k in ("teal", "terra", "indigo", "ochre"))
    wash_t, wash_v = "#EAF2F2", "#F8ECE8"

    # inputs
    kin = box(ax, 0.01, 0.62, 0.13, 0.20, L["kin"], teal, wash_t)
    vid = box(ax, 0.01, 0.22, 0.13, 0.20, L["vid"], terra, wash_v)
    # feature blocks
    fx, fw, fh = 0.20, 0.17, 0.155
    ks = box(ax, fx, 0.80, fw, fh, L["ks"], teal)
    kt = box(ax, fx, 0.605, fw, fh, L["kt"], ochre)
    va = box(ax, fx, 0.40, fw, fh, L["va"], terra)
    vd = box(ax, fx, 0.20, fw, fh, L["vd"], terra)
    vf = box(ax, fx, 0.00 + 0.005, fw, fh, L["vf"], terra)
    for b in (ks, kt):
        arrow(ax, (kin[0] + kin[2], kin[1] + kin[3] / 2), (b[0], b[1] + b[3] / 2))
    for b in (va, vd, vf):
        arrow(ax, (vid[0] + vid[2], vid[1] + vid[3] / 2), (b[0], b[1] + b[3] / 2))
    # improvement marker
    ax.text(fx + 0.004, kt[1] - 0.008, "▲ " + L["imp"], transform=ax.transAxes, ha="left",
            va="top", fontsize=5.4, color=ochre)
    # classifiers
    cx, cw = 0.47, 0.19
    ck = box(ax, cx, 0.62, cw, 0.22, L["clf"], indigo)
    cv = box(ax, cx, 0.16, cw, 0.22, L["clf"], indigo)
    for b in (ks, kt):
        arrow(ax, (b[0] + b[2], b[1] + b[3] / 2), (ck[0], ck[1] + ck[3] / 2))
    for b in (va, vd, vf):
        arrow(ax, (b[0] + b[2], b[1] + b[3] / 2), (cv[0], cv[1] + cv[3] / 2))
    # fusion and output
    fu = box(ax, 0.72, 0.39, 0.12, 0.22, L["fuse"], fs.INK, "#F3F4F6")
    arrow(ax, (ck[0] + ck[2], ck[1] + ck[3] / 2), (fu[0], fu[1] + fu[3] * 0.75))
    arrow(ax, (cv[0] + cv[2], cv[1] + cv[3] / 2), (fu[0], fu[1] + fu[3] * 0.25))
    out = box(ax, 0.885, 0.42, 0.105, 0.16, L["out"], fs.INK, "#F3F4F6", weight="semibold")
    arrow(ax, (fu[0] + fu[2], fu[1] + fu[3] / 2), (out[0], out[1] + out[3] / 2))
    # LOUO banner
    ax.text(0.47, 0.035, L["louo"], transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.6, color=fs.MUTED, linespacing=1.3)
    fs.save(fig, "fig1_pipeline" + ("" if lang == "en" else "_zh"))


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["en", "zh"]):
        draw(lang)
    print("saved to", FIG_DIR)
