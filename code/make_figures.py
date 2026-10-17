"""Result figures 2-6 from results/results.json (English and Chinese labels). No model is run."""
from __future__ import annotations

import json
import sys

import numpy as np
import matplotlib.pyplot as plt

import data as D
from bn5211_style import fs
from plot_helpers import plot_confusion

R = json.loads((D.RESULTS_DIR / "results.json").read_text(encoding="utf-8"))
M = R["models"]
CLS_COL = {"N": fs.PALETTE["teal"], "I": fs.PALETTE["ochre"], "E": fs.PALETTE["terra"]}

T = {
    "en": {
        "N": "Novice", "I": "Intermediate", "E": "Expert",
        "user": "Surgeon (user)", "dur": "Trial duration (s)",
        "time": "Time (s)", "speed": "Right-instrument speed (mm/s)",
        "mf1": "Pooled macro-F1", "delta": "Difference in macro-F1",
        "maj": "majority class", "pre": "pre-declared", "desc": "descriptive",
        "acc": "Trials correct", "probe": "User-identity probe\n(balanced accuracy)",
        "chance": "chance 1/8",
        "names": {"M0": "Majority", "M1": "K-simple", "M2a": "V-appear", "M2b": "V-dyn",
                  "M2c": "V-flow", "M3": "K + V-dyn\n(late, primary)", "M3b": "K + V-appear\n(late)",
                  "M3c": "K + V-flow\n(late)", "M3d": "K + V-dyn\n(early)",
                  "M4": "K-simple +\ntexture", "M5": "K-texture +\nV-dyn (late)",
                  "C1": "Duration\nonly", "C2": "K-simple,\npace removed",
                  "C2t": "K + texture,\npace removed"},
        "short": {"M0": "Majority", "M1": "K-simple", "M2a": "V-appear", "M2b": "V-dyn",
                  "M2c": "V-flow", "M3": "K + V-dyn (primary)", "M3b": "K + V-appear",
                  "M3c": "K + V-flow", "M3d": "K + V-dyn early", "M4": "K + texture",
                  "M5": "K-texture + V-dyn"},
        "cmp": {"M1_vs_M0": "K-simple vs majority", "M3_vs_M1": "K + V-dyn vs K-simple",
                "M4_vs_M1": "K + texture vs K-simple", "M3b_vs_M1": "K + V-appear vs K-simple",
                "M3c_vs_M1": "K + V-flow vs K-simple", "M3d_vs_M1": "K + V-dyn early vs K-simple",
                "M5_vs_M4": "K-texture + V-dyn vs K + texture"},
        "blocks": {"K_simple": "K-simple", "K_texture": "K-texture", "duration": "Duration",
                   "V_appear": "V-appear", "V_dyn": "V-dyn", "V_flow": "V-flow"},
        "share": "Share of true class",
        "nov_ex": "Novice trial {a}", "exp_ex": "Expert trial {b}",
        "prob": "K-simple class probability", "ptrue": "Probability of the true class",
        "true_row": "true", "right": "correct", "wrong": "wrong",
    },
    "zh": {
        "N": "新手", "I": "中级", "E": "专家",
        "user": "外科医生 (用户)", "dur": "试次时长 (s)",
        "time": "时间 (s)", "speed": "右侧器械速度 (mm/s)",
        "mf1": "合并 macro-F1", "delta": "macro-F1 差值",
        "maj": "多数类", "pre": "预先声明", "desc": "描述性",
        "acc": "正确试次数", "probe": "用户身份探针\n(平衡准确率)",
        "chance": "随机水平 1/8",
        "names": {"M0": "多数类", "M1": "K-simple", "M2a": "V-appear", "M2b": "V-dyn",
                  "M2c": "V-flow", "M3": "K + V-dyn\n(后期, 主模型)", "M3b": "K + V-appear\n(后期)",
                  "M3c": "K + V-flow\n(后期)", "M3d": "K + V-dyn\n(早期)",
                  "M4": "K-simple +\n纹理", "M5": "K-纹理 +\nV-dyn (后期)",
                  "C1": "仅时长", "C2": "K-simple,\n去除节奏",
                  "C2t": "K + 纹理,\n去除节奏"},
        "short": {"M0": "多数类", "M1": "K-simple", "M2a": "V-appear", "M2b": "V-dyn",
                  "M2c": "V-flow", "M3": "K + V-dyn (主模型)", "M3b": "K + V-appear",
                  "M3c": "K + V-flow", "M3d": "K + V-dyn 早期", "M4": "K + 纹理",
                  "M5": "K-纹理 + V-dyn"},
        "cmp": {"M1_vs_M0": "K-simple 对 多数类", "M3_vs_M1": "K + V-dyn 对 K-simple",
                "M4_vs_M1": "K + 纹理 对 K-simple", "M3b_vs_M1": "K + V-appear 对 K-simple",
                "M3c_vs_M1": "K + V-flow 对 K-simple", "M3d_vs_M1": "K + V-dyn 早期 对 K-simple",
                "M5_vs_M4": "K-纹理 + V-dyn 对 K + 纹理"},
        "blocks": {"K_simple": "K-simple", "K_texture": "K-纹理", "duration": "时长",
                   "V_appear": "V-appear", "V_dyn": "V-dyn", "V_flow": "V-flow"},
        "share": "占真实类别比例",
        "nov_ex": "新手试次 {a}", "exp_ex": "专家试次 {b}",
        "prob": "K-simple 类别概率", "ptrue": "真实类别的概率",
        "true_row": "真实", "right": "正确", "wrong": "错误",
    },
}


def setup(lang):
    fs.apply()
    if lang == "zh":
        plt.rcParams["font.family"] = ["IBM Plex Sans", "Microsoft YaHei"]
    return T[lang], ("" if lang == "en" else "_zh")


def fig2_data(lang):
    L, sfx = setup(lang)
    meta = D.load_meta()
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FIG_W_FULL, 2.2), width_ratios=[1, 1.25])
    users = sorted(meta.user.unique())
    rng = np.random.default_rng(0)
    for i, u in enumerate(users):
        sub = meta[meta.user == u]
        cls = sub.skill.iloc[0]
        dur = [D.load_kinematics(t).shape[0] / D.FPS for t in sub.trial_id]
        x = i + rng.uniform(-0.12, 0.12, len(dur))
        a.plot(x, dur, linestyle="none", marker=fs.MARKERS[D.CLASSES.index(cls)], markersize=4,
               color=CLS_COL[cls], markeredgecolor="white", markeredgewidth=0.3)
        a.plot([i - 0.25, i + 0.25], [np.median(dur)] * 2, color=fs.INK, lw=0.8)
    a.set_xticks(range(len(users)), users)
    a.set_xlabel(L["user"]); a.set_ylabel(L["dur"])
    handles = [plt.Line2D([], [], linestyle="none", marker=fs.MARKERS[j], color=CLS_COL[c],
                          markersize=4, label=L[c]) for j, c in enumerate(D.CLASSES)]
    a.legend(handles=handles, loc="upper left", ncol=3, columnspacing=0.8)
    fs.panel_tag(a, "a")
    # speed traces: one novice and one expert trial
    ex = {"N": "Suturing_B001", "E": "Suturing_D001"}
    for j, (c, t) in enumerate(ex.items()):
        X = D.load_kinematics(t)
        v = np.linalg.norm(D.manipulator(X, "SR")["vel"], axis=1) * 1000
        k = 15
        vs = np.convolve(v, np.ones(k) / k, mode="same")
        tt = np.arange(len(v)) / D.FPS
        lab = (L["nov_ex"] if c == "N" else L["exp_ex"]).format(a=t.split("_")[1], b=t.split("_")[1])
        b.plot(tt, vs, color=CLS_COL[c], lw=0.6, label=lab, linestyle=fs.DASHES[j * 1])
    b.set_xlabel(L["time"]); b.set_ylabel(L["speed"])
    b.set_xlim(0, None); b.set_ylim(0, None)
    b.legend(loc="upper right")
    fs.panel_tag(b, "b")
    fs.save(fig, "fig2_data" + sfx)


def fig3_results(lang):
    L, sfx = setup(lang)
    order = ["M0", "M1", "M2a", "M2b", "M2c", "M3", "M3b", "M3c", "M3d", "M4"]
    fam_col = {"M0": fs.MUTED, "M1": fs.PALETTE["teal"], "M4": fs.PALETTE["teal"],
               "M2a": fs.PALETTE["terra"], "M2b": fs.PALETTE["terra"], "M2c": fs.PALETTE["terra"],
               "M3": fs.PALETTE["indigo"], "M3b": fs.PALETTE["indigo"], "M3c": fs.PALETTE["indigo"],
               "M3d": fs.PALETTE["indigo"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FIG_W_FULL, 2.75), width_ratios=[1.55, 1])
    vals = [M[m]["macro_f1"] for m in order]
    lo = [M[m]["bootstrap_macro_f1"]["lo"] for m in order]
    hi = [M[m]["bootstrap_macro_f1"]["hi"] for m in order]
    x = np.arange(len(order))
    bars = a.bar(x, vals, width=0.66, color=[fam_col[m] for m in order])
    a.errorbar(x, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="none",
               ecolor=fs.AXIS, elinewidth=0.6, capsize=1.6)
    for xi, v in zip(x, vals):
        a.text(xi, 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=fs.FS_SMALL - 0.4,
               color="white", fontweight="semibold")
    a.axhline(M["M0"]["macro_f1"], color=fs.MUTED, lw=0.6, linestyle=(0, (4, 2)), zorder=0)
    a.set_xticks(x, [L["short"][m].replace(" (primary)", "*").replace(" (主模型)", "*") for m in order],
                 rotation=40, ha="right", rotation_mode="anchor", fontsize=fs.FS_SMALL)
    a.set_ylabel(L["mf1"]); a.set_ylim(0, 0.75)
    fs.panel_tag(a, "a")
    # paired differences
    cmp = ["M1_vs_M0", "M3_vs_M1", "M4_vs_M1", "M3b_vs_M1", "M3c_vs_M1", "M3d_vs_M1", "M5_vs_M4"]
    P = R["paired"]
    yy = np.arange(len(cmp))[::-1]
    for y, k in zip(yy, cmp):
        p = P[k]
        col = fs.PALETTE["indigo"] if p["pre_declared"] else fs.MUTED
        b.plot([p["lo"], p["hi"]], [y, y], color=col, lw=1.0)
        b.plot([p["point"]], [y], marker="o" if p["pre_declared"] else "s", color=col, markersize=3.8,
               markeredgecolor="white", markeredgewidth=0.3)
    b.axvline(0, color=fs.AXIS, lw=0.6)
    b.set_yticks(yy, [L["cmp"][k] for k in cmp], fontsize=fs.FS_SMALL)
    b.tick_params(axis="y", length=0)
    b.spines["left"].set_visible(False)
    b.set_xlabel(L["delta"])
    b.set_xlim(-0.25, 0.5)
    h = [plt.Line2D([], [], color=fs.PALETTE["indigo"], marker="o", markersize=3.8, lw=1, label=L["pre"]),
         plt.Line2D([], [], color=fs.MUTED, marker="s", markersize=3.8, lw=1, label=L["desc"])]
    b.legend(handles=h, loc="lower right")
    fs.panel_tag(b, "b")
    fs.save(fig, "fig3_results" + sfx)


def fig4_confusion(lang):
    L, sfx = setup(lang)
    mids = ["M1", "M3", "M4"]
    fig, axes = plt.subplots(1, 3, figsize=(fs.FIG_W_FULL, 2.05))
    names = [L[c] for c in D.CLASSES]
    for ax, m, tag in zip(axes, mids, "abc"):
        plot_confusion(fig, ax, M[m]["confusion_matrix"], names,
                       cbar_label=L["share"] if m == mids[-1] else None)
        ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")
        fs.note(ax, L["short"][m] + f"  (F1 {M[m]['macro_f1']:.2f})", x=0.0, y=1.02, va="bottom",
                color=fs.INK)
        fs.panel_tag(ax, tag)
        if m != "M1":
            ax.set_ylabel("")
    fs.save(fig, "fig4_confusion" + sfx)


def fig5_users(lang):
    L, sfx = setup(lang)
    mids = ["M0", "M1", "M2a", "M2b", "M2c", "M3", "M3c", "M4", "C1"]
    users = sorted(M["M1"]["per_user"].keys())
    mat = np.array([[M[m]["per_user"][u]["accuracy"] * M[m]["per_user"][u]["n"] for u in users] for m in mids])
    nmat = np.array([[M[m]["per_user"][u]["n"] for u in users] for m in mids])
    fig, ax = plt.subplots(figsize=(fs.FIG_W_FULL, 2.45))
    im = ax.imshow(mat / nmat, cmap="bn_seq", vmin=0, vmax=1, aspect="auto")
    for i in range(len(mids)):
        for j in range(len(users)):
            frac = mat[i, j] / nmat[i, j]
            ax.text(j, i, f"{int(mat[i, j])}/{nmat[i, j]}", ha="center", va="center", fontsize=fs.FS_SMALL,
                    color="white" if frac > 0.55 else fs.INK)
    cls = [M["M1"]["per_user"][u]["true_class"] for u in users]
    ax.set_xticks(range(len(users)), [f"{u}\n{L[c]}" for u, c in zip(users, cls)])
    for t, c in zip(ax.get_xticklabels(), cls):
        t.set_color(CLS_COL[c])
    ax.set_yticks(range(len(mids)), [L["short"].get(m, L["names"][m].replace("\n", " ")) for m in mids])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(users)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(mids)), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel(L["user"])
    fs.colorbar(fig, im, ax, L["acc"], ticks=[0, 0.5, 1.0])
    fs.save(fig, "fig5_users" + sfx)


def fig6_audit(lang):
    L, sfx = setup(lang)
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FIG_W_FULL, 2.2), width_ratios=[1, 1.1])
    blocks = ["K_simple", "K_texture", "duration", "V_appear", "V_dyn", "V_flow"]
    ip = R["identity_probe"]
    vals = [ip[k]["balanced_accuracy"] for k in blocks]
    cols = [fs.PALETTE["teal"]] * 3 + [fs.PALETTE["terra"]] * 3
    x = np.arange(len(blocks))
    bars = a.bar(x, vals, width=0.62, color=cols)
    fs.bar_labels(a, bars, fmt="{:.2f}")
    a.axhline(0.125, color=fs.MUTED, lw=0.6, linestyle=(0, (4, 2)), label=L["chance"])
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 1.08))
    a.set_xticks(x, [L["blocks"][k] for k in blocks], rotation=30, ha="right", rotation_mode="anchor")
    a.set_ylabel(L["probe"]); a.set_ylim(0, 1.1)
    fs.panel_tag(a, "a")
    mids = ["M1", "M4", "C1", "C2", "C2t"]
    vals = [M[m]["macro_f1"] for m in mids]
    lo = [M[m]["bootstrap_macro_f1"]["lo"] for m in mids]
    hi = [M[m]["bootstrap_macro_f1"]["hi"] for m in mids]
    cols = [fs.PALETTE["teal"], fs.PALETTE["teal"], fs.PALETTE["ochre"], fs.PALETTE["clay"], fs.PALETTE["clay"]]
    x = np.arange(len(mids))
    bars = b.bar(x, vals, width=0.62, color=cols)
    b.errorbar(x, vals, yerr=[np.subtract(vals, lo), np.subtract(hi, vals)], fmt="none", ecolor=fs.AXIS,
               elinewidth=0.6, capsize=1.6)
    for xi, v in zip(x, vals):
        b.text(xi, 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=fs.FS_SMALL - 0.4, color="white",
               fontweight="semibold")
    b.axhline(M["M0"]["macro_f1"], color=fs.MUTED, lw=0.6, linestyle=(0, (4, 2)), zorder=0, label=L["maj"])
    b.legend(loc="upper right", bbox_to_anchor=(1.0, 1.08))
    b.set_xticks(x, [L["names"][m] for m in mids], fontsize=fs.FS_SMALL)
    b.set_ylabel(L["mf1"]); b.set_ylim(0, 0.75)
    fs.panel_tag(b, "b")
    fs.save(fig, "fig6_audit" + sfx)


CASES = ["Suturing_G001", "Suturing_C004", "Suturing_E001", "Suturing_D005", "Suturing_I005"]


def fig7_cases(lang):
    """Trial-level K-simple predictions: probabilities per trial and the true-class probability vs duration."""
    import pandas as pd
    L, sfx = setup(lang)
    p = pd.read_csv(D.RESULTS_DIR / "predictions_M1.csv")
    dur = pd.read_csv(D.RESULTS_DIR / "features_K_simple.csv").set_index("trial_id")["duration_s"]
    p["dur"] = p.trial_id.map(dur)
    p["p_true"] = [r[f"p_{r.true}"] for _, r in p.iterrows()]
    fig, (a, b) = plt.subplots(1, 2, figsize=(fs.FIG_W_FULL, 2.45), width_ratios=[1.75, 1])
    # (a) stacked class probabilities, trials grouped by surgeon
    users = sorted(p.user.unique())
    xs, x, ticks = [], 0.0, []
    for u in users:
        n = int((p.user == u).sum())
        xs += [x + k for k in range(n)]
        ticks.append(x + (n - 1) / 2)
        x += n + 1.2
    xs = np.array(xs)
    bottom = np.zeros(len(p))
    for c in D.CLASSES:
        a.bar(xs, p[f"p_{c}"], bottom=bottom, width=0.82, color=CLS_COL[c], label=L[c], linewidth=0)
        bottom += p[f"p_{c}"].to_numpy()
    a.scatter(xs, np.full(len(p), -0.07), marker="s", s=7, c=[CLS_COL[c] for c in p.true],
              clip_on=False, linewidths=0)
    a.text(xs[0] - 0.9, -0.07, L["true_row"], ha="right", va="center", fontsize=fs.FS_SMALL, color=fs.MUTED)
    wrong = (p.pred != p.true).to_numpy()
    a.scatter(xs[wrong], np.full(wrong.sum(), 1.04), marker="v", s=9, color=fs.INK, clip_on=False,
              linewidths=0)
    a.set_xticks(ticks, users)
    for lab, u in zip(a.get_xticklabels(), users):
        lab.set_color(CLS_COL[p.true[p.user == u].iloc[0]])
    a.tick_params(axis="x", length=0, pad=12)
    a.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    a.set_ylim(0, 1)
    a.set_ylabel(L["prob"])
    a.set_xlabel(L["user"])
    a.legend(loc="upper center", bbox_to_anchor=(0.5, 1.17), ncol=3, columnspacing=0.9, handlelength=1.0)
    fs.panel_tag(a, "a")
    # (b) probability of the true class against trial duration
    for c in D.CLASSES:
        s = p[p.true == c]
        ok = s.pred == s.true
        b.scatter(s.dur[ok], s.p_true[ok], s=14, color=CLS_COL[c], marker=fs.MARKERS[D.CLASSES.index(c)],
                  edgecolors="white", linewidths=0.3, zorder=3)
        b.scatter(s.dur[~ok], s.p_true[~ok], s=14, facecolors="white", marker=fs.MARKERS[D.CLASSES.index(c)],
                  edgecolors=CLS_COL[c], linewidths=0.8, zorder=3)
    b.axhline(1 / 3, color=fs.MUTED, lw=0.6, linestyle=(0, (4, 2)), zorder=0)
    for t in CASES:
        r = p[p.trial_id == t].iloc[0]
        dx = -6 if r.dur > 200 else 5
        b.annotate(t.split("_")[1], (r.dur, r.p_true), xytext=(dx, 3), textcoords="offset points",
                   ha="right" if dx < 0 else "left", fontsize=fs.FS_SMALL - 0.4, color=fs.INK)
    b.set_xscale("log")
    b.set_xticks([60, 100, 200, 300], ["60", "100", "200", "300"])
    b.minorticks_off()
    b.set_xlabel(L["dur"]); b.set_ylabel(L["ptrue"]); b.set_ylim(0, 1.02)
    h = [plt.Line2D([], [], linestyle="none", marker="o", color=fs.MUTED, markersize=3.6, label=L["right"]),
         plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white", markeredgecolor=fs.MUTED,
                    markeredgewidth=0.8, markersize=3.6, label=L["wrong"])]
    b.legend(handles=h, loc="center right")
    fs.panel_tag(b, "b")
    fs.save(fig, "fig7_cases" + sfx)


if __name__ == "__main__":
    for lang in (sys.argv[1:] or ["en", "zh"]):
        for f in (fig2_data, fig3_results, fig4_confusion, fig5_users, fig6_audit, fig7_cases):
            f(lang)
    print("figures written")
