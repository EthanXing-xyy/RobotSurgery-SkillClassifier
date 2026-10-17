"""Run every model of PROTOCOL.md once and write results/*.json and predictions CSVs.

Usage:  python code/run_all.py
"""
from __future__ import annotations

import json
import platform
import sys
import time

import numpy as np
import pandas as pd

import data as D
import evaluate as E
import features_kin as FK
import features_video as FV


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ----------------------------------------------------------------------
# features
# ----------------------------------------------------------------------

def build_features(meta: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cache = D.CACHE_DIR / f"features_{FK.VERSION}_{FV.VERSION}.pkl"
    if cache.exists():
        log(f"features from cache {cache.name}")
        return pd.read_pickle(cache)
    simple, texture, appear, dyn, flow, vinfo = [], [], [], [], [], []
    for i, t in enumerate(meta["trial_id"]):
        X = D.load_kinematics(t)
        simple.append(FK.simple_features(X))
        texture.append(FK.texture_features(X))
        v = FV.trial_features(t)
        appear.append(v["appear"])
        dyn.append(v["dyn"])
        flow.append(v["flow"])
        vinfo.append({"kin_frames": X.shape[0], "video_frames_1hz": v["n_frames"],
                      "flow_pairs_2hz": v["n_pairs"]})
        log(f"[{i + 1:2d}/{len(meta)}] {t}: {X.shape[0]} kin frames, "
            f"{v['n_frames']} video frames, {v['n_pairs']} flow pairs")
    idx = meta["trial_id"]
    feats = {
        "K_simple": pd.DataFrame(simple, index=idx),
        "K_texture": pd.DataFrame(texture, index=idx),
        "V_appear": pd.DataFrame(np.stack(appear), index=idx,
                                 columns=[f"emb_mean_{j}" for j in range(512)] +
                                         [f"emb_sd_{j}" for j in range(512)]),
        "V_dyn": pd.DataFrame(dyn, index=idx),
        "V_flow": pd.DataFrame(flow, index=idx),
        "info": pd.DataFrame(vinfo, index=idx),
    }
    for k, df in feats.items():
        assert not df.isna().any().any(), f"NaN in feature block {k}"
    pd.to_pickle(feats, cache)
    return feats


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main() -> None:
    D.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = D.load_meta()
    y = D.y_int(meta)
    users = meta["user"].to_numpy()
    folds = D.louo_folds(meta)
    log(f"{len(meta)} trials, classes {dict(meta['skill'].value_counts())}, "
        f"{len(folds)} official LOUO folds")

    F = build_features(meta)
    Ks, Kt = F["K_simple"].to_numpy(), F["K_texture"].to_numpy()
    Va, Vd, Vf = F["V_appear"].to_numpy(), F["V_dyn"].to_numpy(), F["V_flow"].to_numpy()
    dur = F["K_simple"]["duration_s"].to_numpy()
    no_dur_cols = [c for c in F["K_simple"].columns if c != "duration_s"]
    Ks_nodur = F["K_simple"][no_dur_cols].to_numpy()

    runs: dict[str, dict] = {}

    def run(mid: str, desc: str, out: dict) -> None:
        out["desc"] = desc
        runs[mid] = out
        m = E.metrics(y, out["pred"], users, folds)
        log(f"{mid:4s} macro-F1 {m['macro_f1']:.3f}  acc {m['accuracy']:.3f}  ({desc})")

    log("M0 majority")
    run("M0", "Majority class", E.majority_oof(y, users, folds))
    log("M1 kinematics simple")
    run("M1", "Kinematics: simple statistics", E.louo_oof(Ks, y, users, folds))
    log("M2a-c video")
    run("M2a", "Video: appearance (ResNet-18)", E.louo_oof(Va, y, users, folds, use_pca=True))
    run("M2b", "Video: scene-change dynamics", E.louo_oof(Vd, y, users, folds))
    run("M2c", "Video: optical flow", E.louo_oof(Vf, y, users, folds))
    log("M3 fusion")
    run("M3", "Late fusion K-simple + V-dyn (primary)",
        E.late_fuse(runs["M1"]["proba"], runs["M2b"]["proba"]))
    run("M3b", "Late fusion K-simple + V-appear",
        E.late_fuse(runs["M1"]["proba"], runs["M2a"]["proba"]))
    run("M3c", "Late fusion K-simple + V-flow",
        E.late_fuse(runs["M1"]["proba"], runs["M2c"]["proba"]))
    run("M3d", "Early fusion K-simple + V-dyn",
        E.louo_oof(np.hstack([Ks, Vd]), y, users, folds))
    log("M4/M5 improvement")
    run("M4", "Kinematics: simple + motion texture", E.louo_oof(np.hstack([Ks, Kt]), y, users, folds))
    run("M5", "Late fusion K-texture + V-dyn",
        E.late_fuse(runs["M4"]["proba"], runs["M2b"]["proba"]))
    log("controls")
    run("C1", "Control: duration only", E.louo_oof(dur.reshape(-1, 1), y, users, folds))
    run("C2", "Control: K-simple residualised on log duration",
        E.louo_oof_with_index_preprocess(Ks_nodur, y, users, folds, E.residualise_on(dur)))
    run("C2t", "Control: K-simple + texture residualised on log duration",
        E.louo_oof_with_index_preprocess(np.hstack([Ks_nodur, Kt]), y, users, folds,
                                         E.residualise_on(dur)))

    # ---------------- metrics, uncertainty, permutation ----------------
    summary = {}
    for mid, out in runs.items():
        m = E.metrics(y, out["pred"], users, folds)
        m["bootstrap_macro_f1"] = E.user_bootstrap(y, out["pred"], users)
        m["permutation"] = E.exact_user_label_permutation(y, out["pred"], users)
        m["desc"] = out["desc"]
        m["chosen"] = out["chosen"]
        m["inner_folds_skipped"] = {u: v["inner_folds_skipped"] for u, v in out["inner"].items()}
        summary[mid] = m
        pd.DataFrame({
            "trial_id": meta["trial_id"], "user": users, "true": [D.CLASSES[i] for i in y],
            "pred": [D.CLASSES[i] for i in out["pred"]],
            **{f"p_{c}": out["proba"][:, j] for j, c in enumerate(D.CLASSES)},
        }).to_csv(D.RESULTS_DIR / f"predictions_{mid}.csv", index=False)

    comparisons = {
        "M1_vs_M0": ("M1", "M0"), "M3_vs_M1": ("M3", "M1"), "M4_vs_M1": ("M4", "M1"),
        # descriptive (not pre-declared): every fusion and video block against kinematics
        "M3b_vs_M1": ("M3b", "M1"), "M3c_vs_M1": ("M3c", "M1"), "M3d_vs_M1": ("M3d", "M1"),
        "M5_vs_M4": ("M5", "M4"), "M2b_vs_M0": ("M2b", "M0"),
    }
    paired = {k: {"a": a, "b": b, "pre_declared": k in ("M1_vs_M0", "M3_vs_M1", "M4_vs_M1"),
                  **E.paired_user_bootstrap(y, runs[a]["pred"], runs[b]["pred"], users)}
              for k, (a, b) in comparisons.items()}

    # ---------------- shortcut audit: identity probe ----------------
    blocks = {"K_simple": (Ks, False), "K_texture": (Kt, False), "duration": (dur.reshape(-1, 1), False),
              "V_appear": (Va, True), "V_dyn": (Vd, False), "V_flow": (Vf, False)}
    id_probe = {k: E.identity_probe(X, users, use_pca=p) for k, (X, p) in blocks.items()}
    for k, v in id_probe.items():
        log(f"identity probe {k:10s} balanced acc {v['balanced_accuracy']:.3f} (chance {v['chance']:.3f})")

    # correlation of each block's features with log duration (pace content)
    logd = np.log(dur)
    pace = {}
    for k in ["K_simple", "K_texture", "V_dyn", "V_flow"]:
        df = F[k]
        r = [abs(np.corrcoef(df[c].to_numpy(), logd)[0, 1]) for c in df.columns if c != "duration_s"]
        pace[k] = {"median_abs_r_logdur": float(np.median(r)), "max_abs_r_logdur": float(np.max(r))}

    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import sklearn, torch, torchvision, cv2, scipy
        env.update({"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "opencv": cv2.__version__,
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"})
    except Exception as exc:          # pragma: no cover
        env["error"] = repr(exc)

    dataset = {
        "n_trials": int(len(meta)),
        "class_counts": {c: int((meta["skill"] == c).sum()) for c in D.CLASSES},
        "users": {u: {"label": meta.loc[meta.user == u, "skill"].iloc[0],
                      "n_trials": int((meta.user == u).sum())} for u in sorted(set(users))},
        "folds": [{"held_out": u, "n_train": int(len(tr)), "n_test": int(len(te))} for u, tr, te in folds],
        "duration_s": {"min": float(dur.min()), "max": float(dur.max()), "median": float(np.median(dur))},
        "duration_by_class": {c: float(np.median(dur[y == i])) for i, c in enumerate(D.CLASSES)},
        "feature_dims": {k: int(F[k].shape[1]) for k in ["K_simple", "K_texture", "V_appear", "V_dyn", "V_flow"]},
        "video_frames_1hz_total": int(F["info"]["video_frames_1hz"].sum()),
        "flow_pairs_2hz_total": int(F["info"]["flow_pairs_2hz"].sum()),
    }

    out = {"models": summary, "paired": paired, "identity_probe": id_probe, "pace": pace,
           "dataset": dataset, "environment": env,
           "feature_names": {k: list(F[k].columns) for k in ["K_simple", "K_texture", "V_dyn", "V_flow"]}}
    (D.RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, (F_, X_) in {"K_simple": (F["K_simple"], None), "K_texture": (F["K_texture"], None),
                        "V_dyn": (F["V_dyn"], None), "V_flow": (F["V_flow"], None)}.items():
        F_.assign(user=users, skill=meta["skill"].to_numpy()).to_csv(D.RESULTS_DIR / f"features_{k}.csv")
    log("wrote results/results.json")
    for k, v in paired.items():
        log(f"{k:10s} delta {v['point']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]"
            f"{'  (pre-declared)' if v['pre_declared'] else ''}")


if __name__ == "__main__":
    main()
