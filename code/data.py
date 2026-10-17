"""Dataset access for JIGSAWS Suturing skill classification (read-only raw data)."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data locations. Override with the environment variables JIGSAWS_SUTURING (the Suturing folder
# of the JIGSAWS release) and JIGSAWS_SPLITS (the Experimental_setup folder of the official splits).
DATA_ROOT = Path(os.environ.get("JIGSAWS_SUTURING", PROJECT_ROOT / "data" / "Suturing"))
SPLITS_ROOT = Path(os.environ.get("JIGSAWS_SPLITS", PROJECT_ROOT / "data" / "Experimental_setup"))
KIN_DIR = DATA_ROOT / "kinematics" / "AllGestures"
VIDEO_DIR = DATA_ROOT / "video"
META_FILE = DATA_ROOT / "meta_file_Suturing.txt"

CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

CLASSES = ["N", "I", "E"]            # novice, intermediate, expert
CLASS_NAMES = {"N": "Novice", "I": "Intermediate", "E": "Expert"}
FPS = 30.0

# 0-indexed column offsets of the four manipulators in the 76-column kinematics file.
# Within a manipulator: position 0:3, rotation matrix 3:12, translational velocity 12:15,
# rotational velocity 15:18, gripper angle 18.
MANIPULATORS = {"ML": 0, "MR": 19, "SL": 38, "SR": 57}
SLAVES = ["SL", "SR"]                # patient-side instruments that act on the tissue
MASTERS = ["ML", "MR"]               # surgeon-side teleoperation inputs


def load_meta() -> pd.DataFrame:
    """One row per trial: trial_id, user, skill, GRS total and the six GRS components."""
    rows = []
    for line in META_FILE.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        trial_id = parts[0]
        rows.append({
            "trial_id": trial_id,
            "user": trial_id.split("_")[1][0],
            "trial_no": int(trial_id.split("_")[1][1:]),
            "skill": parts[1],
            "grs_total": int(parts[2]),
        })
    df = pd.DataFrame(rows).sort_values(["user", "trial_no"]).reset_index(drop=True)
    assert df["skill"].isin(CLASSES).all(), "unexpected skill label"
    return df


def load_kinematics(trial_id: str) -> np.ndarray:
    """(T, 76) float array at 30 Hz."""
    return np.loadtxt(KIN_DIR / f"{trial_id}.txt", dtype=np.float64)


def manipulator(X: np.ndarray, name: str) -> dict[str, np.ndarray]:
    """Split one manipulator's channels out of the 76-column array."""
    b = MANIPULATORS[name]
    return {"pos": X[:, b:b + 3], "rot": X[:, b + 3:b + 12],
            "vel": X[:, b + 12:b + 15], "rotvel": X[:, b + 15:b + 18],
            "grip": X[:, b + 18]}


def video_path(trial_id: str, capture: int = 1) -> Path:
    """capture1 is the left endoscope view, capture2 the right one."""
    return VIDEO_DIR / f"{trial_id}_capture{capture}.avi"


OFFICIAL_LOUO_DIR = SPLITS_ROOT / "Suturing" / "unBalanced" / "SkillDetection" / "UserOut"


def official_louo_test_trials() -> dict[str, list[str]]:
    """Test trial ids of the 8 official SkillDetection/UserOut folds, keyed by fold name."""
    assert OFFICIAL_LOUO_DIR.is_dir(), f"official split folder not found: {OFFICIAL_LOUO_DIR} (set JIGSAWS_SPLITS)"
    out = {}
    for d in sorted(OFFICIAL_LOUO_DIR.glob("*_Out"), key=lambda p: int(p.name.split("_")[0])):
        lines = (d / "itr_1" / "Test.txt").read_text(encoding="utf-8").splitlines()
        trials = []
        for line in lines:
            tok = line.split()
            if not tok:
                continue
            parts = tok[0].split("_")
            trials.append(f"{parts[0]}_{parts[1]}")     # Suturing_B001_000080_005635.txt -> Suturing_B001
        out[d.name] = trials
    return out


def louo_folds(meta: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The official leave-one-user-out folds as (held-out user, train rows, test rows).

    Each official fold holds out exactly one user, which is checked here; the fold is then
    named after that user so that results can be read per surgeon.
    """
    index = {t: i for i, t in enumerate(meta["trial_id"])}
    folds = []
    for fold_name, trials in official_louo_test_trials().items():
        missing = [t for t in trials if t not in index]
        assert not missing, f"official fold {fold_name} lists unknown trials {missing}"
        test = np.array([index[t] for t in trials], dtype=int)
        users = sorted({meta["user"].iloc[i] for i in test})
        assert len(users) == 1, f"official fold {fold_name} mixes users {users}"
        u = users[0]
        train = np.flatnonzero((meta["user"] != u).to_numpy())
        # the held-out user contributes nothing to training
        assert set(meta["user"].iloc[train]) == set(meta["user"]) - {u}
        folds.append((u, train, test))
    assert sum(len(t) for _, _, t in folds) == len(meta), "official folds do not cover all trials"
    return sorted(folds, key=lambda f: f[0])


def y_int(meta: pd.DataFrame) -> np.ndarray:
    """Skill labels as 0 = N, 1 = I, 2 = E."""
    return meta["skill"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy(dtype=int)
