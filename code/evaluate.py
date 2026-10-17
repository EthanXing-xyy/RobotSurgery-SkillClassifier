"""Leave-one-user-out evaluation, metrics and uncertainty for the skill-classification models.

Every model is evaluated with the same machinery: an outer loop over the 8 official LOUO folds
and, inside each outer fold, an inner LOUO loop over that fold's 7 training users which chooses
the hyperparameters. The held-out user never enters a scaler, a PCA, a residualisation or a
hyperparameter choice.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, \
    precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import CLASSES

LR_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
PCA_GRID = [4, 8, 16]
SEED = 0
N_LABELS = list(range(len(CLASSES)))


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

def make_model(C: float, n_components: int | None = None) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if n_components is not None:
        steps.append(("pca", PCA(n_components=n_components, random_state=SEED)))
        steps.append(("scale2", StandardScaler()))
    steps.append(("lr", LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                           solver="lbfgs", random_state=SEED)))
    return Pipeline(steps)


def _grid(use_pca: bool, n_features: int) -> list[tuple[float, int | None]]:
    if not use_pca:
        return [(c, None) for c in LR_GRID]
    return [(c, k) for k in PCA_GRID if k < n_features for c in LR_GRID]


def _fit_predict_proba(model: Pipeline, Xtr, ytr, Xte) -> np.ndarray:
    """Class probabilities over all three classes, also when a class is absent from ytr."""
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)
    full = np.zeros((len(Xte), len(CLASSES)))
    for j, c in enumerate(model.named_steps["lr"].classes_):
        full[:, int(c)] = p[:, j]
    return full


# ----------------------------------------------------------------------
# nested LOUO
# ----------------------------------------------------------------------

def inner_select(X, y, users, use_pca: bool) -> tuple[tuple[float, int | None], dict]:
    """Choose (C, n_components) by inner LOUO over the given training users.

    Inner folds whose own training part does not contain all three classes are skipped:
    with two users per trained class, holding out one of them can leave a class empty, and
    such a fold cannot produce a meaningful three-class score. The number of skipped folds
    is returned for the record.
    """
    uu = sorted(set(users))
    cand = _grid(use_pca, X.shape[1])
    scores, skipped = [], 0
    folds = []
    for u in uu:
        tr = np.flatnonzero(users != u)
        te = np.flatnonzero(users == u)
        if len(set(y[tr])) < len(CLASSES):
            skipped += 1
            continue
        folds.append((tr, te))
    for C, k in cand:
        pred, true = [], []
        for tr, te in folds:
            p = _fit_predict_proba(make_model(C, k), X[tr], y[tr], X[te])
            pred.append(p.argmax(axis=1))
            true.append(y[te])
        if not pred:
            scores.append(float("nan"))
            continue
        scores.append(f1_score(np.concatenate(true), np.concatenate(pred),
                               labels=N_LABELS, average="macro", zero_division=0))
    scores = np.asarray(scores, dtype=float)
    if np.all(np.isnan(scores)):
        best = (1.0, PCA_GRID[0] if use_pca else None)
    else:
        # ties -> stronger regularisation (smaller C) and fewer components: cand is already
        # ordered with k ascending and C ascending, so the first maximum satisfies both
        best = cand[int(np.nanargmax(scores))]
    return best, {"inner_scores": scores.tolist(), "inner_folds_used": len(folds),
                  "inner_folds_skipped": skipped,
                  "candidates": [[c, k] for c, k in cand]}


def louo_oof(X, y, users, folds, use_pca: bool = False, preprocess=None) -> dict:
    """Out-of-fold probabilities over the outer LOUO folds, with nested hyperparameters.

    preprocess(Xtr, Xte) -> (Xtr', Xte') is fitted on the training users only (used by the
    duration-residualisation control).
    """
    X = np.asarray(X, dtype=float)
    proba = np.zeros((len(y), len(CLASSES)))
    chosen = {}
    inner_info = {}
    for u, tr, te in folds:
        Xtr, Xte = X[tr], X[te]
        if preprocess is not None:
            Xtr, Xte = preprocess(Xtr, Xte)
        (C, k), info = inner_select(Xtr, y[tr], users[tr], use_pca)
        proba[te] = _fit_predict_proba(make_model(C, k), Xtr, y[tr], Xte)
        chosen[u] = {"C": C, "n_components": k}
        inner_info[u] = info
    return {"proba": proba, "pred": proba.argmax(axis=1), "chosen": chosen,
            "inner": inner_info}


def majority_oof(y, users, folds) -> dict:
    """Majority class of each fold's training users."""
    proba = np.zeros((len(y), len(CLASSES)))
    for u, tr, te in folds:
        counts = np.bincount(y[tr], minlength=len(CLASSES))
        proba[te, int(counts.argmax())] = 1.0
    return {"proba": proba, "pred": proba.argmax(axis=1), "chosen": {}, "inner": {}}


def late_fuse(*probas: np.ndarray) -> dict:
    """Average the class probabilities of several models (equal weights)."""
    p = np.mean(np.stack(probas, axis=0), axis=0)
    return {"proba": p, "pred": p.argmax(axis=1), "chosen": {}, "inner": {}}


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def macro_f1(y, pred) -> float:
    return float(f1_score(y, pred, labels=N_LABELS, average="macro", zero_division=0))


def metrics(y, pred, users, folds) -> dict:
    """Pooled out-of-fold metrics plus the per-user and per-fold breakdown."""
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=N_LABELS, zero_division=0)
    per_user = {}
    for u, _, te in folds:
        per_user[u] = {"n": int(len(te)), "accuracy": float(np.mean(pred[te] == y[te])),
                       "true_class": CLASSES[int(y[te][0])]}
    fold_acc = np.array([per_user[u]["accuracy"] for u, _, _ in folds])
    return {
        "macro_f1": macro_f1(y, pred),
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "per_class": {CLASSES[i]: {"precision": float(pr[i]), "recall": float(rc[i]),
                                    "f1": float(f1[i]), "support": int(sup[i])}
                      for i in N_LABELS},
        "confusion_matrix": confusion_matrix(y, pred, labels=N_LABELS).tolist(),
        "per_user": per_user,
        "fold_accuracy_mean": float(fold_acc.mean()),
        "fold_accuracy_sd": float(fold_acc.std(ddof=1)),
    }


# ----------------------------------------------------------------------
# uncertainty
# ----------------------------------------------------------------------

def user_bootstrap(y, pred, users, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Percentile interval of pooled macro-F1 under resampling of the 8 users."""
    rng = np.random.default_rng(seed)
    uu = np.array(sorted(set(users)))
    idx = {u: np.flatnonzero(users == u) for u in uu}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[u] for u in rng.choice(uu, size=len(uu), replace=True)])
        vals.append(macro_f1(y[take], pred[take]))
    v = np.asarray(vals)
    return {"point": macro_f1(y, pred), "lo": float(np.percentile(v, 2.5)),
            "hi": float(np.percentile(v, 97.5)), "n_boot": n_boot}


def paired_user_bootstrap(y, pred_a, pred_b, users, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Interval of the paired difference macro-F1(a) - macro-F1(b) over user resamples."""
    rng = np.random.default_rng(seed)
    uu = np.array(sorted(set(users)))
    idx = {u: np.flatnonzero(users == u) for u in uu}
    vals = []
    for _ in range(n_boot):
        take = np.concatenate([idx[u] for u in rng.choice(uu, size=len(uu), replace=True)])
        vals.append(macro_f1(y[take], pred_a[take]) - macro_f1(y[take], pred_b[take]))
    v = np.asarray(vals)
    return {"point": macro_f1(y, pred_a) - macro_f1(y, pred_b),
            "lo": float(np.percentile(v, 2.5)), "hi": float(np.percentile(v, 97.5)),
            "p_sign": float(2 * min(np.mean(v <= 0), np.mean(v >= 0))), "n_boot": n_boot}


def exact_user_label_permutation(y, pred, users) -> dict:
    """Exact permutation test over all reassignments of the user labels.

    A skill label belongs to a user, not to a trial, so the only admissible permutation
    keeps each user's trials together and permutes which users carry which label. With
    4 novice, 2 intermediate and 2 expert users there are 8!/(4!2!2!) = 420 distinct
    reassignments, so the null distribution is enumerated exactly rather than sampled.
    """
    uu = sorted(set(users))
    rows = {u: np.flatnonzero(users == u) for u in uu}
    user_label = {u: int(y[rows[u]][0]) for u in uu}
    assert all(len(set(y[rows[u]])) == 1 for u in uu), "a user carries more than one label"
    labels = [user_label[u] for u in uu]
    observed = macro_f1(y, pred)
    seen, null = set(), []
    for perm in permutations(labels):
        if perm in seen:
            continue
        seen.add(perm)
        y_perm = np.empty_like(y)
        for u, lab in zip(uu, perm):
            y_perm[rows[u]] = lab
        null.append(macro_f1(y_perm, pred))
    null = np.asarray(null)
    return {"observed": observed, "n_permutations": int(len(null)),
            "p_value": float(np.mean(null >= observed)),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1))}


# ----------------------------------------------------------------------
# shortcut audit
# ----------------------------------------------------------------------

def identity_probe(X, users, use_pca: bool = False, n_splits: int = 5, seed: int = SEED) -> dict:
    """How well a feature block predicts which of the 8 users produced the trial.

    This cannot be asked leave-one-user-out (the held-out class is unseen), so trials are
    split with a stratified 5-fold over the user label. Chance level is 1/8 = 0.125.
    """
    X = np.asarray(X, dtype=float)
    u = np.asarray(users)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.empty_like(u)
    for tr, te in skf.split(X, u):
        model = make_model(1.0, PCA_GRID[1] if use_pca else None)
        model.fit(X[tr], u[tr])
        pred[te] = model.predict(X[te])
    return {"balanced_accuracy": float(balanced_accuracy_score(u, pred)),
            "accuracy": float(np.mean(pred == u)), "chance": 1.0 / len(set(u))}


def residualise_on(column: np.ndarray):
    """preprocess callable that removes a linear fit on `column` from every feature.

    The fit (slope and intercept per feature) is estimated on the training users only, so the
    held-out user does not influence it.
    """
    col = np.asarray(column, dtype=float)

    def _pp_factory(tr_idx, te_idx):
        def _pp(Xtr, Xte):
            z_tr = np.log(col[tr_idx]).reshape(-1, 1)
            z_te = np.log(col[te_idx]).reshape(-1, 1)
            A = np.hstack([z_tr, np.ones_like(z_tr)])
            coef, *_ = np.linalg.lstsq(A, Xtr, rcond=None)
            fit_tr = A @ coef
            fit_te = np.hstack([z_te, np.ones_like(z_te)]) @ coef
            return Xtr - fit_tr, Xte - fit_te
        return _pp

    return _pp_factory


def louo_oof_with_index_preprocess(X, y, users, folds, pp_factory, use_pca: bool = False) -> dict:
    """louo_oof variant whose preprocessing needs to know the row indices of each fold."""
    X = np.asarray(X, dtype=float)
    proba = np.zeros((len(y), len(CLASSES)))
    chosen, inner_info = {}, {}
    for u, tr, te in folds:
        Xtr, Xte = pp_factory(tr, te)(X[tr], X[te])
        (C, k), info = inner_select(Xtr, y[tr], users[tr], use_pca)
        proba[te] = _fit_predict_proba(make_model(C, k), Xtr, y[tr], Xte)
        chosen[u] = {"C": C, "n_components": k}
        inner_info[u] = info
    return {"proba": proba, "pred": proba.argmax(axis=1), "chosen": chosen, "inner": inner_info}
