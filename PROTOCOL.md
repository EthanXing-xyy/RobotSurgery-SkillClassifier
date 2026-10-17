# Evaluation protocol (fixed before any result was computed)

BN5211 Final Project, Topic 1: surgical skill assessment from kinematics and video.
Written 2026-10-17, before the pipeline was run. Nothing in this file may be changed
after the first result is read; deviations must be recorded in `logs/DEVIATIONS.md`.

## 1. Task and data

- Task: classify the skill level of a complete JIGSAWS **Suturing** trial as
  N (novice, < 10 h), I (intermediate, 10-100 h) or E (expert, > 100 h).
- Labels: self-proclaimed skill level, column 2 of `meta_file_Suturing.txt`.
- Trials: all 39 Suturing trials listed in the metadata file (users B, C, D, E, F, G, H, I;
  5 trials each except H, which has 4 because trial H002 is absent from the official release).
- Class counts: N = 19 (B, G, H, I), I = 10 (C, F), E = 10 (D, E).
- Inputs: 76-column kinematics at 30 Hz (`kinematics/AllGestures/`) and the left endoscope
  video `capture1` (`video/`). Gesture transcriptions are **not** used by any model.
- GRS scores (metadata columns 3-9) are **not** used as inputs by any model.
- Data are read-only from `A:\BN5211Project\Topic1\Baseline_shared\data\Suturing`.

## 2. Split and leakage rules

- **Leave-one-user-out (LOUO)**: 8 folds, fold *u* tests all trials of user *u* and trains on
  the trials of the other 7 users.
- The held-out user influences nothing: standardisation, PCA, feature residualisation,
  hyperparameter choice and class priors are all fitted inside the training users only.
- Hyperparameters are chosen by an **inner LOUO** loop over the 7 training users
  (nested cross-validation). No value is chosen from test-fold performance.
  Inner folds whose training part lacks one of the three classes (possible because only two
  users carry each of the I and E labels) are skipped, and their number is recorded.
- The fold definitions are the official JIGSAWS `Suturing/unBalanced/SkillDetection/UserOut`
  files (`1_Out` ... `8_Out`, `itr_1`); the code asserts that each holds out exactly one user.
- Per-trial predictions of all models are written once to `results/predictions_<model>.csv`.
  Models are not revised after their out-of-fold predictions have been inspected; any such
  revision has to be reported as a new model, with both versions kept.

## 3. Metrics

- **Primary: pooled out-of-fold macro-F1.** The 39 out-of-fold predictions are pooled and
  macro-F1 is computed once over the three classes.
  Rationale: a skill label is constant within a user, so a single LOUO fold contains exactly
  one true class and per-fold macro-F1 is undefined for the two absent classes. Pooling is
  the only form of macro-F1 that is defined here, and it is what the literature on this
  dataset reports.
- Secondary: pooled accuracy, pooled per-class F1 / precision / recall, confusion matrix,
  per-user accuracy, and the mean +- SD of per-fold accuracy.
- Uncertainty: 2000-draw **user-level bootstrap** (resample the 8 users with replacement) for
  the primary metric, and the same bootstrap on the paired difference for the three
  pre-declared comparisons in section 5.
- Significance: **exact user-level label permutation**. Each user's trials stay together and the
  user labels are permuted; with 4 N, 2 I and 2 E users there are 8!/(4!2!2!) = 420 distinct
  reassignments, so the null distribution of pooled macro-F1 (for fixed predictions) is
  enumerated exactly. The smallest attainable p-value is 1/420.

## 4. Models (fixed list)

Required by the assignment:

| id | inputs | model |
|---|---|---|
| M0 | none | majority class of the training users |
| M1 | K-simple: duration, path length, velocity, gripper motion statistics | standardise + L2 logistic regression |
| M2a | V-appear: ImageNet ResNet-18 frame embeddings, 1 fps, mean+SD pooled, PCA | standardise + L2 logistic regression |
| M2b | V-dyn: statistics of the frame-to-frame change of the same embeddings (scene-change rate) | standardise + L2 logistic regression |
| M2c | V-flow: Farneback optical-flow statistics, 2 Hz sampling of consecutive frame pairs | standardise + L2 logistic regression |
| M3 | **primary multimodal**: late fusion, mean of the M1 and M2b class probabilities | - |
| M3b | late fusion of M1 and M2a | - |
| M3c | late fusion of M1 and M2c | - |
| M3d | early fusion: concatenate K-simple with the V-dyn features | standardise + L2 logistic regression |

V-dyn is the pre-declared video block of the primary multimodal model, for two reasons. First
principles: a static endoscopic view mostly shows the setup of a session, which is a property of
the recording rather than of the surgeon's skill, whereas the rate at which the view changes
reflects instrument motion and tissue change. Prior evidence: in our own earlier exploratory work
on this dataset, embedding-change statistics were the only video block that carried skill
information while remaining a weak carrier of user identity. That earlier work read out-of-fold
numbers on the same 39 trials, so the choice is not selection-free; the report states this, and
all three video blocks and all four fusion variants are reported regardless of outcome.

One focused improvement (motivated by the task, not by complexity):

| id | inputs | model |
|---|---|---|
| M4 | K-simple + motion-texture features (approximate entropy, jerk, spectral) | standardise + L2 logistic regression |
| M5 | late fusion of M4 and M2a | - |

Controls and shortcut audit (reported, never used for selection):

| id | what it tests |
|---|---|
| C1 | duration only: how much of the decision is trial length |
| C2 | K-simple with every feature residualised on log duration inside the training users: does any signal survive the removal of pace |
| C3 | user-identity probe: 8-way user classification from each feature block (grouped 5-fold over trials, chance = 1/8 = 0.125), measuring how strongly a block encodes who performed the trial |
| C4 | video-only appearance features with the trial's own user excluded (identical to M2a; listed to make the comparison with C3 explicit) |

Hyperparameter grid (identical for every logistic-regression model, selected by inner LOUO):
`C` in {0.01, 0.03, 0.1, 0.3, 1, 3, 10}; PCA components for video blocks in {4, 8, 16}.
`class_weight="balanced"`, `max_iter=5000`, `solver="lbfgs"` (multinomial loss, the default for
this solver in scikit-learn 1.9), `random_state=0`. Inner selection maximises inner pooled macro-F1; ties take the smaller `C`
(stronger regularisation) and the smaller number of components.

## 5. Pre-declared comparisons

1. M1 vs M0: do kinematic statistics beat the majority class?
2. M3 vs M1: does adding video to kinematics help?
3. M4 vs M1: does motion texture add to the simple statistics?

Each is reported as the paired difference in pooled macro-F1 with a 2000-draw user-level
bootstrap interval. A difference is called supported only if the interval excludes 0; otherwise
it is reported as not distinguishable at this sample size.

## 6. Reproducibility

- Single fixed seed 0 for every randomised step (PCA solver, bootstrap, permutation, shuffles).
- Feature caches under `cache/` are content-addressed by the extractor version string.
- Environment: Python 3.11, scikit-learn, PyTorch (CUDA) on one RTX 3060 Laptop GPU, Windows 11.
- `code/run_all.py` reproduces every number in the report from the raw data.
