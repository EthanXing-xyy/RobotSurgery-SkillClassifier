"""Trial-level video features from sampled frames of the left endoscope view (capture1).

One decode pass per trial produces three blocks:

V-appear  mean and standard deviation, over frames sampled at 1 Hz, of the 512-d penultimate
          activation of an ImageNet-pretrained ResNet-18. This describes what the scene looks
          like, which includes the setup of the session as well as the instruments.
V-dyn     statistics of the cosine distance between consecutive 1 Hz embeddings, i.e. how fast
          the view changes. Invariant to a constant appearance offset, so a session's static
          setup cancels out.
V-flow    Farneback optical-flow statistics over consecutive frame pairs sampled at 2 Hz:
          magnitude, the fraction of moving pixels and the entropy of the flow direction.

Features are cached per trial under cache/video_<VERSION>/.
"""
from __future__ import annotations

import numpy as np

from data import CACHE_DIR, FPS, video_path

VERSION = "vid-v1"

EMB_HZ = 1.0          # embedding sampling rate
FLOW_HZ = 2.0         # optical-flow pair sampling rate
FLOW_SIZE = (320, 240)
MOVING_PX = 0.5       # px/frame; above this a pixel counts as moving
RESNET_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL = None


def _resnet18():
    """ImageNet ResNet-18 truncated after global pooling (512-d), on GPU when available."""
    global _MODEL
    if _MODEL is None:
        import torch
        import torchvision
        net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()
        net.eval()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL = (net.to(dev), dev)
    return _MODEL


def _decode(trial_id: str) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Single pass over the video: 1 Hz RGB frames and 2 Hz consecutive grey pairs."""
    import cv2
    cap = cv2.VideoCapture(str(video_path(trial_id)))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video for {trial_id}")
    emb_step = int(round(FPS / EMB_HZ))
    flow_step = int(round(FPS / FLOW_HZ))
    frames: list[np.ndarray] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    pending_first: np.ndarray | None = None
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % emb_step == 0:
            rgb = cv2.cvtColor(cv2.resize(bgr, (RESNET_SIZE, RESNET_SIZE),
                                          interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        if i % flow_step == 0:
            pending_first = cv2.cvtColor(cv2.resize(bgr, FLOW_SIZE, interpolation=cv2.INTER_AREA),
                                         cv2.COLOR_BGR2GRAY)
        elif pending_first is not None and i % flow_step == 1:
            second = cv2.cvtColor(cv2.resize(bgr, FLOW_SIZE, interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2GRAY)
            pairs.append((pending_first, second))
            pending_first = None
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded for {trial_id}")
    return frames, pairs


def _embed(frames: list[np.ndarray], batch: int = 64) -> np.ndarray:
    """(n_frames, 512) embeddings of the sampled frames."""
    import torch
    net, dev = _resnet18()
    out = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            arr = np.stack(frames[i:i + batch]).astype(np.float32) / 255.0
            arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
            t = torch.from_numpy(arr.transpose(0, 3, 1, 2)).to(dev)
            out.append(net(t).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _dyn_features(emb: np.ndarray) -> dict[str, float]:
    """Statistics of the frame-to-frame embedding change (scene-change rate)."""
    if len(emb) < 3:
        return {k: float("nan") for k in
                ["dyn_cos_mean", "dyn_cos_std", "dyn_cos_p10", "dyn_cos_p50", "dyn_cos_p90",
                 "dyn_cos_cv", "dyn_l2_mean", "dyn_l2_std"]}
    nrm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    cos_d = 1.0 - np.sum(nrm[:-1] * nrm[1:], axis=1)
    l2_d = np.linalg.norm(emb[:-1] - emb[1:], axis=1)
    return {
        "dyn_cos_mean": float(cos_d.mean()), "dyn_cos_std": float(cos_d.std()),
        "dyn_cos_p10": float(np.percentile(cos_d, 10)), "dyn_cos_p50": float(np.percentile(cos_d, 50)),
        "dyn_cos_p90": float(np.percentile(cos_d, 90)),
        "dyn_cos_cv": float(cos_d.std() / (cos_d.mean() + 1e-9)),
        "dyn_l2_mean": float(l2_d.mean()), "dyn_l2_std": float(l2_d.std()),
    }


def _flow_features(pairs: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, float]:
    """Farneback flow statistics, aggregated over the sampled frame pairs."""
    import cv2
    names = ["mag_mean", "mag_std", "mag_p90", "moving_frac", "dir_entropy"]
    if not pairs:
        return {f"flow_{n}_{a}": float("nan") for n in names for a in ("mean", "std", "p90")}
    per_pair = []
    for a, b in pairs:
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        moving = mag > MOVING_PX
        hist, _ = np.histogram(ang[moving], bins=8, range=(0, 2 * np.pi),
                               weights=mag[moving]) if moving.any() else (np.zeros(8), None)
        p = hist / hist.sum() if hist.sum() > 0 else np.full(8, 1 / 8)
        p = np.clip(p, 1e-12, None)
        per_pair.append([float(mag.mean()), float(mag.std()), float(np.percentile(mag, 90)),
                         float(moving.mean()), float(-np.sum(p * np.log(p)) / np.log(8))])
    arr = np.asarray(per_pair)
    out = {}
    for j, n in enumerate(names):
        out[f"flow_{n}_mean"] = float(arr[:, j].mean())
        out[f"flow_{n}_std"] = float(arr[:, j].std())
        out[f"flow_{n}_p90"] = float(np.percentile(arr[:, j], 90))
    return out


def trial_features(trial_id: str, use_cache: bool = True) -> dict[str, np.ndarray | dict]:
    """{'appear': (1024,), 'dyn': {...}, 'flow': {...}} for one trial, cached on disk."""
    cdir = CACHE_DIR / f"video_{VERSION}"
    cdir.mkdir(parents=True, exist_ok=True)
    cfile = cdir / f"{trial_id}.npz"
    if use_cache and cfile.exists():
        z = np.load(cfile, allow_pickle=True)
        return {"appear": z["appear"], "dyn": z["dyn"].item(), "flow": z["flow"].item(),
                "n_frames": int(z["n_frames"]), "n_pairs": int(z["n_pairs"])}
    frames, pairs = _decode(trial_id)
    emb = _embed(frames)
    appear = np.concatenate([emb.mean(axis=0), emb.std(axis=0)]).astype(np.float32)
    dyn = _dyn_features(emb)
    flow = _flow_features(pairs)
    np.savez_compressed(cfile, appear=appear, dyn=dyn, flow=flow,
                        n_frames=len(frames), n_pairs=len(pairs))
    return {"appear": appear, "dyn": dyn, "flow": flow,
            "n_frames": len(frames), "n_pairs": len(pairs)}
