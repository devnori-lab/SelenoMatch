"""SelenoMatch core pipeline: read -> normalize -> coarse-to-fine match -> verify -> explain."""
import logging, threading, time
from pathlib import Path
import cv2, numpy as np, torch
import kornia.feature as KF

log = logging.getLogger("uvicorn.error")
cv2.setNumThreads(0)                       # avoids rare OpenCV/torch thread hangs
torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS = Path("models/loftr-lunar-v1.pt")   # your fine-tuned weights (optional)
_matcher = None
_lock = threading.Lock()


def get_matcher():
    global _matcher
    with _lock:
      if _matcher is None:
        log.info("Loading LoFTR on %s (first run downloads ~45 MB of weights)...", DEVICE)
        t = time.time()
        if WEIGHTS.exists():
            m = KF.LoFTR(pretrained=None)
            m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        else:
            m = KF.LoFTR(pretrained="outdoor")      # start here, swap in fine-tuned later
        _matcher = m.eval().to(DEVICE)
        log.info("LoFTR ready in %.1fs", time.time() - t)
    return _matcher


# ---------- 1. ingestion ----------
def load_image(path, bands=None):
    """GeoTIFF/PDS via rasterio, else PNG/JPG. For IIRS pass reflectance band indices (1-based);
    they are averaged into a pseudo-panchromatic image. Returns float32 [0,1]."""
    try:
        import rasterio
        with rasterio.open(path) as ds:
            idx = bands or ([1] if ds.count == 1 else list(range(1, min(ds.count, 3) + 1)))
            arr = ds.read(idx).astype("float32").mean(0)
    except Exception:
        arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise ValueError(f"Cannot read {path}")
        arr = arr.astype("float32")
    lo, hi = np.percentile(arr, (1, 99))
    return np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1)


# ---------- 2. illumination normalization ----------
def normalize(img, sun_el=None, deshade=True):
    if deshade and sun_el:                        # crude Lambertian de-shading
        img = img / max(np.sin(np.radians(sun_el)), 0.1)
        img = np.clip(img / (np.percentile(img, 99) + 1e-6), 0, 1)
    u8 = (img * 255).astype(np.uint8)
    u8 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(u8)
    f = u8.astype("float32") / 255
    mu = cv2.GaussianBlur(f, (0, 0), 15)
    sd = np.sqrt(cv2.GaussianBlur((f - mu) ** 2, (0, 0), 15)) + 1e-3
    n = (f - mu) / sd                             # local contrast normalization
    return np.clip(n * 0.25 + 0.5, 0, 1).astype("float32")


# ---------- 3/4. learned matcher on one pyramid level ----------
def _prep(x, size):
    h, w = x.shape
    s = size / max(h, w)
    nh, nw = max(8, int(h * s) // 8 * 8), max(8, int(w * s) // 8 * 8)
    r = cv2.resize(x, (nw, nh), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(r)[None, None].to(DEVICE), np.array([w / nw, h / nh])


def _loftr(a, b, size):
    ta, sa = _prep(a, size)
    tb, sb = _prep(b, size)
    with torch.no_grad():
        out = get_matcher()({"image0": ta, "image1": tb})
    return (out["keypoints0"].cpu().numpy() * sa,
            out["keypoints1"].cpu().numpy() * sb,
            out["confidence"].cpu().numpy())


def _apply(H, pts):
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype("float32"), H).reshape(-1, 2)


def _reason(img, pt, res, conf):
    x, y = int(np.clip(pt[0], 8, img.shape[1] - 9)), int(np.clip(pt[1], 8, img.shape[0] - 9))
    patch = img[y - 8:y + 8, x - 8:x + 8]
    kind = "crater rim / ridge edge structure" if patch.std() > 0.12 else "low-relief texture cluster"
    return f"matched via {kind}, illumination-normalized, residual {res:.1f} px, conf {conf:.2f}"


# ---------- full pipeline ----------
def run_match(img_a, img_b, meta_a=None, meta_b=None, opts=None, progress=lambda s: None):
    opts = opts or {}
    t0 = time.time()
    levels = int(opts.get("levels", 4))
    conf_th = float(opts.get("conf_th", 0.6))
    meta_a, meta_b = meta_a or {}, meta_b or {}

    progress("Illumination normalization")
    a = normalize(img_a, meta_a.get("sun_el"), opts.get("deshade", True)) if opts.get("normalize", True) else img_a
    b0 = normalize(img_b, meta_b.get("sun_el"), opts.get("deshade", True)) if opts.get("normalize", True) else img_b

    progress("Multi-scale pyramid")
    hb, wb = b0.shape
    ha, wa = a.shape
    S = np.diag([wa / wb, ha / hb, 1.0])                       # bring B onto A's pixel grid (GSD bridge)
    b = cv2.resize(b0, (wa, ha), interpolation=cv2.INTER_AREA)
    sizes = sorted({min(512 * 2 ** i, max(wa, ha)) for i in range(levels)})

    if DEVICE == "cpu":                                         # keep CPU runs fast
        sizes = [s for s in sizes if s <= 1024] or [512]
    progress("Learned matcher (loading model)")
    get_matcher()
    progress("Learned matcher")
    H = np.eye(3)                                               # maps A -> B(resized)
    p = qw = c = None
    for s in sizes:                                             # coarse -> fine
        warped = cv2.warpPerspective(b, np.linalg.inv(H), (wa, ha))
        t1 = time.time()
        p, qw, c = _loftr(a, warped, s)
        log.info("level %d px: %d matches in %.1fs", s, len(p), time.time() - t1)
        if len(p) < 8:
            break
        R, _ = cv2.findHomography(p, qw, cv2.RANSAC, 3.0)
        if R is None:
            break
        H = H @ R
    if p is None or len(p) < 4:
        raise RuntimeError("Not enough matches found")
    q = _apply(H, qw)                                           # matches in B(resized) coords

    progress("RANSAC / geometric verification")
    Hf, mask = cv2.findHomography(p, q, cv2.RANSAC, float(opts.get("ransac_px", 3.0)))
    Hf = H if Hf is None else Hf
    res = np.linalg.norm(_apply(Hf, p) - q, axis=1)
    kept = (mask.ravel() > 0) & (c >= conf_th) if mask is not None else c >= conf_th
    # TODO: DEM slope pruning / thin-plate-spline for high-relief patches

    progress("Confidence & explanation")
    inv_S = np.linalg.inv(S)
    q_orig = _apply(inv_S, q)                                   # back to B's native pixels
    order = np.argsort(-c)[:300]
    matches = [{"p": p[i].tolist(), "q": q_orig[i].tolist(), "conf": float(c[i]),
                "residual": float(res[i]), "kept": bool(kept[i]),
                "reason": _reason(a, p[i], res[i], c[i])} for i in order]

    progress("Output & registration")
    reg = cv2.warpPerspective(b, np.linalg.inv(Hf), (wa, ha))
    rms = float(np.sqrt(np.mean(res[kept] ** 2))) if kept.any() else None
    return {"matches": matches, "transform": Hf.tolist(),  # maps A -> B (B resized to A grid)
            "rms": rms, "time_ms": int((time.time() - t0) * 1000),
            "size_a": [wa, ha], "size_b": [wb, hb],
            "_registered": reg, "_a": a}
