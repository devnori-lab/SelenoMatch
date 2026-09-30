"""Synthetic training pairs: render one DEM under two Sun angles/scales; GT homography is exact."""
import numpy as np, cv2
from pathlib import Path


def load_dem(path=None, n=1024, seed=0):
    if path:
        import rasterio
        with rasterio.open(path) as ds:
            return ds.read(1).astype("float32")
    rng = np.random.default_rng(seed)                 # procedural craters fallback
    h = np.zeros((n, n), "float32")
    yy, xx = np.mgrid[:n, :n]
    for _ in range(300):
        cx, cy, r = rng.uniform(0, n, 2).tolist() + [rng.uniform(8, 90)]
        d = np.hypot(xx - cx, yy - cy) / r
        h += np.where(d < 1, -0.25 * r * (1 - d ** 2), 0.15 * r * np.exp(-((d - 1) * 4) ** 2) * (d < 1.6))
    return h


def shade(dem, sun_el, sun_az, mpp=1.0, shader="lambertian"):
    e, a = np.radians(sun_el), np.radians(sun_az)
    L = np.array([np.cos(e) * np.sin(a), -np.cos(e) * np.cos(a), np.sin(e)])
    gy, gx = np.gradient(dem, mpp)
    n = np.sqrt(gx ** 2 + gy ** 2 + 1)
    v = np.clip((-gx * L[0] - gy * L[1] + L[2]) / n, 0, 1)
    if shader == "hapke":                              # TODO: proper Hapke model
        v = v ** 0.8
    return v.astype("float32")


def make_pair(dem, size=512, el_range=(5, 75), scale_ratios=(1, 4, 16), rng=None):
    rng = rng or np.random.default_rng()
    el0, el1 = rng.uniform(*el_range, 2)
    az0, az1 = rng.uniform(0, 359, 2)
    i0, i1 = shade(dem, el0, az0), shade(dem, el1, az1)
    y, x = rng.integers(0, dem.shape[0] - size), rng.integers(0, dem.shape[1] - size)
    crop = lambda im: im[y:y + size, x:x + size]
    im0, im1 = crop(i0), crop(i1)
    s = float(rng.choice(scale_ratios))                # simulate GSD gap: view 1 is coarser
    if s > 1:
        small = cv2.resize(im1, (int(size / s) // 8 * 8 or 8,) * 2, interpolation=cv2.INTER_AREA)
        im1 = cv2.resize(small, (size, size))
    ang, t = np.radians(rng.uniform(-10, 10)), rng.uniform(-20, 20, 2)
    A = np.array([[np.cos(ang), -np.sin(ang), t[0]], [np.sin(ang), np.cos(ang), t[1]], [0, 0, 1]])
    im1 = cv2.warpPerspective(im1, A, (size, size))
    im1 = np.clip(im1 + rng.normal(0, 0.02, im1.shape), 0, 1).astype("float32")
    return im0, im1, A, dict(el0=el0, az0=az0, el1=el1, az1=az1, scale=s)


def generate_dataset(out_dir, n=1000, dem_path=None, progress=lambda i, n: None, **kw):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    dem, rng = load_dem(dem_path), np.random.default_rng(0)
    for i in range(n):
        a, b, H, m = make_pair(dem, rng=rng, **kw)
        np.savez_compressed(out / f"pair_{i:05d}.npz", img0=a, img1=b, H=H, **m)   # H maps img0 -> img1
        progress(i + 1, n)
    return str(out)
