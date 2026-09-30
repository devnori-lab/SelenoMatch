# SelenoMatch (SIH26166)

**Sun-angle and scale-invariant image matching for Chandrayaan-2 (OHRC · TMC · IIRS).**

## What it does

Given two lunar images of the same terrain, taken by different sensors and/or under different Sun angles, SelenoMatch finds matching points between them, estimates the transform that aligns them, and reports a confidence score and a short reason for each match.

Pipeline: read image → illumination normalization → coarse-to-fine pyramid → learned matcher (LoFTR) → RANSAC verification → confidence + explanation → registered output.

The repo contains:
- a **web UI** (`selenomatch.html`) covering the full planned feature set: upload, Sun-angle/sensor/pipeline settings, match visualization, swipe overlay, exports, synthetic-data generator, training, benchmarks, job history;
- a **FastAPI backend** (`backend/`) with a working baseline pipeline and a synthetic data generator.

## Current status and limitations (read this first)

This is a **working prototype/baseline, not the finished system.**

| Area | Today | Limitation |
|---|---|---|
| Matcher | Pretrained LoFTR ("outdoor") from Kornia | Trained on Earth photos, **not lunar terrain**. Expect weak results on low-texture or strongly shadowed lunar images |
| Sun-angle invariance | CLAHE + local contrast normalization + crude Lambertian de-shading | Not yet learned; large Sun-angle gaps will still hurt |
| Scale gap | B is resized to A's grid; coarse-to-fine pyramid | The roughly 300x gap between OHRC (0.25 m) and IIRS (80 m) has not been tested at real scale; on CPU the pyramid is capped at 1024 px |
| Synthetic training data | Generator works, but uses a **procedural crater DEM** by default and a basic Lambertian shader | Not yet real LOLA/OHRC DEMs; Hapke shader is a stub |
| Training | Dataset + loop written | **Loss function is not implemented** (`train.py` raises `NotImplementedError`), so no fine-tuned model exists yet |
| Real ISRO data | Not used | Sun angles are typed into the UI; PDS3/PDS4 label parsing and IIRS QUB cube reading are not implemented |
| Geometry | Homography + RANSAC | Thin-plate-spline and DEM slope pruning are not implemented; GeoTIFF export has no real map projection |
| Benchmarks | Placeholder numbers in the UI | No real comparison against SIFT/ORB yet |
| Edge deployment | Runs on CPU | ONNX export is unfinished; no on-device testing |
| Engineering | Jobs are kept in memory | No auth, no persistence, no tests |

Accuracy targets on the project slides are **targets**, not measured results.

## How the final project will solve these

| Limitation | Plan |
|---|---|
| Matcher not lunar-aware | Fine-tune LoFTR on synthetic lunar pairs (implement the official LoFTR coarse focal loss + fine L2 loss using the exact ground-truth homography from `synth.py`) |
| Sun-angle sensitivity | Train on thousands of DEM renders with varied Sun elevation/azimuth so invariance is learned, not hand-crafted |
| Synthetic ≠ real | Use real LOLA / OHRC-derived DEMs and a Hapke shader; add noise/radiometry jitter; domain-adapt on unlabelled real crops (cycle consistency + photometric loss) |
| Big GSD gap | Generate training pairs at 1×–320× scale ratios; keep coarse-to-fine matching; add IIRS band selection and a WAC seleno-referencing bootstrap for pre-alignment |
| Real data handling | Parse PDS labels for Sun angles and GSD; read IIRS cubes and build the pseudo-panchromatic band automatically; use ISIS3 for map projection and GeoTIFF export |
| Weak geometry | Add DEM-based slope pruning and thin-plate-spline fitting for high-relief patches |
| No evidence of performance | Benchmark vs OpenCV SIFT/ORB across Sun-angle gaps and sensor pairs; feed results to the Benchmarks tab |
| Edge use | Export to ONNX and run CPU-only on a laptop |

## Repo layout

```
selenomatch.html        # frontend (single file)
Dockerfile              # for free hosting (Hugging Face Spaces) / local Docker
backend/
  main.py               # FastAPI app + endpoints
  pipeline.py           # read → normalize → match → verify → explain
  synth.py              # synthetic DEM-rendered training pairs
  train.py              # fine-tuning skeleton (loss TODO)
  make_test_pair.py     # creates test_A.png / test_B.png
  check_matcher.py      # diagnoses model loading/matching problems
  requirements.txt
```

## Run it locally

Requires Python 3.10+ and internet on first run (pretrained weights ≈ 45 MB are downloaded).

```bash
git clone https://github.com/devnori-lab/SelenoMatch.git && cd SelenoMatch/backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Wait until the terminal prints `LoFTR ready`. Then open **http://localhost:8000**. The badge at top right should say **LIVE**.

If `rasterio` fails to install, use conda (`conda install -c conda-forge rasterio`). The code falls back to OpenCV for PNG/JPG.

## Test it

1. In a second terminal (same venv, inside `backend/`): `python make_test_pair.py`. This creates `test_A.png` and `test_B.png` (same terrain, different Sun angle, scale gap, small shift) and prints the true transform.
2. Optional sanity check: `python check_matcher.py` (prints model load and match timings).
3. In the UI: upload `test_A.png` as Image A and `test_B.png` as Image B, then click **Run matching**.
4. Expect: stages turn green; match lines run roughly parallel; the transform is close to the printed true one; the swipe overlay lines up.
5. Other tabs: in **Synthetic Data** set Pairs to 20 and click *Start dataset job* (files appear in `backend/data/synth/`). **Data & Jobs** lists your jobs. **Training** will show an error until `loftr_loss` is implemented (expected).
6. API docs: http://localhost:8000/docs. Quick check: `curl http://localhost:8000/api/jobs` → `[]`.

Use small crops (about 512–1024 px) when testing on CPU.

## Free hosting

- **Code:** push to a public GitHub repo.
- **Frontend only (mock mode):** GitHub Pages can serve `selenomatch.html`; it runs with fake data because there is no backend.
- **Full live demo:** a **Hugging Face Space** using the Docker SDK runs the backend and frontend together on free CPU hardware.
  1. Create a new Space → SDK: **Docker** → blank.
  2. Add this to the top of the Space's `README.md` (not the GitHub one):
     ```
     ---
     title: SelenoMatch
     sdk: docker
     app_port: 7860
     ---
     ```
  3. Push this repo (with the `Dockerfile`) to the Space. Build takes several minutes; then open the Space URL.
  Free tiers change, and free Spaces sleep when idle, so check current limits.
- **Finale fallback:** run locally on a laptop. The project is designed to work offline and on CPU anyway.

## API summary

| Endpoint | Purpose |
|---|---|
| `POST /api/upload` | Upload image + sensor + Sun angles → `image_id`, preview |
| `POST /api/match` | Start a matching job → `job_id` |
| `GET /api/jobs/{id}`, `GET /api/jobs` | Job status/result, job list |
| `POST /api/synth/generate`, `GET /api/synth/status` | Build synthetic dataset |
| `POST /api/train`, `WS /ws/train` | Fine-tuning + progress |
| `GET /api/benchmark` | Reads `backend/benchmark.json` if present |
| `GET /api/export?job_id=&format=geotiff\|json\|png` | Export results |

## Troubleshooting

- **Badge says MOCK:** backend isn't running, or you opened a hosted copy. Open http://localhost:8000. If you open the HTML as a file it targets `localhost:8000`; use `?api=http://host:port` to change it.
- **Stuck at "Learned matcher":** usually the first-run weight download or a very large image on CPU. Check the uvicorn terminal, run `python check_matcher.py`, try another network, use smaller crops.
- **"upload both images first":** both slots need a file in live mode.

## Data sources

ISRO PRADAN/ISSDC (OHRC, TMC, IIRS) · NASA PDS Geosciences/Imaging Nodes (LRO WAC/NAC, LOLA DEM) · LoFTR (Sun et al., CVPR 2021) via Kornia.

## License

Add your license here.
