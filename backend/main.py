"""SelenoMatch API. Run: uvicorn main:app --reload --port 8000"""
import asyncio, json, threading, time, uuid
from pathlib import Path
import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import pipeline, synth, train

ROOT = Path(__file__).parent
UP, OUT, MODELS = ROOT / "uploads", ROOT / "outputs", ROOT / "models"
for d in (UP, OUT, MODELS):
    d.mkdir(exist_ok=True)

app = FastAPI(title="SelenoMatch")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/files", StaticFiles(directory=OUT), name="files")

IMAGES: dict = {}       # image_id -> {path, sensor, sun_el, sun_az}
JOBS: dict = {}         # job_id   -> {status, stage, result, pair}
TRAIN = {"status": "idle", "epoch": 0, "epochs": 0, "loss": None}


@app.on_event("startup")
def warmup():
    threading.Thread(target=pipeline.get_matcher, daemon=True).start()   # load/download model at boot


# ---------- upload ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), sensor: str = Form("OHRC"),
                 sun_el: float = Form(45), sun_az: float = Form(90), bands: str = Form("")):
    iid = uuid.uuid4().hex[:8]
    path = UP / f"{iid}_{file.filename}"
    path.write_bytes(await file.read())
    # TODO: parse real Sun angles / GSD from PDS3/PDS4 labels instead of trusting form values
    bl = [int(b) for b in bands.split(",") if b.strip()] or None
    img = pipeline.load_image(path, bl)
    prev = OUT / f"{iid}_preview.png"
    cv2.imwrite(str(prev), (img * 255).astype("uint8"))
    IMAGES[iid] = dict(path=str(path), sensor=sensor, sun_el=sun_el, sun_az=sun_az, bands=bl)
    return {"image_id": iid, "preview_png": f"/files/{prev.name}",
            "meta": {"sun_el": sun_el, "sun_az": sun_az, "sensor": sensor, "size": list(img.shape[::-1])}}


# ---------- match ----------
class MatchReq(BaseModel):
    image_a: str
    image_b: str
    options: dict = {}


def _do_match(job_id: str, req: MatchReq):
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        A, B = IMAGES[req.image_a], IMAGES[req.image_b]
        ia, ib = pipeline.load_image(A["path"], A["bands"]), pipeline.load_image(B["path"], B["bands"])
        r = pipeline.run_match(ia, ib, A, B, req.options, progress=lambda s: job.update(stage=s))
        reg = OUT / f"{job_id}_registered.png"
        cv2.imwrite(str(reg), (r.pop("_registered") * 255).astype("uint8"))
        r.pop("_a")
        r["registered_png"] = f"/files/{reg.name}"
        r["geotiff_url"] = f"/api/export?job_id={job_id}&format=geotiff"
        job.update(status="done", result=r)
    except Exception as e:
        job.update(status="error", error=str(e))


@app.post("/api/match")
def match(req: MatchReq, bg: BackgroundTasks):
    for k in (req.image_a, req.image_b):
        if k not in IMAGES:
            raise HTTPException(404, f"unknown image {k}")
    jid = "job-" + uuid.uuid4().hex[:6]
    JOBS[jid] = dict(status="queued", stage="", result=None, pair=f"{IMAGES[req.image_a]['sensor']}–{IMAGES[req.image_b]['sensor']}")
    bg.add_task(_do_match, jid, req)
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404)
    return JOBS[jid]


@app.get("/api/jobs")
def jobs():
    return [{"id": k, "pair": v["pair"], "status": v["status"]} for k, v in JOBS.items()]


# ---------- synthetic data ----------
class SynthReq(BaseModel):
    dem: str | None = None
    sun_el: list[float] = [5, 75]
    gsd_ratios: list[float] = [1, 4, 16, 64]
    n: int = 1000


SYNTH = {"status": "idle", "done": 0, "n": 0}


@app.post("/api/synth/generate")
def synth_gen(req: SynthReq):
    def work():
        SYNTH.update(status="running", n=req.n, done=0)
        synth.generate_dataset("data/synth", req.n, req.dem, lambda i, n: SYNTH.update(done=i),
                               el_range=tuple(req.sun_el), scale_ratios=tuple(req.gsd_ratios))
        SYNTH["status"] = "done"
    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.get("/api/synth/status")
def synth_status():
    return SYNTH


# ---------- training ----------
class TrainReq(BaseModel):
    stage: str = "synthetic"
    epochs: int = 30


@app.post("/api/train")
def start_train(req: TrainReq):
    def work():
        try:
            train.run("data/synth", req.epochs, TRAIN)
        except Exception as e:
            TRAIN.update(status="error", error=str(e))
    threading.Thread(target=work, daemon=True).start()
    return {"started": True}


@app.websocket("/ws/train")
async def ws_train(ws: WebSocket):
    await ws.accept()
    while True:
        await ws.send_text(json.dumps(TRAIN))
        if TRAIN["status"] in ("done", "error"):
            break
        await asyncio.sleep(1)
    await ws.close()


# ---------- benchmark / export ----------
@app.get("/api/benchmark")
def benchmark():
    f = ROOT / "benchmark.json"      # produce this with an eval script: SelenoMatch vs cv2 SIFT/ORB
    return json.loads(f.read_text()) if f.exists() else {"pairs": [], "curves": {}}


@app.get("/api/export")
def export(job_id: str, format: str = "png"):
    j = JOBS.get(job_id)
    if not j or j["status"] != "done":
        raise HTTPException(404)
    if format == "json":
        return j["result"]
    png = OUT / f"{job_id}_registered.png"
    if format == "geotiff":
        import rasterio
        img = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        tif = OUT / f"{job_id}.tif"
        # TODO: pass a real CRS/transform (e.g. from ISIS3 map projection) here
        with rasterio.open(tif, "w", driver="GTiff", height=img.shape[0], width=img.shape[1], count=1, dtype="uint8") as d:
            d.write(img, 1)
        return FileResponse(tif)
    return FileResponse(png)


# optional: serve the frontend from the same server
if (ROOT.parent / "selenomatch.html").exists():
    @app.get("/")
    def index():
        return FileResponse(ROOT.parent / "selenomatch.html")
