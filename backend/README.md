# SelenoMatch backend

## Run
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Docs: http://localhost:8000/docs · Put `selenomatch.html` in the parent folder and open http://localhost:8000

## Connect the frontend
In `selenomatch.html` set `const API_BASE="http://localhost:8000", USE_MOCK=false;` then replace the mock in `runMatch()`:
```js
async function runMatch(){
  const fd=(f,s,el,az)=>{const x=new FormData();x.append("file",f);x.append("sensor",s);x.append("sun_el",el);x.append("sun_az",az);return x};
  const [A,B]=await Promise.all([
    fetch(API_BASE+"/api/upload",{method:"POST",body:fd(fileA,$("sA").value,$("elA").value,$("azA").value)}).then(r=>r.json()),
    fetch(API_BASE+"/api/upload",{method:"POST",body:fd(fileB,$("sB").value,$("elB").value,$("azB").value)}).then(r=>r.json())]);
  const {job_id}=await fetch(API_BASE+"/api/match",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({image_a:A.image_id,image_b:B.image_id,options:{levels:+$("pl").value,conf_th:$("th").value/100}})}).then(r=>r.json());
  let j; do{ await new Promise(r=>setTimeout(r,500)); j=await fetch(API_BASE+"/api/jobs/"+job_id).then(r=>r.json()); /* highlight stage j.stage */ }while(j.status==="queued"||j.status==="running");
  // j.result.matches -> draw using size_a / size_b to scale p and q (they're native pixel coords)
}
```
The frontend needs `fileA`/`fileB` from its file inputs, and match drawing must scale by `size_a`/`size_b`.

## Build order
1. Test `/api/match` with two PNG crops (pretrained LoFTR works out of the box).
2. `POST /api/synth/generate` with n=500 to create `data/synth/*.npz`.
3. Implement `loftr_loss` in `train.py` (official LoFTR loss), train, save `models/loftr-lunar-v1.pt` (auto-loaded).
4. Add a benchmark script writing `benchmark.json` (SelenoMatch vs OpenCV SIFT/ORB across Sun-angle gaps).
5. Real data: parse PDS labels for Sun angles/GSD in `/api/upload`; add ISIS3 map-projection for GeoTIFF export; add DEM slope pruning in `pipeline.run_match`.
6. Export ONNX for CPU inference (`train.export_onnx`).

## Known TODOs (honest list)
Real PDS/IIRS QUB parsing, Hapke shader, thin-plate-spline fitting, LoFTR loss, ONNX wrapper, auth, persistent job store.
