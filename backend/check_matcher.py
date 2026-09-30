"""Diagnose the 'stuck at Learned matcher' problem. Run from backend/:  python check_matcher.py"""
import time, cv2, numpy as np, torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
t = time.time()
import pipeline
print("1) loading model (downloads weights the first time)...")
pipeline.get_matcher()
print("   loaded in %.1fs" % (time.time() - t))
a = cv2.imread("test_A.png", 0).astype("float32") / 255
b = cv2.imread("test_B.png", 0).astype("float32") / 255
t = time.time()
p, q, c = pipeline._loftr(a, b, 512)
print("2) matching 512px: %d matches in %.1fs" % (len(p), time.time() - t))
print("OK - the matcher works. If the UI still hangs, restart uvicorn and check its terminal logs.")
