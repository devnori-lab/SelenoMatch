"""Create a test image pair (no ISRO data needed): same terrain, different Sun angle + scale + small shift.
Run from backend/:  python make_test_pair.py   ->  test_A.png, test_B.png (+ prints the true transform)"""
import cv2, numpy as np
import synth

dem = synth.load_dem(n=1024, seed=1)
rng = np.random.default_rng(42)
a, b, H, m = synth.make_pair(dem, size=512, el_range=(15, 70), scale_ratios=(1, 4), rng=rng)
cv2.imwrite("test_A.png", (a * 255).astype("uint8"))
cv2.imwrite("test_B.png", (b * 255).astype("uint8"))
print("Sun A: el %.0f az %.0f | Sun B: el %.0f az %.0f | scale gap x%.0f" % (m["el0"], m["az0"], m["el1"], m["az1"], m["scale"]))
print("TRUE transform A->B (compare with the UI's estimated matrix):\n", np.round(H, 3))
