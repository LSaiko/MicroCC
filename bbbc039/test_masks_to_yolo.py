"""Run: python test_masks_to_yolo.py"""
import numpy as np
from masks_to_yolo import mask_to_yolo_lines, instances

# instance-labelled mask (>4 distinct high-valued labels): one box per label
m = np.zeros((100, 100), np.uint16)
m[10:20, 10:30] = 7        # 20w x 10h, centre (20,15)
for i, v in enumerate((21, 40, 55, 88, 120)):
    m[60:64, 5 + i * 15: 9 + i * 15] = v
lines = mask_to_yolo_lines(m, min_area=4, pad=0)
assert len(lines) == 6, lines
cls, cx, cy, w, h = lines[0].split()
assert cls == "0"
assert abs(float(cx) - 0.20) < 1e-6 and abs(float(cy) - 0.15) < 1e-6, lines[0]
assert abs(float(w) - 0.20) < 1e-6 and abs(float(h) - 0.10) < 1e-6, lines[0]

# empty mask -> no lines
assert mask_to_yolo_lines(np.zeros((50, 50), np.uint8)) == []

# binary mask falls back to connected components
b = np.zeros((50, 50), np.uint8)
b[5:10, 5:10] = 255
b[30:40, 30:40] = 255
assert len(list(instances(b))) == 2

# BBBC039 semantic mask: interior=1, boundary=2; interior split gives instances
sem = np.zeros((60, 60), np.uint8)
sem[10:20, 10:20] = 1; sem[9, 9:21] = 2      # one nucleus + boundary ring pixel
sem[30:45, 30:50] = 1; sem[29, 30:51] = 2    # another
assert len(mask_to_yolo_lines(sem)) == 2

# RGBA semantic mask (label in one colour channel) is handled
rgba = np.zeros((60, 60, 4), np.uint8); rgba[..., 3] = 255
rgba[10:20, 10:20, 2] = 1
rgba[30:45, 30:50, 2] = 1
assert len(mask_to_yolo_lines(rgba)) == 2

# min_area filters stray pixels
noisy = np.zeros((50, 50), np.uint16)
noisy[0, 0] = 3
assert mask_to_yolo_lines(noisy, min_area=4) == []

print("ok")
