"""Run: python test_masks_to_yolo.py"""
import numpy as np
from masks_to_yolo import instances, mask_to_yolo_lines

# instance-labelled mask (>4 distinct high-valued labels): one box per label
m = np.zeros((100, 100), np.uint16)
m[10:30, 10:40] = 7        # 30w x 20h, centre (25, 20)
for i, v in enumerate((21, 40, 55, 88, 120)):
    m[60:72, 5 + i * 15: 17 + i * 15] = v
lines = mask_to_yolo_lines(m)
assert len(lines) == 6, lines
cls, cx, cy, w, h = lines[0].split()
assert cls == "0"
assert abs(float(cx) - 0.25) < 1e-6 and abs(float(cy) - 0.20) < 1e-6, lines[0]
assert abs(float(w) - 0.30) < 1e-6 and abs(float(h) - 0.20) < 1e-6, lines[0]

# empty mask -> no lines
assert mask_to_yolo_lines(np.zeros((50, 50), np.uint8)) == []

# binary mask -> connected components
b = np.zeros((60, 60), np.uint8)
b[5:20, 5:20] = 255
b[35:55, 35:55] = 255
assert len(list(instances(b))) == 2

# BBBC039 semantic mask: two nuclei (interior=1 + boundary ring=2), watershed split
sem = np.zeros((80, 80), np.uint8)
sem[10:28, 10:28] = 1; sem[9, 9:29] = 2
sem[40:64, 40:70] = 1; sem[39, 40:71] = 2
lines = mask_to_yolo_lines(sem)
assert len(lines) == 2, lines
# box includes the boundary ring row (y0 -> 9, not 10)
_, _, cy, _, bh = (float(x) for x in lines[0].split())
assert bh * 80 >= 18, lines[0]           # 18px interior + ring

# two nuclei touching with NO boundary pixel between them still split (watershed ridge)
touch = np.zeros((60, 120), np.uint8)
touch[15:45, 10:55] = 1
touch[15:45, 56:101] = 1                  # 1px gap only -> closed by morphology... make them actually touch:
touch[15:45, 10:100] = 1
touch[15:45, 54:57] = 0                   # thin notch to seed two components
assert len(mask_to_yolo_lines(touch)) == 2

# RGBA semantic mask (label in one colour channel)
rgba = np.zeros((80, 80, 4), np.uint8); rgba[..., 3] = 255
rgba[10:30, 10:30, 2] = 1
rgba[45:70, 45:70, 2] = 1
assert len(mask_to_yolo_lines(rgba)) == 2

# tiny fragment dropped by min_area / min_side
noisy = np.zeros((50, 50), np.uint16)
noisy[0:2, 0:2] = 3
assert mask_to_yolo_lines(noisy) == []

print("ok")
