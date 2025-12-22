import os
import shutil
import re

YOLO_CROP_DIR = "runs/detect/exp/crops"
OUT_DIR = "cmcnet_patches"

os.makedirs(OUT_DIR, exist_ok=True)

pattern = r"(.*)_([LR])_(CC|MLO)_(\d+)\.jpg"

for cls in os.listdir(YOLO_CROP_DIR):
    cls_dir = os.path.join(YOLO_CROP_DIR, cls)
    if not os.path.isdir(cls_dir):
        continue

    for fname in os.listdir(cls_dir):
        m = re.match(pattern, fname)
        if not m:
            continue

        patient, side, view, idx = m.groups()
        pid = f"{patient}_{side}"

        save_dir = os.path.join(OUT_DIR, pid)
        os.makedirs(save_dir, exist_ok=True)

        src = os.path.join(cls_dir, fname)
        dst = os.path.join(
            save_dir,
            f"{patient}_{side}_{view}_pred{idx}_yolo{idx}.png"
        )

        shutil.copy(src, dst)
