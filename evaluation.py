import os
import argparse
import numpy as np
import pandas as pd
import torch
import math
import re
from collections import defaultdict

# YOLO official metric utils
from utils.metrics import ap_per_class, box_iou


# ==================================================
# Utils
# ==================================================
def parse_patch_name(name):
    """
    012e0595adba5173b6e60a97f9e84b6e_L_CC_3.jpg
    -> image_id: 012e0595adba5173b6e60a97f9e84b6e_L_CC
    """
    name = name.replace(".jpg", "").replace(".png", "")
    m = re.match(r"(.*)_([LR])_(CC|MLO)_\d+", name)
    if not m:
        return None
    return f"{m.group(1)}_{m.group(2)}_{m.group(3)}"


def xywh2xyxy(x):
    xc, yc, w, h = x
    return [
        xc - w / 2,
        yc - h / 2,
        xc + w / 2,
        yc + h / 2
    ]


# ==================================================
# MAIN
# ==================================================
def evaluate(gt_dir, yolo_pred_dir, siamese_csv):

    df = pd.read_csv(siamese_csv)

    # stats for YOLO ap_per_class
    stats = []

    # IoU thresholds (YOLO / COCO)
    iouv = np.linspace(0.5, 0.95, 10)

    # ------------------------------------------------
    # Load GT
    # ------------------------------------------------
    gt_by_image = defaultdict(list)

    for fname in os.listdir(gt_dir):
        if not fname.endswith(".txt"):
            continue
        img_id = fname.replace(".txt", "")
        with open(os.path.join(gt_dir, fname)) as f:
            line = f.readline().strip()
        cls, xc, yc, w, h = map(float, line.split())
        gt_by_image[img_id].append({
            "cls": int(cls),
            "bbox": [xc, yc, w, h]
        })

    print(f"✅ Loaded GT for {len(gt_by_image)} images")

    # ------------------------------------------------
    # Process Siamese + YOLO predictions
    # ------------------------------------------------
    for _, row in df.iterrows():

        for patch_col, cls_col in [
            ("CC_patch", "cc_class"),
            ("MLO_patch", "mlo_class")
        ]:
            pred_cls = int(row[cls_col])
            if pred_cls not in [0, 1]:
                continue

            patch_name = row[patch_col]
            image_id = parse_patch_name(patch_name)
            if image_id is None:
                continue

            # YOLO prediction txt
            txt_path = os.path.join(
                yolo_pred_dir,
                patch_name.replace(".jpg", ".txt").replace(".png", ".txt")
            )
            if not os.path.exists(txt_path):
                continue

            with open(txt_path) as f:
                line = f.readline().strip()

            _, xc, yc, w, h, conf = map(float, line.split())

            # Siamese fusion
            dist = float(row["distance"])
            match_score = math.exp(-dist)
            final_conf = conf * match_score

            # ------------------------------------------------
            # Build YOLO-style tensors
            # ------------------------------------------------
            pred_xyxy = xywh2xyxy([xc, yc, w, h])
            pred = torch.tensor([[*pred_xyxy, final_conf, pred_cls]])

            if image_id not in gt_by_image:
                continue

            gt = gt_by_image[image_id][0]
            gt_xyxy = xywh2xyxy(gt["bbox"])
            label = torch.tensor([[gt["cls"], *gt_xyxy]])

            # ------------------------------------------------
            # IoU & correctness matrix
            # ------------------------------------------------
            correct = np.zeros((1, len(iouv)), dtype=bool)

            iou = box_iou(
                torch.tensor([pred_xyxy]),
                torch.tensor([gt_xyxy])
            )[0, 0].item()

            for j, thr in enumerate(iouv):
                if iou >= thr:
                    correct[0, j] = True

            # ------------------------------------------------
            # Collect stats (YOLO format)
            # ------------------------------------------------
            stats.append((
                correct,
                np.array([final_conf]),
                np.array([pred_cls]),
                np.array([gt["cls"]])
            ))

    # ------------------------------------------------
    # Final evaluation
    # ------------------------------------------------
    if len(stats) == 0:
        print("❌ No valid predictions found.")
        return

    stats = [np.concatenate(x, 0) for x in zip(*stats)]

    tp, fp, p, r, ap50, ap, ap_class = ap_per_class(*stats)

    print("\n==============================")
    print("  FINAL YOLO-STYLE EVALUATION ")
    print("==============================\n")

    print(f"P        : {p.mean():.4f}")
    print(f"R        : {r.mean():.4f}")
    print(f"mAP@0.5  : {ap[:, 0].mean():.4f}")
    print(f"mAP@0.5:0.95 : {ap.mean():.4f}")

    print("\nPer-class AP:")
    for i, c in enumerate(ap_class):
        name = "Mass" if c == 0 else "Suspicious_Calcification"
        print(f"{name:<25} AP@0.5={ap[i,0]:.4f}  AP@0.5:0.95={ap[i].mean():.4f}")


# ==================================================
# CLI
# ==================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Siamese + YOLO Evaluation (YOLO-aligned)")
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--yolo_pred_dir", type=str, required=True)
    parser.add_argument("--siamese_csv", type=str, required=True)

    args = parser.parse_args()

    evaluate(
        gt_dir=args.gt_dir,
        yolo_pred_dir=args.yolo_pred_dir,
        siamese_csv=args.siamese_csv
    )
