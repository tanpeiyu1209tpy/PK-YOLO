#!/usr/bin/env python3
"""
Construct patch-level dataset.

Positive:
- From GT boxes only
- Random jitter around GT
- IoU with GT > 0.5
- Randomly sample top-K
- Saved into Mass / Suspicious_Calcification

Negative:
- From YOLO predictions
- IoU < threshold w.r.t all GT
- Top-K by confidence
- Saved into Negative
"""

import argparse
import random
from pathlib import Path

import cv2
import torch
from utils.metrics import box_iou


# --------------------------------------------------
# Args
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--save-dir", required=True)

    parser.add_argument("--pos-topk", type=int, default=2)
    parser.add_argument("--neg-topk", type=int, default=5)

    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--img-ext", default=".png")

    return parser.parse_args()


# --------------------------------------------------
# Utils
# --------------------------------------------------
def load_gt_boxes(gt_path, img_shape):
    h, w = img_shape[:2]
    boxes = []

    if not gt_path.exists():
        return boxes

    with open(gt_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())
            xc *= w; yc *= h
            bw *= w; bh *= h
            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2
            boxes.append((int(cls), x1, y1, x2, y2))

    return boxes


def jitter_box(x1, y1, x2, y2, img_w, img_h, scale=0.15):
    w = x2 - x1
    h = y2 - y1

    dx = random.uniform(-scale, scale) * w
    dy = random.uniform(-scale, scale) * h

    nx1 = max(0, x1 + dx)
    ny1 = max(0, y1 + dy)
    nx2 = min(img_w, x2 + dx)
    ny2 = min(img_h, y2 + dy)

    return nx1, ny1, nx2, ny2


def crop(img, x1, y1, x2, y2, size):
    patch = img[int(y1):int(y2), int(x1):int(x2)]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    save_dir = Path(args.save_dir)

    mass_dir = save_dir / "Mass"
    calc_dir = save_dir / "Suspicious_Calcification"
    neg_dir = save_dir / "Negative"
    for d in [mass_dir, calc_dir, neg_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for gt_path in gt_dir.glob("*.txt"):
        image_id = gt_path.stem
        img_path = images_dir / f"{image_id}{args.img_ext}"

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        gt_boxes = load_gt_boxes(gt_path, img.shape)

        # -------------------------
        # POSITIVE (from GT)
        # -------------------------
        pos_patches = []

        for cls, x1, y1, x2, y2 in gt_boxes:
            gt_tensor = torch.tensor([[x1, y1, x2, y2]])

            for _ in range(10):  # generate candidates
                nx1, ny1, nx2, ny2 = jitter_box(x1, y1, x2, y2, w, h)
                prop = torch.tensor([[nx1, ny1, nx2, ny2]])
                iou = box_iou(prop, gt_tensor).item()

                if iou > args.iou_thr:
                    pos_patches.append((cls, nx1, ny1, nx2, ny2))

        random.shuffle(pos_patches)
        pos_patches = pos_patches[:args.pos_topk]

        for i, (cls, x1, y1, x2, y2) in enumerate(pos_patches):
            patch = crop(img, x1, y1, x2, y2, args.patch_size)
            if patch is None:
                continue
            out_dir = mass_dir if cls == 0 else calc_dir
            cv2.imwrite(str(out_dir / f"{image_id}_pos_{i}.png"), patch)

        # -------------------------
        # NEGATIVE (from YOLO)
        # -------------------------
        neg_candidates = []

        pred_path = pred_dir / f"{image_id}.txt"
        if pred_path.exists():
            gt_xyxy = torch.tensor([[b[1], b[2], b[3], b[4]] for b in gt_boxes]) \
                if gt_boxes else torch.empty((0, 4))

            with open(pred_path) as f:
                for line in f:
                    _, xc, yc, bw, bh, conf = map(float, line.split())
                    x1 = (xc - bw/2) * w
                    y1 = (yc - bh/2) * h
                    x2 = (xc + bw/2) * w
                    y2 = (yc + bh/2) * h

                    prop = torch.tensor([[x1, y1, x2, y2]])
                    iou = box_iou(prop, gt_xyxy).max().item() if gt_xyxy.numel() else 0

                    if iou < args.iou_thr:
                        neg_candidates.append((conf, x1, y1, x2, y2))

        neg_candidates.sort(reverse=True)
        neg_candidates = neg_candidates[:args.neg_topk]

        for i, (_, x1, y1, x2, y2) in enumerate(neg_candidates):
            patch = crop(img, x1, y1, x2, y2, args.patch_size)
            if patch is not None:
                cv2.imwrite(str(neg_dir / f"{image_id}_neg_{i}.png"), patch)

        print(f"✔ {image_id}: {len(pos_patches)} pos, {len(neg_candidates)} neg")

    print("✅ Dataset construction finished.")


if __name__ == "__main__":
    main()
