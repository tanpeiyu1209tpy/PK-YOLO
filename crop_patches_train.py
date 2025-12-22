#!/usr/bin/env python3
"""
Crop YOLO hard negative patches only.

- Negative = YOLO predictions with IoU < threshold w.r.t GT
- Top-K negatives per image (by confidence)

Positive patches are assumed to be cropped separately from GT.
"""

import argparse
from pathlib import Path

import cv2
import torch
from utils.metrics import box_iou


# --------------------------------------------------
# Argument parser
# --------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Crop YOLO hard negative patches only"
    )

    parser.add_argument("--images-dir", type=str, required=True,
                        help="Directory containing original images")
    parser.add_argument("--pred-dir", type=str, required=True,
                        help="Directory containing YOLO prediction .txt files")
    parser.add_argument("--gt-dir", type=str, required=True,
                        help="Directory containing GT YOLO label .txt files")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save negative patches")

    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="IoU threshold for negative samples")
    parser.add_argument("--topk", type=int, default=5,
                        help="Max number of negatives per image")
    parser.add_argument("--patch-size", type=int, default=128,
                        help="Output patch size")
    parser.add_argument("--img-ext", type=str, default=".png",
                        help="Image extension")

    return parser.parse_args()


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def load_gt_boxes(label_path, img_shape):
    """Load GT YOLO labels and convert to xyxy pixel format."""
    if not label_path.exists():
        return torch.empty((0, 4))

    h, w = img_shape[:2]
    boxes = []

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            _, xc, yc, bw, bh = map(float, parts[:5])
            xc *= w; yc *= h
            bw *= w; bh *= h

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append([x1, y1, x2, y2])

    if len(boxes) == 0:
        return torch.empty((0, 4))

    return torch.tensor(boxes)


def crop_patch(img, xc, yc, bw, bh, out_size):
    """Crop patch from image using normalized YOLO bbox."""
    h, w = img.shape[:2]

    xc *= w; yc *= h
    bw *= w; bh *= h

    x1 = int(xc - bw / 2)
    y1 = int(yc - bh / 2)
    x2 = int(xc + bw / 2)
    y2 = int(yc + bh / 2)

    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(w, x2); y2 = min(h, y2)

    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    return cv2.resize(patch, (out_size, out_size))


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    save_dir = Path(args.save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(pred_dir.glob("*.txt"))
    print(f"🔍 Found {len(pred_files)} YOLO prediction files")

    for pred_path in pred_files:
        image_id = pred_path.stem
        img_path = images_dir / f"{image_id}{args.img_ext}"
        gt_path = gt_dir / f"{image_id}.txt"

        if not img_path.exists():
            print(f"⚠ Missing image: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Failed to read image: {img_path}")
            continue

        gt_boxes = load_gt_boxes(gt_path, img.shape)

        neg_candidates = []

        # -------------------------------
        # Parse YOLO predictions
        # -------------------------------
        with open(pred_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                _, xc, yc, bw, bh, conf = map(float, parts)

                pred_xyxy = torch.tensor([[
                    (xc - bw / 2) * img.shape[1],
                    (yc - bh / 2) * img.shape[0],
                    (xc + bw / 2) * img.shape[1],
                    (yc + bh / 2) * img.shape[0],
                ]])

                iou = 0.0
                if gt_boxes.numel() > 0:
                    iou = box_iou(pred_xyxy, gt_boxes).max().item()

                if iou < args.iou_thr:
                    neg_candidates.append((conf, xc, yc, bw, bh))

        if len(neg_candidates) == 0:
            continue

        # Top-K by confidence
        neg_candidates.sort(key=lambda x: x[0], reverse=True)
        neg_candidates = neg_candidates[:args.topk]

        # Save negative patches
        for idx, (conf, xc, yc, bw, bh) in enumerate(neg_candidates):
            patch = crop_patch(img, xc, yc, bw, bh, args.patch_size)
            if patch is None:
                continue

            out_name = f"{image_id}_neg_{idx}.png"
            cv2.imwrite(str(save_dir / out_name), patch)

        print(f"✔ {image_id}: {len(neg_candidates)} negatives saved")

    print("✅ Hard negative cropping finished.")


if __name__ == "__main__":
    main()
