import os
import cv2
import argparse
import numpy as np

# -----------------------------
# Utils
# -----------------------------
def parse_image_name(fname):
    """
    fe61ad0161113db6541d2eb036a643ed_L_MLO.png
    → (patient_id, side, view)
    """
    base = os.path.basename(fname).replace(".png", "")
    parts = base.split("_")
    if len(parts) < 3:
        return None
    patient_id = parts[0]
    side = parts[1]
    view = parts[2]   # CC or MLO
    return patient_id, side, view


def load_yolo_labels(txt_path):
    """
    YOLO format:
    class cx cy w h conf
    """
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            vals = list(map(float, line.strip().split()))
            if len(vals) < 5:
                continue
            boxes.append(vals)
    return boxes


# -----------------------------
# Main crop logic
# -----------------------------
def crop_yolo_patches(
    yolo_img_dir,
    yolo_label_dir,
    output_dir,
    patch_size=128
):
    os.makedirs(output_dir, exist_ok=True)

    images = [f for f in os.listdir(yolo_img_dir) if f.endswith(".png")]

    print(f"📌 Found {len(images)} YOLO images")

    for img_name in images:
        img_path = os.path.join(yolo_img_dir, img_name)
        label_path = os.path.join(
            yolo_label_dir, img_name.replace(".png", ".txt")
        )

        if not os.path.exists(label_path):
            continue

        parsed = parse_image_name(img_name)
        if parsed is None:
            continue

        patient_id, side, view = parsed
        pid = f"{patient_id}_{side}"

        save_dir = os.path.join(output_dir, pid)
        os.makedirs(save_dir, exist_ok=True)

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]
        boxes = load_yolo_labels(label_path)

        for idx, box in enumerate(boxes):
            cls, cx, cy, bw, bh = box[:5]

            # YOLO normalized → pixel
            x1 = int((cx - bw / 2) * W)
            y1 = int((cy - bh / 2) * H)
            x2 = int((cx + bw / 2) * W)
            y2 = int((cy + bh / 2) * H)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            patch = img[y1:y2, x1:x2]
            patch = cv2.resize(patch, (patch_size, patch_size))

            out_name = (
                f"{patient_id}_{side}_{view}_pred{idx}_yolo{idx}.png"
            )
            out_path = os.path.join(save_dir, out_name)

            cv2.imwrite(out_path, patch)

        print(f"✔ Processed {img_name}")

    print("🎯 Patch cropping finished.")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("YOLO → CMCNet patch crop")
    parser.add_argument("--yolo_img_dir", required=True)
    parser.add_argument("--yolo_label_dir", required=True)
    parser.add_argument("--out_dir", default="cmcnet_patches")
    parser.add_argument("--patch_size", type=int, default=128)

    args = parser.parse_args()

    crop_yolo_patches(
        yolo_img_dir=args.yolo_img_dir,
        yolo_label_dir=args.yolo_label_dir,
        output_dir=args.out_dir,
        patch_size=args.patch_size
    )
