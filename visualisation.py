import os
import re
import glob
import cv2
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# ======================================================
# Utils
# ======================================================
def parse_patch_name(name):
    """
    012e0595adba5173b6e60a97f9e84b6e_L_CC_8.jpg
    return:
        gt_image_id = 012e0595adba5173b6e60a97f9e84b6e_L_CC
    """
    name = name.replace(".jpg", "").replace(".png", "")
    m = re.match(r"(.*)_([LR])_(CC|MLO)_(\d+)", name)
    if not m:
        return None
    gt_image_id = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return gt_image_id


def find_image(image_dir, patch_name):
    """
    Robust image finder:
    - supports jpg / png
    - supports subfolders
    - supports suffix variants
    """
    stem = os.path.splitext(patch_name)[0]
    candidates = glob.glob(
        os.path.join(image_dir, "**", stem + ".*"),
        recursive=True,
    )
    return candidates[0] if candidates else None


def load_single_bbox(txt_path):
    """
    YOLO txt: cls xc yc w h conf
    (assume only ONE bbox)
    """
    with open(txt_path, "r") as f:
        line = f.readline().strip()
    cls, xc, yc, w, h, conf = map(float, line.split())
    return int(cls), [xc, yc, w, h], conf


def draw_bbox(img, bbox, label, color):
    h, w, _ = img.shape
    xc, yc, bw, bh = bbox
    x1 = int((xc - bw / 2) * w)
    y1 = int((yc - bh / 2) * h)
    x2 = int((xc + bw / 2) * w)
    y2 = int((yc + bh / 2) * h)

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        img,
        label,
        (x1, max(20, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def show_missing(ax, title):
    ax.set_title(title, color="red")
    ax.axis("off")


# ======================================================
# Visualization per view (3 images)
# ======================================================
def visualize_one_view(ax_row, patch_name, distance, image_dir, gt_dir, yolo_pred_dir):
    """
    ax_row = [ax_original, ax_gt, ax_pred]
    """

    # ---------- find image ----------
    img_path = find_image(image_dir, patch_name)
    if img_path is None:
        show_missing(ax_row[0], "Image NOT FOUND")
        show_missing(ax_row[1], "GT skipped")
        show_missing(ax_row[2], "Pred skipped")
        return

    img = cv2.imread(img_path)
    if img is None:
        show_missing(ax_row[0], "Image load FAILED")
        show_missing(ax_row[1], "GT skipped")
        show_missing(ax_row[2], "Pred skipped")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ---------- parse GT ----------
    parsed = parse_patch_name(patch_name)
    if parsed is None:
        show_missing(ax_row[0], "Parse error")
        show_missing(ax_row[1], "GT skipped")
        show_missing(ax_row[2], "Pred skipped")
        return

    gt_txt = os.path.join(gt_dir, f"{parsed}.txt")
    pred_txt = os.path.join(
        yolo_pred_dir,
        patch_name.replace(".jpg", ".txt").replace(".png", ".txt"),
    )

    # ---------- Original ----------
    ax_row[0].imshow(img)
    ax_row[0].set_title("Original")
    ax_row[0].axis("off")

    # ---------- GT ----------
    if os.path.exists(gt_txt):
        img_gt = img.copy()
        gt_cls, gt_bbox, _ = load_single_bbox(gt_txt)
        draw_bbox(img_gt, gt_bbox, f"GT:{gt_cls}", (0, 255, 0))
        ax_row[1].imshow(img_gt)
        ax_row[1].set_title("GT")
    else:
        show_missing(ax_row[1], "GT NOT FOUND")

    ax_row[1].axis("off")

    # ---------- YOLO + Siamese ----------
    if os.path.exists(pred_txt):
        img_pred = img.copy()
        y_cls, y_bbox, y_conf = load_single_bbox(pred_txt)
        draw_bbox(
            img_pred,
            y_bbox,
            f"YOLO:{y_cls} ({y_conf:.2f})",
            (255, 0, 0),
        )
        ax_row[2].imshow(img_pred)
        ax_row[2].set_title(f"YOLO + Siamese\n(dist={distance:.3f})")
    else:
        show_missing(ax_row[2], "YOLO pred NOT FOUND")

    ax_row[2].axis("off")


# ======================================================
# Main
# ======================================================
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.siamese_csv)

    if args.max_cases > 0:
        df = df.head(args.max_cases)

    for idx, row in df.iterrows():
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))

        # CC row
        visualize_one_view(
            axes[0],
            row["CC_patch"],
            row["distance"],
            args.image_dir,
            args.gt_dir,
            args.yolo_pred_dir,
        )

        # MLO row
        visualize_one_view(
            axes[1],
            row["MLO_patch"],
            row["distance"],
            args.image_dir,
            args.gt_dir,
            args.yolo_pred_dir,
        )

        patient_id = row["CC_patch"].split("_")[0]
        out_path = os.path.join(args.out_dir, f"{idx:04d}_{patient_id}.png")

        plt.suptitle(
            f"Patient {patient_id} | Siamese distance = {row['distance']:.3f}",
            fontsize=14,
        )

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

        print(f"✅ Saved {out_path}")


# ======================================================
# CLI
# ======================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize CC & MLO with GT, YOLO, and Siamese distance (6 images)"
    )

    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--yolo_pred_dir", type=str, required=True)
    parser.add_argument("--siamese_csv", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="vis_results")

    args = parser.parse_args()
    main(args)
