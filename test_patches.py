import os
import shutil
import re
import argparse

# --------------------------------------------------
# Parse YOLO crop filename
# Example:
# fe61ad0161113db6541d2eb036a643ed_L_MLO_0.jpg
# --------------------------------------------------
def parse_crop_name(fname):
    """
    return: patient_id, side, view, idx
    """
    name = fname.replace(".jpg", "")
    m = re.match(r"(.*)_([LR])_(CC|MLO)_(\d+)", name)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3), m.group(4)


def prepare_cmcnet_patches(yolo_crop_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    classes = os.listdir(yolo_crop_dir)
    total = 0

    for cls in classes:
        cls_dir = os.path.join(yolo_crop_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        for fname in os.listdir(cls_dir):
            if not fname.endswith(".jpg"):
                continue

            parsed = parse_crop_name(fname)
            if parsed is None:
                continue

            patient_id, side, view, idx = parsed
            pid = f"{patient_id}_{side}"

            save_dir = os.path.join(out_dir, pid)
            os.makedirs(save_dir, exist_ok=True)

            src = os.path.join(cls_dir, fname)
            dst = os.path.join(
                save_dir,
                f"{patient_id}_{side}_{view}_pred{idx}_yolo{idx}.png"
            )

            shutil.copy(src, dst)
            total += 1

    print(f"✅ Done. Copied {total} patches to {out_dir}")


# --------------------------------------------------
# CLI
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare YOLO cropped patches for CMCNet inference"
    )
    parser.add_argument(
        "--yolo-crop-dir",
        type=str,
        required=True,
        help="Path to YOLO crops directory, e.g. runs/detect/exp/crops"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="cmcnet_patches",
        help="Output directory for CMCNet inference"
    )

    args = parser.parse_args()

    prepare_cmcnet_patches(
        yolo_crop_dir=args.yolo_crop_dir,
        out_dir=args.out_dir
    )
