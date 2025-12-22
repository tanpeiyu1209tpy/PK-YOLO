import os
import cv2
import argparse

# -------------------------------------------------
# Utils
# -------------------------------------------------
def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):

    CLASS_MAP = {
        0: "mass",
        1: "Suspicious_Calcification"
    }

    os.makedirs(args.output_dir, exist_ok=True)

    label_files = [f for f in os.listdir(args.label_dir) if f.endswith(".txt")]

    print(f"📌 Found {len(label_files)} label files")

    for txt_name in label_files:

        base = os.path.splitext(txt_name)[0]
        img_path = os.path.join(args.img_dir, base + args.img_ext)

        if not os.path.exists(img_path):
            print(f"❌ Image not found: {img_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to read image: {img_path}")
            continue

        h, w = img.shape[:2]

        with open(os.path.join(args.label_dir, txt_name)) as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if len(lines) == 0:
            continue

        class_counter = {0: 0, 1: 0}

        for line_idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) < 5:
                continue
        
            cls, xc, yc, bw, bh = map(float, parts[:5])
            conf = float(parts[5]) if len(parts) > 5 else None
            cls = int(cls)
        
            if cls not in CLASS_MAP:
                continue
        
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            crop = img[y1:y2, x1:x2]
        
            if crop.size == 0:
                continue
        
            class_name = CLASS_MAP[cls]
        
            crop_dir  = os.path.join(args.output_dir, "crops", class_name)
            label_dir = os.path.join(args.output_dir, "labels", class_name)
        
            os.makedirs(crop_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
        
            crop_name  = f"{base}_{line_idx}.jpg"
            label_name = f"{base}_{line_idx}.txt"
        
            cv2.imwrite(os.path.join(crop_dir, crop_name), crop)
        
            with open(os.path.join(label_dir, label_name), "w") as f:
                f.write(line + "\n")

        print(f"✅ Processed {base}")


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert YOLO labels to class-wise crop folders"
    )

    parser.add_argument("--img_dir", type=str, required=True,
                        help="Directory of original images")

    parser.add_argument("--label_dir", type=str, required=True,
                        help="Directory of YOLO txt labels")

    parser.add_argument("--output_dir", type=str, default="output",
                        help="Output root directory")

    parser.add_argument("--img_ext", type=str, default=".jpg",
                        help="Image extension (.jpg or .png)")

    args = parser.parse_args()
    main(args)
