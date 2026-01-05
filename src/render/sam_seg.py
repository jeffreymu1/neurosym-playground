import os
import json
import argparse
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

r'''
cli:

python sam_seg.py `
  --rgb_dir  "..\..\data\output\725\rgb" `
  --out_dir  "..\..\data\output\725sam" `
  --sam_ckpt "..\..\checkpoints\sam_vit_h_4b8939.pth" `
  --points_per_side 16 --device cpu --topk 10 --rank "quality_area"

'''

# when sorting masks, use this as a cli argument
def sort_masks(masks, mode: str):

    mode = mode.lower()
    if mode == "area":
        return sorted(masks, key=lambda d: d.get("area", 0), reverse=True)

    if mode == "quality":
        return sorted(masks,key=lambda d: (
                float(d.get("predicted_iou", 0.0)),
                float(d.get("stability_score", 0.0)),
                int(d.get("area", 0))), 
                reverse=True)

    if mode == "quality_area":
        # assign an arbitrary (small) area weight
        area_weight = 0.05
        def score(d):
            piou = float(d.get("predicted_iou", 0.0))
            stab = float(d.get("stability_score", 0.0))
            area = float(d.get("area", 0.0))
            return piou + 0.5 * stab + area_weight * (area ** 0.5)

        return sorted(masks, key=score, reverse=True)

    raise ValueError(f"unknown --rank '{mode}'")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--model_type", default="vit_h", choices=["vit_h","vit_l","vit_b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--points_per_side", type=int, default=32)
    ap.add_argument("--pred_iou_thresh", type=float, default=0.88)
    ap.add_argument("--stability_score_thresh", type=float, default=0.95)
    ap.add_argument("--min_mask_region_area", type=int, default=200)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--rank", default="quality", choices=["area", "quality", "quality_area"])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_ckpt)
    sam.to(device=args.device)

    gen = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area)

    imgs = sorted([f for f in os.listdir(args.rgb_dir) if f.lower().endswith(".png")])

    for fn in imgs:
        stem = os.path.splitext(fn)[0]
        img_path = os.path.join(args.rgb_dir, fn)
        img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print("skipping, couldn't read", img_path)
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        masks = gen.generate(img_rgb)

        # sort by the argument given
        masks = sort_masks(masks, args.rank)[: args.topk]

        view_dir = os.path.join(args.out_dir, f"view_{stem}")
        mask_dir = os.path.join(view_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        meta = []

        for j, m in enumerate(masks):
            seg = m["segmentation"].astype("uint8")*255
            mpath = os.path.join(mask_dir, f"mask_{j:03d}.png")
            cv2.imwrite(mpath, seg)
            meta.append({
                "mask_id": j,
                "area": int(m["area"]),
                "bbox": [int(x) for x in m["bbox"]],
                "predicted_iou": float(m.get("predicted_iou", -1.0)),
                "stability_score": float(m.get("stability_score", -1.0)),
                "mask_path": os.path.relpath(mpath, args.out_dir).replace("\\","/")})

        with open(os.path.join(view_dir, "meta.json"), "w") as f: json.dump(meta, f, indent=2)

        print(f"finished, {fn}: wrote {len(masks)} masks to {view_dir}")

if __name__ == "__main__":
    main()
