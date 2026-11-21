"""
Measure real model performance on a labelled folder (State Farm layout).

Usage:
    python evaluate.py --data path/to/imgs/train --model efficientnet
    python evaluate.py --data path/to/imgs/train --model yolo --limit 200

Folder layout expected:  <data>/c0/*.jpg ... <data>/c9/*.jpg
IMPORTANT: use images from drivers NOT used in training (driver-wise split),
otherwise the numbers will be inflated. Copy your held-out drivers into a folder.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model_loader import predict_driver_state, CLASS_KEYS, CLASS_MAP

# YOLO model has 4 classes; map the 10 State Farm classes onto them
YOLO_MAP = {"c0": "safe", "c1": "phone", "c2": "phone", "c3": "phone", "c4": "phone",
            "c5": "distracted", "c6": "drink", "c7": "distracted",
            "c8": "distracted", "c9": "distracted"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="efficientnet", choices=["efficientnet", "yolo"])
    ap.add_argument("--limit", type=int, default=0, help="max images per class (0 = all)")
    args = ap.parse_args()

    if args.model == "efficientnet":
        labels = CLASS_KEYS
        truth_of = lambda c: c
        pred_of = lambda r: r["class_code"]
    else:
        labels = ["safe", "phone", "drink", "distracted"]
        truth_of = lambda c: YOLO_MAP[c]
        pred_of = lambda r: r["class_code"]

    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)

    for c in CLASS_KEYS:
        folder = os.path.join(args.data, c)
        if not os.path.isdir(folder):
            print(f"[WARN] missing folder {folder}")
            continue
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        if args.limit:
            files = files[:args.limit]
        for f in files:
            r = predict_driver_state(os.path.join(folder, f), model_type=args.model)
            cm[idx[truth_of(c)], idx[pred_of(r)]] += 1
        print(f"[INFO] {c} done ({len(files)} images)")

    total = cm.sum()
    if total == 0:
        sys.exit("No images evaluated. Check --data path.")

    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    print(f"\nModel: {args.model} | images: {total} | accuracy: {tp.sum()/total*100:.2f}%")
    print(f"{'class':<22}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
    for l in labels:
        i = idx[l]
        name = CLASS_MAP[l]["name"] if l in CLASS_MAP else l
        print(f"{name:<22}{prec[i]*100:>9.1f}%{rec[i]*100:>8.1f}%{f1[i]*100:>7.1f}%{int(cm[i].sum()):>9}")
    print(f"{'MACRO AVG':<22}{prec.mean()*100:>9.1f}%{rec.mean()*100:>8.1f}%{f1.mean()*100:>7.1f}%")
    print("\nConfusion matrix (rows = truth, cols = predicted):")
    print("      " + " ".join(f"{l:>6}" for l in labels))
    for l in labels:
        print(f"{l:>6}" + " ".join(f"{v:>6}" for v in cm[idx[l]]))

    out = f"eval_{args.model}.json"
    json.dump({"accuracy": float(tp.sum() / total), "labels": labels,
               "precision": prec.tolist(), "recall": rec.tolist(), "f1": f1.tolist(),
               "confusion_matrix": cm.tolist()}, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
