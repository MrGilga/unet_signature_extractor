#!/usr/bin/env python3
"""
split_dataset.py
----------------
Copia (o crea symlink) dei file di   dataset/augmented_sig   e   dataset/clean_sig
in una struttura compatibile con lo script di training:

data/
  ├── train/
  │   ├── input/   (augmented)   ─┐
  │   └── target/  (clean)       ─┘  stessi nomi file
  ├── val/
  │   ├── input/
  │   └── target/
  └── test/
      ├── input/
      └── target/

Uso:
    python split_dataset.py            # 80-10-10 default
    python split_dataset.py --train 0.7 --val 0.2 --test 0.1
"""

import argparse
import random
import shutil
from pathlib import Path

# ------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src_aug",  type=str, default="dataset/augmented_sig")
    p.add_argument("--src_clean", type=str, default="dataset/clean_sig")
    p.add_argument("--dst",      type=str, default="data")
    p.add_argument("--train",    type=float, default=0.8, help="fraction for training")
    p.add_argument("--val",      type=float, default=0.1, help="fraction for validation")
    p.add_argument("--test",     type=float, default=0.1, help="fraction for testing")
    p.add_argument("--link",     action="store_true", help="use symlinks instead of copy")
    return p.parse_args()

# ------------------------------------------------------------
def main() -> None:
    args = parse_args()

    src_aug   = Path(args.src_aug)
    src_clean = Path(args.src_clean)
    dst_root  = Path(args.dst)

    if not (src_aug.exists() and src_clean.exists()):
        raise FileNotFoundError("Cartelle di origine non trovate")

    # Trova file comuni (stessi nomi) e con estensione immagine
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = sorted(
        f.name for f in src_aug.iterdir()
        if f.suffix.lower() in exts and (src_clean / f.name).exists()
    )
    if not files:
        raise RuntimeError("Nessun file corrispondente trovato")

    random.shuffle(files)
    n = len(files)
    n_train = int(n * args.train)
    n_val   = int(n * args.val)
    n_test  = n - n_train - n_val

    split_map = {
        "train": files[:n_train],
        "val":   files[n_train:n_train + n_val],
        "test":  files[n_train + n_val:],
    }

    # Copia o link
    for split, fnames in split_map.items():
        for sub in ("input", "target"):
            (dst_root / split / sub).mkdir(parents=True, exist_ok=True)

        for name in fnames:
            dst_in  = dst_root / split / "input"  / name
            dst_gt  = dst_root / split / "target" / name
            src_in  = src_aug   / name
            src_gt  = src_clean / name

            if args.link:
                dst_in.symlink_to(src_in.resolve())
                dst_gt.symlink_to(src_gt.resolve())
            else:
                shutil.copy2(src_in, dst_in)
                shutil.copy2(src_gt, dst_gt)

    # Report
    for split, fnames in split_map.items():
        print(f"{split:5s}: {len(fnames):6d} immagini")

    print(f"\nDataset pronto in  →  {dst_root.resolve()}")

if __name__ == "__main__":
    main()
