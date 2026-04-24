"""
Download datasets for the Safeguard Studio project.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --dataset jigsaw
  python scripts/download_data.py --dataset hatexplain
  python scripts/download_data.py --dataset hh-rlhf
"""
import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Error: 'datasets' package not installed. Run: pip install datasets")
    exit(1)

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

def download_jigsaw():
    print("\n📥 Downloading Jigsaw Toxic Comment Classification...")
    dest = DATA_DIR / "jigsaw"
    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("jigsaw_toxicity_pred", trust_remote_code=True)
    for split_name, split_data in ds.items():
        path = dest / f"{split_name}.parquet"
        split_data.to_parquet(str(path))
        print(f"  ✓ Saved {split_name} → {path} ({len(split_data):,} rows)")

def download_hatexplain():
    print("\n📥 Downloading HateXplain...")
    dest = DATA_DIR / "hatexplain"
    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("hatexplain", trust_remote_code=True)
    for split_name, split_data in ds.items():
        path = dest / f"{split_name}.parquet"
        split_data.to_parquet(str(path))
        print(f"  ✓ Saved {split_name} → {path} ({len(split_data):,} rows)")

def download_hh_rlhf():
    print("\n📥 Downloading Anthropic HH-RLHF (harmless-base)...")
    dest = DATA_DIR / "hh-rlhf"
    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", trust_remote_code=True)
    for split_name, split_data in ds.items():
        path = dest / f"harmless_{split_name}.parquet"
        split_data.to_parquet(str(path))
        print(f"  ✓ Saved {split_name} → {path} ({len(split_data):,} rows)")

DATASETS = {"jigsaw": download_jigsaw, "hatexplain": download_hatexplain, "hh-rlhf": download_hh_rlhf}

def main():
    parser = argparse.ArgumentParser(description="Download datasets for Safeguard Studio")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download a specific dataset (default: all)")
    args = parser.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR.resolve()}")
    if args.dataset:
        DATASETS[args.dataset]()
    else:
        for name, func in DATASETS.items():
            try:
                func()
            except Exception as e:
                print(f"\n⚠️  Failed to download {name}: {e}")
                print("  Retry with: python scripts/download_data.py --dataset", name)
    print("\n✅ Done! Datasets saved to data/raw/")

if __name__ == "__main__":
    main()
