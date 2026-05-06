"""
dataset_loader.py — Lists available emotion classes from FER-2013 dataset
"""
import os

dataset_path = "../dataset/fer2013/train"

if os.path.exists(dataset_path):
    emotions = sorted(os.listdir(dataset_path))
    print("Available emotion classes:")
    for e in emotions:
        print(f"  {e}")
else:
    print(f"[WARNING] Dataset path not found: {dataset_path}")
    print("  Place FER-2013 dataset at: dataset/fer2013/train/<emotion>/")
