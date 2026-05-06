"""
dataset_stats.py — Shows class distribution and imbalance info
"""
import os

dataset_path = "../dataset/fer2013/train"

if not os.path.exists(dataset_path):
    print(f"[WARNING] Path not found: {dataset_path}")
else:
    total = 0
    counts = {}
    for emotion in sorted(os.listdir(dataset_path)):
        p = os.path.join(dataset_path, emotion)
        if os.path.isdir(p):
            n = len(os.listdir(p))
            counts[emotion] = n
            total += n

    print("Emotion Distribution")
    print("─" * 35)
    for emo, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * (cnt // 200)
        print(f"{emo:12s}: {cnt:5d}  {bar}")
    print("─" * 35)
    print(f"{'Total':12s}: {total:5d}")
    print()
    print("Class imbalance ratio:", round(max(counts.values()) / min(counts.values()), 1), "x")
    print("→ Use class_weight='balanced' in training to fix this.")
