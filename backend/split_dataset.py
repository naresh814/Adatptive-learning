import os, shutil, random

source = "../dataset/fer2013/train"
dest = "../dataset/fer2013/test"

for cls in os.listdir(source):
    os.makedirs(os.path.join(dest, cls), exist_ok=True)

    images = os.listdir(os.path.join(source, cls))
    random.shuffle(images)

    split = int(0.2 * len(images))

    for img in images[:split]:
        shutil.copy(
            os.path.join(source, cls, img),
            os.path.join(dest, cls, img)
        )

print("✅ Dataset split completed")
