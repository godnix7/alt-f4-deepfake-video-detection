import kagglehub
import os
import shutil

# Target directory in current folder
target_folder = os.path.join(os.getcwd(), "data")
os.makedirs(target_folder, exist_ok=True)

print(f"Downloading dataset... It will be moved to: {target_folder}")

# Download to cache first (most stable way)
cache_path = kagglehub.dataset_download("sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset")

# Move to local folder
print("\nDownload finished! Moving files to Fusion X/data...")
for item in os.listdir(cache_path):
    s = os.path.join(cache_path, item)
    d = os.path.join(target_folder, item)
    if os.path.exists(d):
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            os.remove(d)
    shutil.move(s, d)

print(f"Success! Data is now in: {target_folder}")
