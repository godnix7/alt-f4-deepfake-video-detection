import os
from PIL import Image
from tqdm import tqdm

def clean_dataset(base_dir):
    print(f"Scanning {base_dir} for corrupted images...")
    corrupted_count = 0
    valid_count = 0
    
    for root, _, files in os.walk(base_dir):
        for file in tqdm(files, desc=f"Checking {os.path.basename(root)}"):
            file_path = os.path.join(root, file)
            
            # File size check
            if os.path.getsize(file_path) == 0:
                os.remove(file_path)
                corrupted_count += 1
                continue
                
            # PIL check
            try:
                with Image.open(file_path) as img:
                    img.verify() # verify that it is, in fact, an image
                valid_count += 1
            except Exception:
                os.remove(file_path)
                corrupted_count += 1
                
    print(f"\nScan complete. Found and deleted {corrupted_count} corrupted images.")
    print(f"Total healthy images remaining: {valid_count}")

if __name__ == "__main__":
    clean_dataset("data/train")
