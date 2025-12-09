"""
Download VisDrone DET dataset for person detection training.
Creates data/visdrone/ with train and val subsets.
"""
import os
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

DATA_ROOT = Path("data/visdrone")

# VisDrone DET URLs (official Ultralytics mirrors)
URLS = {
    "train": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip",
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    """Download file with progress bar."""
    print(f"Downloading: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    """Extract zip file."""
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)  # Clean up zip after extraction

def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    
    for split, url in URLS.items():
        zip_name = f"VisDrone2019-DET-{split}.zip"
        zip_path = DATA_ROOT / zip_name
        
        # Check if already extracted
        extracted_dir = DATA_ROOT / f"VisDrone2019-DET-{split}"
        if extracted_dir.exists():
            print(f"[SKIP] {split} already exists at {extracted_dir}")
            continue
        
        # Download
        download_file(url, zip_path)
        
        # Extract
        extract_zip(zip_path, DATA_ROOT)
    
    print("\n✅ Download complete!")
    print(f"Data saved to: {DATA_ROOT.absolute()}")
    print("\nNext step: Run 'python scripts/prepare_visdrone.py' to filter for people-only class")

if __name__ == "__main__":
    main()
