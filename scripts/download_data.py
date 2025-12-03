import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, dest_path):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def prepare_nomad():
    print("Preparing NOMAD dataset...")
    # Placeholder for NOMAD download logic
    # Since NOMAD is on GitHub, we might need to clone or download specific releases
    # For now, we'll just print instructions as direct download links might vary
    print("Please download the NOMAD dataset from https://github.com/ArtRuss/NOMAD")
    print("Extract it to data/nomad/")

def prepare_vtsar():
    print("Preparing VTSaR dataset...")
    # Placeholder for VTSaR download logic
    print("Please download VTSaR from https://github.com/zxq309/VTSaR")
    print("Extract it to data/vtsar/")

def prepare_visdrone():
    print("Preparing VisDrone dataset...")
    print("VisDrone requires registration at http://aiskyeye.com/")
    print("Please download the dataset manually and extract it to data/visdrone/")

if __name__ == "__main__":
    os.makedirs("data/nomad", exist_ok=True)
    os.makedirs("data/visdrone", exist_ok=True)
    os.makedirs("data/vtsar", exist_ok=True)
    
    prepare_nomad()
    prepare_visdrone()
    prepare_vtsar()
