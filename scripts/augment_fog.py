"""
Fog Augmentation Script for VisDrone Person Detection Dataset.

Uses Monodepth2 for realistic depth estimation and atmospheric scattering model for fog synthesis.

Usage:
    python scripts/augment_fog.py --test        # Test on 5 sample images
    python scripts/augment_fog.py               # Run on full dataset (35%)
"""
import os
import sys
import random
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import cv2

import torch
from torchvision import transforms

import subprocess
import urllib.request
import zipfile

# Add monodepth2 to path
SCRIPT_DIR = Path(__file__).parent
MONODEPTH_DIR = SCRIPT_DIR / "monodepth2"

def setup_monodepth():
    """Ensure Monodepth2 repo and model are available."""
    # 1. Clone Repository if missing
    if not MONODEPTH_DIR.exists():
        print("Mondepth2 not found. Cloning repository...")
        try:
            subprocess.check_call([
                "git", "clone", 
                "https://github.com/nianticlabs/monodepth2.git", 
                str(MONODEPTH_DIR)
            ])
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone monodepth2: {e}")
            sys.exit(1)

    # 2. Download Model if missing
    model_name = "mono+stereo_640x192"
    model_dir = MONODEPTH_DIR / "models" / model_name
    
    if not model_dir.exists():
        print(f"Model {model_name} not found. Downloading...")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        url = "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip"
        zip_path = model_dir / f"{model_name}.zip"
        
        try:
            print(f"Downloading from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            
            print("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            
            # Cleanup
            zip_path.unlink()
            print("Model setup complete.")
            
        except Exception as e:
            print(f"Failed to download/extract model: {e}")
            # Clean up partial download
            if model_dir.exists():
                shutil.rmtree(model_dir)
            sys.exit(1)

# Run setup before importing
setup_monodepth()

sys.path.insert(0, str(MONODEPTH_DIR))
import networks

# Configuration
DATA_ROOT = Path("data")
DATASET_DIR = DATA_ROOT / "visdrone_person"
AUGMENT_PERCENTAGE = 0.30
RANDOM_SEED = 42
SAMPLE_OUTPUT_DIR = Path("data/fog_samples")

# Monodepth2 model config
MODEL_NAME = "mono+stereo_640x192"
MODEL_PATH = SCRIPT_DIR / "monodepth2" / "models" / MODEL_NAME


class DepthEstimator:
    """Monodepth2 depth estimation wrapper."""
    
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading Monodepth2 model from {model_path}...")
        print(f"Using device: {self.device}")
        
        encoder_path = model_path / "encoder.pth"
        decoder_path = model_path / "depth.pth"
        
        # Load encoder
        self.encoder = networks.ResnetEncoder(18, False)
        loaded_dict = torch.load(encoder_path, map_location=self.device)
        self.feed_height = loaded_dict['height']
        self.feed_width = loaded_dict['width']
        filtered_dict = {k: v for k, v in loaded_dict.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict)
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Load decoder
        self.decoder = networks.DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        loaded_dict = torch.load(decoder_path, map_location=self.device)
        self.decoder.load_state_dict(loaded_dict)
        self.decoder.to(self.device)
        self.decoder.eval()
        
        print("Model loaded successfully!")
    
    def estimate_depth(self, image):
        """
        Estimate depth from RGB image.
        
        Args:
            image: PIL Image or numpy array (RGB)
        
        Returns:
            depth_map: numpy array (H, W) with values 0-255 (0=far, 255=near)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        original_width, original_height = image.size
        
        # Preprocess
        input_image = image.resize((self.feed_width, self.feed_height), Image.LANCZOS)
        input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            features = self.encoder(input_tensor)
            outputs = self.decoder(features)
        
        # Get disparity and resize to original
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False
        )
        
        # Convert to numpy depth map (invert disparity -> depth)
        disp_np = disp_resized.squeeze().cpu().numpy()
        
        # Normalize to 0-255
        disp_norm = (disp_np - disp_np.min()) / (disp_np.max() - disp_np.min() + 1e-8)
        depth_map = (255 - disp_norm * 255).astype(np.uint8)  # Invert: high disp = near = low depth for fog
        
        return depth_map


def add_fog(image_array, depth_map, beta=2.0, airlight=180):
    """
    Add fog using atmospheric scattering model with depth map.
    
    I(x) = J(x) * t(x) + A * (1 - t(x))
    
    Args:
        image_array: RGB image (H, W, 3) uint8
        depth_map: Depth map (H, W) uint8, 0=far, 255=near
        beta: Fog density (1.0-3.0 recommended)
        airlight: Atmospheric light intensity (150-255)
    
    Returns:
        Foggy image (H, W, 3) uint8
    """
    # Expand depth to 3 channels
    depth_3c = np.stack([depth_map] * 3, axis=-1)
    
    # Normalize depth and compute transmission
    norm_depth = depth_3c / 255.0
    transmission = np.exp(-norm_depth * beta)
    
    # Apply atmospheric scattering model
    foggy = image_array * transmission + airlight * (1 - transmission)
    foggy = np.clip(foggy, 0, 255).astype(np.uint8)
    
    return foggy


def process_single_image(args):
    """Process a single image with fog augmentation."""
    img_path, label_path, out_img_dir, out_lbl_dir, depth_estimator, beta, airlight = args
    
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img)
        
        # Estimate depth
        depth_map = depth_estimator.estimate_depth(img)
        
        # Add fog
        foggy_array = add_fog(img_array, depth_map, beta=beta, airlight=airlight)
        foggy_img = Image.fromarray(foggy_array)
        
        # Save foggy image
        fog_img_name = f"{img_path.stem}_fog{img_path.suffix}"
        fog_img_path = out_img_dir / fog_img_name
        foggy_img.save(fog_img_path, quality=95)
        
        # Copy label file
        if label_path and label_path.exists():
            fog_lbl_name = f"{label_path.stem}_fog{label_path.suffix}"
            fog_lbl_path = out_lbl_dir / fog_lbl_name
            shutil.copy2(label_path, fog_lbl_path)
        
        return (True, str(img_path), str(fog_img_path))
    
    except Exception as e:
        return (False, str(img_path), str(e))


def get_image_label_pairs(split):
    """Get all image-label pairs for a split."""
    img_dir = DATASET_DIR / "images" / split
    lbl_dir = DATASET_DIR / "labels" / split
    
    pairs = []
    for img_path in img_dir.glob("*.jpg"):
        if "_fog" in img_path.stem:
            continue
        label_path = lbl_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path if label_path.exists() else None))
    
    return pairs


def test_on_samples(num_samples=5):
    """Test fog augmentation on sample images."""
    print("=" * 60)
    print("FOG AUGMENTATION - TEST MODE (Monodepth2)")
    print("=" * 60)
    
    # Initialize depth estimator
    depth_estimator = DepthEstimator(MODEL_PATH)
    
    SAMPLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    pairs = get_image_label_pairs("train")
    if len(pairs) == 0:
        print("❌ No images found!")
        return []
    
    random.seed(RANDOM_SEED)
    samples = random.sample(pairs, min(num_samples, len(pairs)))
    
    # Different fog intensities
    fog_configs = [(1.5, 160), (2.0, 180), (2.5, 200)]
    
    print(f"\nProcessing {len(samples)} sample images...")
    print(f"Output: {SAMPLE_OUTPUT_DIR.absolute()}\n")
    
    results = []
    for i, (img_path, label_path) in enumerate(samples):
        beta, airlight = fog_configs[i % len(fog_configs)]
        
        args = (img_path, label_path, SAMPLE_OUTPUT_DIR, SAMPLE_OUTPUT_DIR, 
                depth_estimator, beta, airlight)
        success, orig_path, fog_path = process_single_image(args)
        
        if success:
            orig_copy = SAMPLE_OUTPUT_DIR / f"{img_path.stem}_original{img_path.suffix}"
            shutil.copy2(img_path, orig_copy)
            
            print(f"✅ [{i+1}/{len(samples)}] beta={beta}, airlight={airlight}")
            print(f"   {img_path.name} → {Path(fog_path).name}")
            results.append((orig_copy, fog_path))
        else:
            print(f"❌ [{i+1}/{len(samples)}] Failed: {fog_path}")
    
    print(f"\n📁 Samples saved to: {SAMPLE_OUTPUT_DIR.absolute()}")
    return results


def augment_dataset():
    """Run fog augmentation on full dataset (35%)."""
    print("=" * 60)
    print("FOG AUGMENTATION - FULL DATASET (Monodepth2)")
    print("=" * 60)
    print(f"Augmentation: {AUGMENT_PERCENTAGE * 100:.0f}%")
    
    depth_estimator = DepthEstimator(MODEL_PATH)
    
    random.seed(RANDOM_SEED)
    total = 0
    
    for split in ["train", "val"]:
        img_dir = DATASET_DIR / "images" / split
        lbl_dir = DATASET_DIR / "labels" / split
        
        if not img_dir.exists():
            continue
        
        pairs = get_image_label_pairs(split)
        num_aug = int(len(pairs) * AUGMENT_PERCENTAGE)
        selected = random.sample(pairs, num_aug)
        
        print(f"\n📂 {split.upper()}: {num_aug}/{len(pairs)} images")
        
        success = 0
        for img_path, label_path in tqdm(selected, desc=f"{split}"):
            beta = random.uniform(1.5, 2.5)
            airlight = random.randint(150, 200)
            
            args = (img_path, label_path, img_dir, lbl_dir, 
                    depth_estimator, beta, airlight)
            result = process_single_image(args)
            if result[0]:
                success += 1
        
        print(f"   ✅ {success} images augmented")
        total += success
    
    print(f"\n🎉 DONE! Total: {total} fog images")


def main():
    parser = argparse.ArgumentParser(description="Fog augmentation with Monodepth2")
    parser.add_argument("--test", action="store_true", help="Test on samples only")
    parser.add_argument("--samples", type=int, default=5, help="Number of test samples")
    args = parser.parse_args()
    
    if args.test:
        test_on_samples(args.samples)
    else:
        augment_dataset()


if __name__ == "__main__":
    main()
