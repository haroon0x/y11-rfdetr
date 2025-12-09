import os
import sys
import torch
import importlib.util

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"[OK] {module_name} is installed.")
        return True
    except ImportError:
        print(f"[FAIL] {module_name} is NOT installed.")
        return False

def check_dir(path, required=True):
    if os.path.exists(path):
        print(f"[OK] Directory exists: {path}")
        return True
    else:
        status = "[FAIL]" if required else "[WARN]"
        print(f"{status} Directory missing: {path}")
        return not required

def main():
    print("=== Project Verification (Person Detection) ===")
    
    # Check Directories
    required_dirs = [
        "scripts",
        "data",
    ]
    optional_dirs = [
        "data/visdrone",
        "data/visdrone_person",
    ]
    
    all_required_ok = all([check_dir(d, required=True) for d in required_dirs])
    for d in optional_dirs:
        check_dir(d, required=False)
    
    # Check Imports
    modules = [
        "torch",
        "ultralytics",
        "cv2",  # opencv-python
        "PIL",  # Pillow (used in prepare_visdrone.py)
    ]
    all_imports_ok = all([check_import(m) for m in modules])
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA is NOT available. Training will be slow.")
    
    # Check for prepared dataset
    if os.path.exists("data/visdrone_person/data.yaml"):
        print("[OK] Person-only dataset is prepared (data/visdrone_person/data.yaml)")
    else:
        print("[INFO] Person-only dataset not yet prepared. Run: python scripts/prepare_visdrone.py")
        
    if all_required_ok and all_imports_ok:
        print("\n=== Setup Complete! ===")
    else:
        print("\n=== Setup Incomplete. Please check errors above. ===")

if __name__ == "__main__":
    main()
