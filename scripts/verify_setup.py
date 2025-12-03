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

def check_dir(path):
    if os.path.exists(path):
        print(f"[OK] Directory exists: {path}")
        return True
    else:
        print(f"[FAIL] Directory missing: {path}")
        return False

def main():
    print("=== Project Verification ===")
    
    # Check Directories
    dirs = [
        "data/nomad",
        "data/visdrone",
        "data/vtsar",
        "models",
        "scripts",
        "notebooks"
    ]
    all_dirs_ok = all([check_dir(d) for d in dirs])
    
    # Check Imports
    modules = [
        "torch",
        "ultralytics",
        "roboflow",
        "cv2", # opencv-python
    ]
    all_imports_ok = all([check_import(m) for m in modules])
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARN] CUDA is NOT available. Training will be slow.")
        
    if all_dirs_ok and all_imports_ok:
        print("\n=== Setup Complete! ===")
    else:
        print("\n=== Setup Incomplete. Please check errors above. ===")

if __name__ == "__main__":
    main()
