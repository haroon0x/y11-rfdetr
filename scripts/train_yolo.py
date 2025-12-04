from ultralytics import YOLO
import os

def train_yolo():
    # Load a model
    model = YOLO("yolo11s.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data="data/visdrone_person/data.yaml",  # path to dataset YAML
        epochs=100,  # number of epochs to train for
        imgsz=1280,  # training image size
        batch=16,  # batch size
        patience=50,  # epochs to wait for no observable improvement for early stopping of training
        device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        project="yolo11s_person",  # project name
        name="exp",  # experiment name
        exist_ok=True,  # whether to overwrite existing experiment
        pretrained=True,  # whether to use a pretrained model
        optimizer="auto",  # optimizer to use, choices=['SGD', 'Adam', 'Adamax', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto']
        verbose=True,  # whether to print verbose output
        seed=0,  # random seed for reproducibility
        deterministic=True,  # whether to enable deterministic mode
        single_cls=True,  # train multi-class data as single-class
        rect=False,  # rectangular training
        cos_lr=False,  # cosine LR scheduler
        close_mosaic=10,  # (int) disable mosaic augmentation for final epochs (0 to disable)
        resume=False,  # resume training from last checkpoint
        amp=True,  # Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check
        fraction=1.0,  # dataset fraction to train on (default is 1.0, all images in train set)
        profile=False,  # profile ONNX and TensorRT speeds during training for loggers
        freeze=None,  # (int or list, optional) freeze first n layers, or freeze list of layer indices during training
        lr0=0.01,  # initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
        lrf=0.01,  # final learning rate (lr0 * lrf)
        momentum=0.937,  # SGD momentum/Adam beta1
        weight_decay=0.0005,  # optimizer weight decay 5e-4
        warmup_epochs=3.0,  # warmup epochs (fractions ok)
        warmup_momentum=0.8,  # warmup initial momentum
        warmup_bias_lr=0.1,  # warmup initial bias lr
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain (scale with pixels)
        dfl=1.5,  # dfl loss gain
        pose=12.0,  # pose loss gain
        kobj=1.0,  # keypoint obj loss gain
        label_smoothing=0.0,  # label smoothing (fraction)
        nbs=64,  # nominal batch size
        hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # image HSV-Value augmentation (fraction)
        degrees=0.0,  # image rotation (+/- deg)
        translate=0.1,  # image translation (+/- fraction)
        scale=0.5,  # image scale (+/- gain)
        shear=0.0,  # image shear (+/- deg)
        perspective=0.0,  # image perspective (+/- fraction), range 0-0.001
        flipud=0.0,  # image flip up-down (probability)
        fliplr=0.5,  # image flip left-right (probability)
        mosaic=1.0,  # image mosaic (probability)
        mixup=0.0,  # image mixup (probability)
        copy_paste=0.0,  # segment copy-paste (probability)
    )

if __name__ == "__main__":
    # Ensure data yaml exists or is created by merge script
    if not os.path.exists("data/visdrone_person/data.yaml"):
        print("Warning: data/visdrone_person/data.yaml not found. Please run prepare_visdrone.py first.")
    
    train_yolo()
