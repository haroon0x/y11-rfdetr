Train an RF-DETR Model¶
You can train an RF-DETR model on a custom dataset using the rfdetr Python package, or in the cloud using Roboflow.

Training on device is ideal if you want to manage your training pipeline and have a GPU available for training.

Training in the Roboflow Cloud is ideal if you want managed training whose weights you can deploy on your own hardware and with a hosted API.

For this guide, we will train a model using the rfdetr Python package.

Once you have trained a model with this guide, see our deploy an RF-DETR model guide to learn how to run inference with your model.

Dataset structure¶
RF-DETR expects the dataset to be in COCO format. Divide your dataset into three subdirectories: train, valid, and test. Each subdirectory should contain its own _annotations.coco.json file that holds the annotations for that particular split, along with the corresponding image files. Below is an example of the directory structure:


dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── _annotations.coco.json
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
Roboflow allows you to create object detection datasets from scratch or convert existing datasets from formats like YOLO, and then export them in COCO JSON format for training. You can also explore Roboflow Universe to find pre-labeled datasets for a range of use cases.

Fine-tuning¶
You can fine-tune RF-DETR from pre-trained COCO checkpoints. By default, the RF-DETR-B checkpoint will be used. To get started quickly, please refer to our fine-tuning Google Colab notebook.


from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>
)
Different GPUs have different VRAM capacities, so adjust batch_size and grad_accum_steps to maintain a total batch size of 16. For example, on a powerful GPU like the A100, use batch_size=16 and grad_accum_steps=1; on smaller GPUs like the T4, use batch_size=4 and grad_accum_steps=4. This gradient accumulation strategy helps train effectively even with limited memory.

More parameters
Result checkpoints¶
During training, multiple model checkpoints are saved to the output directory:

checkpoint.pth – the most recent checkpoint, saved at the end of the latest epoch.

checkpoint_<number>.pth – periodic checkpoints saved every N epochs (default is every 10).

checkpoint_best_ema.pth – best checkpoint based on validation score, using the EMA (Exponential Moving Average) weights. EMA weights are a smoothed version of the model’s parameters across training steps, often yielding better generalization.

checkpoint_best_regular.pth – best checkpoint based on validation score, using the raw (non-EMA) model weights.

checkpoint_best_total.pth – final checkpoint selected for inference and benchmarking. It contains only the model weights (no optimizer state or scheduler) and is chosen as the better of the EMA and non-EMA models based on validation performance.

Checkpoint file sizes
Resume training¶
You can resume training from a previously saved checkpoint by passing the path to the checkpoint.pth file using the resume argument. This is useful when training is interrupted or you want to continue fine-tuning an already partially trained model. The training loop will automatically load the weights and optimizer state from the provided checkpoint file.


from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>,
    resume=<CHECKPOINT_PATH>
)
Early stopping¶
Early stopping monitors validation mAP and halts training if improvements remain below a threshold for a set number of epochs. This can reduce wasted computation once the model converges. Additional parameters—such as early_stopping_patience, early_stopping_min_delta, and early_stopping_use_ema—let you fine-tune the stopping behavior.


from rfdetr import RFDETRBase

model = RFDETRBase()

model.train(
    dataset_dir=<DATASET_PATH>,
    epochs=10,
    batch_size=4
    grad_accum_steps=4,
    lr=1e-4,
    output_dir=<OUTPUT_PATH>,
    early_stopping=True
)