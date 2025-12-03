Use Cascade R-CNN or YOLO11 anchor-free architecture as your baseline model instead of standard YOLO variants.

Mix negative samples (background-only images) into your training data at a 1:1 to 1:3 ratio with positive examples to reduce false positives.

Set input resolution to 1024×1024 or 1280×1280 pixels to preserve detail in tiny objects rather than standard 640×640.

Replace standard IoU loss with Normalized Wasserstein Distance (NWD) loss for better handling of positioning errors in tiny objects.

Apply mosaic and mixup augmentation to combine multiple training images and force the model to learn multiple scales and contexts simultaneously.

Implement dynamic area-weighted bounding box regression where smaller objects receive higher optimization weights during training.

Train on a diverse dataset combining TinyPerson, HIT-UAV, and custom negative samples to ensure robustness across different altitudes, backgrounds, and lighting conditions.

Integrate Bidirectional Feature Pyramid Network (BiFPN) instead of standard FPN for improved multi-scale feature fusion across pyramid levels.

Combine Cascade R-CNN and YOLO11 predictions through model ensembling (e.g., weighted averaging of bounding boxes) to leverage strengths of both architectures.

Apply Test-Time Augmentation (TTA) at inference by generating multiple augmented versions (flips, rotations, brightness shifts) and averaging predictions across them.

Add a P2 detection head (320×320 feature map) in addition to standard P3, P4, P5 heads to capture extremely small humans.

Add random crop, rotation, cutmix, and cutout augmentations to improve robustness under occlusion and scale variations.

Set weight decay to 0.0005–0.001 to prevent overfitting on small object patterns that may not generalize to new scenes.

Train with a learning rate of 0.01 decaying to 0.0001, batch size of 32–64, and warmup for 3–5 epochs to stabilize training on imbalanced data.

Include Squeeze-and-Excitation (SE) attention blocks to adaptively enhance the representation of small-object features in the backbone.