## We are going to use Jetson Nano for deployment and it supports only jetpack 4
# Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11

Source: https://docs.ultralytics.com/guides/nvidia-jetson/

---

# Quick Start Guide: NVIDIA Jetson with Ultralytics YOLO11

This comprehensive guide provides a detailed walkthrough for deploying Ultralytics YOLO11 on [NVIDIA Jetson](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/) devices. Additionally, it showcases performance benchmarks to demonstrate the capabilities of YOLO11 on these small and powerful devices.

!!! tip "New product support"

    We have updated this guide with the latest [NVIDIA Jetson AGX Thor Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor) which delivers up to 2070 FP4 TFLOPS of AI compute and 128 GB of memory with power configurable between 40 W and 130 W. It delivers over 7.5x higher AI compute than NVIDIA Jetson AGX Orin, with 3.5x better energy efficiency to seamlessly run the most popular AI models.




!!! note

    This guide has been tested with [NVIDIA Jetson AGX Thor Developer Kit(Jetson T5000)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-thor) running the latest stable JetPack release of [JP7.0](https://developer.nvidia.com/embedded/jetpack/downloads), [NVIDIA Jetson AGX Orin Developer Kit (64GB)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin) running JetPack release of [JP6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62), [NVIDIA Jetson Orin Nano Super Developer Kit](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit) running JetPack release of [JP6.1](https://developer.nvidia.com/embedded/jetpack-sdk-61), [Seeed Studio reComputer J4012](https://www.seeedstudio.com/reComputer-J4012-p-5586.html) which is based on NVIDIA Jetson Orin NX 16GB running JetPack release of [JP6.0](https://developer.nvidia.com/embedded/jetpack-sdk-60)/ JetPack release of [JP5.1.3](https://developer.nvidia.com/embedded/jetpack-sdk-513) and [Seeed Studio reComputer J1020 v2](https://www.seeedstudio.com/reComputer-J1020-v2-p-5498.html) which is based on NVIDIA Jetson Nano 4GB running JetPack release of [JP4.6.1](https://developer.nvidia.com/embedded/jetpack-sdk-461). It is expected to work across all the NVIDIA Jetson hardware lineup including latest and legacy.

## What is NVIDIA Jetson?

NVIDIA Jetson is a series of embedded computing boards designed to bring accelerated AI (artificial intelligence) computing to edge devices. These compact and powerful devices are built around NVIDIA's GPU architecture and are capable of running complex AI algorithms and [deep learning](https://www.ultralytics.com/glossary/deep-learning-dl) models directly on the device, without needing to rely on [cloud computing](https://www.ultralytics.com/glossary/cloud-computing) resources. Jetson boards are often used in robotics, autonomous vehicles, industrial automation, and other applications where AI inference needs to be performed locally with low latency and high efficiency. Additionally, these boards are based on the ARM64 architecture and runs on lower power compared to traditional GPU computing devices.



## What is NVIDIA JetPack?

[NVIDIA JetPack SDK](https://developer.nvidia.com/embedded/jetpack) powering the Jetson modules is the most comprehensive solution and provides full development environment for building end-to-end accelerated AI applications and shortens time to market. JetPack includes Jetson Linux with bootloader, Linux kernel, Ubuntu desktop environment, and a complete set of libraries for acceleration of GPU computing, multimedia, graphics, and [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv). It also includes samples, documentation, and developer tools for both host computer and developer kit, and supports higher level SDKs such as [DeepStream](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson/) for streaming video analytics, Isaac for robotics, and Riva for conversational AI.

## Flash JetPack to NVIDIA Jetson

The first step after getting your hands on an NVIDIA Jetson device is to flash NVIDIA JetPack to the device. There are several different way of flashing NVIDIA Jetson devices.

1. If you own an official NVIDIA Development Kit such as the Jetson AGX Thor Developer Kit, you can [download an image and prepare a bootable USB stick to flash JetPack to the included SSD](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/quick_start.html).
2. If you own an official NVIDIA Development Kit such as the Jetson Orin Nano Developer Kit, you can [download an image and prepare an SD card with JetPack for booting the device](https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit).
3. If you own any other NVIDIA Development Kit, you can [flash JetPack to the device using SDK Manager](https://docs.nvidia.com/sdk-manager/install-with-sdkm-jetson/index.html).
4. If you own a Seeed Studio reComputer J4012 device, you can [flash JetPack to the included SSD](https://wiki.seeedstudio.com/reComputer_J4012_Flash_Jetpack/) and if you own a Seeed Studio reComputer J1020 v2 device, you can [flash JetPack to the eMMC/ SSD](https://wiki.seeedstudio.com/reComputer_J2021_J202_Flash_Jetpack/).
5. If you own any other third party device powered by the NVIDIA Jetson module, it is recommended to follow [command-line flashing](https://docs.nvidia.com/jetson/archives/r35.5.0/DeveloperGuide/IN/QuickStart.html).

!!! note

    For methods 1, 4 and 5 above, after flashing the system and booting the device, please enter "sudo apt update && sudo apt install nvidia-jetpack -y" on the device terminal to install all the remaining JetPack components needed.

## JetPack Support Based on Jetson Device

The below table highlights NVIDIA JetPack versions supported by different NVIDIA Jetson devices.

|                   | JetPack 4 | JetPack 5 | JetPack 6 | JetPack 7 |
| ----------------- | --------- | --------- | --------- | --------- |
| Jetson Nano       | ✅        | ❌        | ❌        | ❌        |
| Jetson TX2        | ✅        | ❌        | ❌        | ❌        |
| Jetson Xavier NX  | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Xavier | ✅        | ✅        | ❌        | ❌        |
| Jetson AGX Orin   | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin NX    | ❌        | ✅        | ✅        | ❌        |
| Jetson Orin Nano  | ❌        | ✅        | ✅        | ❌        |
| Jetson AGX Thor   | ❌        | ❌        | ❌        | ✅        |

## Quick Start with Docker

The fastest way to get started with Ultralytics YOLO11 on NVIDIA Jetson is to run with pre-built docker images for Jetson. Refer to the table above and choose the JetPack version according to the Jetson device you own.

=== "JetPack 4"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack4
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 5"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack5
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```

=== "JetPack 6"

    ```bash
    t=ultralytics/ultralytics:latest-jetson-jetpack6
    sudo docker pull $t && sudo docker run -it --ipc=host --runtime=nvidia $t
    ```


After this is done, skip to [Use TensorRT on NVIDIA Jetson section](#use-tensorrt-on-nvidia-jetson).

## Start with Native Installation

For a native installation without Docker, please refer to the steps below.

### Run on JetPack 7.0

#### Install Ultralytics Package

Here we will install Ultralytics package on the Jetson with optional dependencies so that we can export the [PyTorch](https://www.ultralytics.com/glossary/pytorch) models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the Jetson devices.

1. Update packages list, install pip and upgrade to latest

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. Install `ultralytics` pip package with optional dependencies

    ```bash
    pip install ultralytics[export]
    ```

3. Reboot the device

    ```bash
    sudo reboot
    ```

#### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these 2 packages installed via pip are not compatible to run on Jetson AGX Thor which comes with JetPack 7.0 and CUDA 13. Therefore, we need to manually install them.

Install `torch` and `torchvision` according to JP7.0

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

Here we will download and install `onnxruntime-gpu 1.24.0` with `Python3.12` support.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.24.0-cp312-cp312-linux_aarch64.whl
```

### Run on JetPack 6.1

#### Install Ultralytics Package

Here we will install Ultralytics package on the Jetson with optional dependencies so that we can export the [PyTorch](https://www.ultralytics.com/glossary/pytorch) models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the Jetson devices.

1. Update packages list, install pip and upgrade to latest

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. Install `ultralytics` pip package with optional dependencies

    ```bash
    pip install ultralytics[export]
    ```

3. Reboot the device

    ```bash
    sudo reboot
    ```

#### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these two packages installed via pip are not compatible with the Jetson platform, which is based on ARM64 architecture. Therefore, we need to manually install a pre-built PyTorch pip wheel and compile or install Torchvision from source.

Install `torch 2.5.0` and `torchvision 0.20` according to JP6.1

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

!!! note

    Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

Install [`cuSPARSELt`](https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) to fix a dependency issue with `torch 2.5.0`

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

You can find all available `onnxruntime-gpu` packages—organized by JetPack version, Python version, and other compatibility details—in the [Jetson Zoo ONNX Runtime compatibility matrix](https://elinux.org/Jetson_Zoo#ONNX_Runtime).

For **JetPack 6** with `Python 3.10` support, you can install `onnxruntime-gpu 1.23.0`:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

Alternatively, for `onnxruntime-gpu 1.20.0`:

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

### Run on JetPack 5.1.2

#### Install Ultralytics Package

Here we will install Ultralytics package on the Jetson with optional dependencies so that we can export the PyTorch models to other different formats. We will mainly focus on [NVIDIA TensorRT exports](../integrations/tensorrt.md) because TensorRT will make sure we can get the maximum performance out of the Jetson devices.

1. Update packages list, install pip and upgrade to latest

    ```bash
    sudo apt update
    sudo apt install python3-pip -y
    pip install -U pip
    ```

2. Install `ultralytics` pip package with optional dependencies

    ```bash
    pip install ultralytics[export]
    ```

3. Reboot the device

    ```bash
    sudo reboot
    ```

#### Install PyTorch and Torchvision

The above ultralytics installation will install Torch and Torchvision. However, these two packages installed via pip are not compatible with the Jetson platform, which is based on ARM64 architecture. Therefore, we need to manually install a pre-built PyTorch pip wheel and compile or install Torchvision from source.

1. Uninstall currently installed PyTorch and Torchvision

    ```bash
    pip uninstall torch torchvision
    ```

2. Install `torch 2.2.0` and `torchvision 0.17.2` according to JP5.1.2

    ```bash
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.2.0-cp38-cp38-linux_aarch64.whl
    pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.17.2+c1d70fe-cp38-cp38-linux_aarch64.whl
    ```

!!! note

    Visit the [PyTorch for Jetson page](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048) to access all different versions of PyTorch for different JetPack versions. For a more detailed list on the PyTorch, Torchvision compatibility, visit the [PyTorch and Torchvision compatibility page](https://github.com/pytorch/vision).

#### Install `onnxruntime-gpu`

The [onnxruntime-gpu](https://pypi.org/project/onnxruntime-gpu/) package hosted in PyPI does not have `aarch64` binaries for the Jetson. So we need to manually install this package. This package is needed for some of the exports.

You can find all available `onnxruntime-gpu` packages—organized by JetPack version, Python version, and other compatibility details—in the [Jetson Zoo ONNX Runtime compatibility matrix](https://elinux.org/Jetson_Zoo#ONNX_Runtime). Here we will download and install `onnxruntime-gpu 1.17.0` with `Python3.8` support.

```bash
wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl
```

!!! note

    `onnxruntime-gpu` will automatically revert back the numpy version to latest. So we need to reinstall numpy to `1.23.5` to fix an issue by executing:

    `pip install numpy==1.23.5`

## Use TensorRT on NVIDIA Jetson

Among all the model export formats supported by Ultralytics, TensorRT offers the highest inference performance on NVIDIA Jetson devices, making it our top recommendation for Jetson deployments. For setup instructions and advanced usage, see our [dedicated TensorRT integration guide](../integrations/tensorrt.md).

### Convert Model to TensorRT and Run Inference

The YOLO11n model in PyTorch format is converted to TensorRT to run inference with the exported model.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Export the model to TensorRT
        model.export(format="engine")  # creates 'yolo11n.engine'

        # Load the exported TensorRT model
        trt_model = YOLO("yolo11n.engine")

        # Run inference
        results = trt_model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export a YOLO11n PyTorch model to TensorRT format
        yolo export model=yolo11n.pt format=engine # creates 'yolo11n.engine'

        # Run inference with the exported model
        yolo predict model=yolo11n.engine source='https://ultralytics.com/images/bus.jpg'
        ```

!!! note

    Visit the [Export page](../modes/export.md#arguments) to access additional arguments when exporting models to different model formats



## Reproduce Our Results

To reproduce the above Ultralytics benchmarks on all export [formats](../modes/export.md) run this code:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11n PyTorch model
        model = YOLO("yolo11n.pt")

        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all all export formats
        results = model.benchmark(data="coco128.yaml", imgsz=640)
        ```

    === "CLI"

        ```bash
        # Benchmark YOLO11n speed and accuracy on the COCO128 dataset for all all export formats
        yolo benchmark model=yolo11n.pt data=coco128.yaml imgsz=640
        ```

    Note that benchmarking results might vary based on the exact hardware and software configuration of a system, as well as the current workload of the system at the time the benchmarks are run. For the most reliable results use a dataset with a large number of images, i.e. `data='coco.yaml'` (5000 val images).

## Best Practices when using NVIDIA Jetson

When using NVIDIA Jetson, there are a couple of best practices to follow in order to enable maximum performance on the NVIDIA Jetson running YOLO11.

1. Enable MAX Power Mode

    Enabling MAX Power Mode on the Jetson will make sure all CPU, GPU cores are turned on.

    ```bash
    sudo nvpmodel -m 0
    ```

2. Enable Jetson Clocks

    Enabling Jetson Clocks will make sure all CPU, GPU cores are clocked at their maximum frequency.

    ```bash
    sudo jetson_clocks
    ```

3. Install Jetson Stats Application

    We can use jetson stats application to monitor the temperatures of the system components and check other system details such as view CPU, GPU, RAM utilization, change power modes, set to max clocks, check JetPack information

    ```bash
    sudo apt update
    sudo pip install jetson-stats
    sudo reboot
    jtop
    ```

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/jetson-stats-application.avif" alt="Jetson Stats">

