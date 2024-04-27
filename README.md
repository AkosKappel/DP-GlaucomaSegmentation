# DP Glaucoma Segmentation

Author: **Bc. Ákos Kappel**

Thesis Title: **Neural network based semi-automatic segmentation methods to enhance the detection and monitoring of human eye diseases**

Study Programme: **Intelligent Software Systems**

Year: **2022 - 2024**

Supervisor: **doc. RNDr. Silvester Czanner, PhD.**

Institution: **Faculty of Informatics and Information Technologies, Slovak University of Technology in Bratislava**


## Introduction

This repository contains the code for my Master's thesis project at the *Faculty of Informatics and Information Technologies, Slovak University of Technology in Bratislava* ([FIIT STU](https://www.fiit.stuba.sk/)).
The project is focused on the development of a deep learning model for the segmentation of the optic disc and cup in fundus images.
The model is trained on the [ORIGA](https://pubmed.ncbi.nlm.nih.gov/21095735/) dataset and evaluated on the [DRISHTI-GS](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) dataset.
The goal of the project is to propose, implement and evaluate 2 novel segmentation architectures for the task of glaucoma segmentation:

1. **[Cascade architecture](#cascade-architecture)** - a two-stage segmentation model, where the first stage segments the optic disc and the second stage segments the optic cup, conditioned on the optic disc segmentation.
2. **[Dual-decoder architecture](#dual-decoder-architecture)** - a single-stage segmentation model with two decoder branches, where the first decoder segments the optic disc and the second decoder segments the optic cup, while sharing the encoder features.

The project is implemented in [Python](https://www.python.org/) 3.10.2 using the [PyTorch](https://pytorch.org/) 2.0.1 deep learning framework, and other libraries such as [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/), [Matplotlib](https://matplotlib.org/), [OpenCV](https://opencv.org/) and [Albumentations](https://albumentations.ai/).

![Optic Disc and Cup](./assets/OD-OC-NRR-ISNT.jpg)


## Installation

```bash
# clone github repository   
git clone https://github.com/AkosKappel/DP-GlaucomaSegmentation
cd DP-GlaucomaSegmentation

# create new virtual environment
python3 -m venv venv

# activate virtual environment
venv/Scripts/activate # Windows
# or
source venv/bin/activate # Linux

# install dependencies
pip install -r requirements.txt
```


## Usage

When the installation is finished, you can start using the project.
The project consists of 3 main parts:

1. **[Region of Interest detection with CenterNet](#region-of-interest-detection-with-centernet)** - detect the region of interest in the fundus images.
2. **[Cascade architecture](#cascade-architecture)** - train, evaluate and perform inference with the cascade architecture.
3. **[Dual-decoder architecture](#dual-decoder-architecture)** - train, evaluate and perform inference with the dual-decoder architecture.

To run the project, you need to download the [ORIGA](https://www.dropbox.com/s/7z3z3z1z9v6zv6o/ORIGA.zip?dl=0) and [DRISHTI-GS](https://cvit.iiit.ac.in/projects/mip/drishti-gs/mip-dataset2/Home.php) datasets.
After downloading the datasets, extract the images to the `./data/ORIGA` and `./data/DRISHTI` directories, respectively.
Inside these directories, create the following subdirectories:

- `./data/ORIGA/TrainImages` - contains the training images from the ORIGA dataset.
- `./data/ORIGA/TrainMasks` - contains the training masks from the ORIGA dataset.
- `./data/ORIGA/TestImages` - contains the validation images from the ORIGA dataset.
- `./data/ORIGA/TestMasks` - contains the validation masks from the ORIGA dataset.

You can use 3 scripts to run the project:

- **`segment.py`** - detect the region of interest in the fundus images and perform the segmentation.
- **`train.py`** - train the models.
- **`test.py`** - evaluate the models.

Each script provides the `-h` flag to display the help message with the available options.
When using these scripts, you can select between these architectures:

- **`-a binary`** - binary segmentation of the optic disc.
- **`-a cascade`** - cascade architecture for the optic disc and cup segmentation.
- **`-a dual`** - dual-decoder architecture for the optic disc and cup segmentation.

As per the model selection, all our models are modified variants of the [U-Net](https://arxiv.org/abs/1505.04597) architecture.
You can choose from these types of models:

1. [Residual Attention U-Net++](https://www.mdpi.com/2076-3417/12/14/7149) (RAU-net++)
2. [Refined U-net 3+ with CBAM](https://www.mdpi.com/2075-4418/13/3/576) (RefU-net3+)
3. [Shifted-Window Vision Transformer U-net](https://link.springer.com/chapter/10.1007/978-3-031-25066-8_9) (Swin-Unet)


### Region of Interest detection with CenterNet

![Region of Interest detection](./assets/CenterNetPipeline.png)

The first step in the proposed pipeline is to detect the region of interest (ROI) in the fundus images.
For this purpose, the [CenterNet](https://openaccess.thecvf.com/content_ICCV_2019/html/Duan_CenterNet_Keypoint_Triplets_for_Object_Detection_ICCV_2019_paper.html) model is used to detect the bounding box of the optic disc.
To run the ROI detection, use the following command:

```bash
python segment.py ./ImagesForSegmentation --centernet ./models/roi/centernet.pth --roi-output-dir ./results/roi
```


### Cascade architecture

![Cascade architecture](./assets/CascadeArchitecture.png)

#### Model training

The cascade architecture consists of two models: the base model for the optic disc segmentation and the cascade model for the optic cup segmentation.
The base model is trained first:

```bash
python train.py -a binary -m ref -o ./output --epochs 10
```

And then the cascade model is trained:

```bash
python train.py -a cascade -m ref -o ./output --epochs 10 --base-model ./models/polar/ref/binary.pth
```

#### Model evaluation

To evaluate the final model, run the following command:

```bash
python test.py -a binary -m ./models/polar/ref/binary.pth
```

And to evaluate both models combined, use the following command:

```bash
python test.py -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth
```

#### Inference

To perform inference on a new set of images, use the following command:

```bash
python segment.py ./results/roi -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth -o ./results/cascade
```


### Dual-decoder architecture

![Dual-decoder architecture](./assets/DualArchitecture.png)

#### Model training

The dual-decoder architecture consists of a single model with two decoder branches for the optic disc and cup segmentation.
To train the model, run the following command:

```bash
python train.py -a dual -m ref -o ./output --epochs 10
```

#### Model evaluation

When the training is finished, evaluate the model using the following command:

```bash
python test.py -a dual -m ./models/polar/ref/dual.pth
```

#### Inference

If you want to perform inference on a new set of images, use the following command:

```bash
python segment.py ./results/roi -a dual -m ./models/polar/ref/dual.pth -o ./results/dual
```


## Results

The models were trained on images in their Polar Coordinate representation.
Because of this, the raw results from the models look like this:

![Cascade Results](./assets/cascade-plar-results.png)

After transforming the results back to Cartesian coordinates, the final segmentation masks look like:

![Dual Results](./assets/dual-cartesian-results.png)

