# DP Glaucoma Segmentation

## Master Thesis Project

## Description   


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

## Model training

### Cascade architecture
```bash
python train.py -a binary -m ref -o ./output --epochs 10
```

```bash
python train.py -a cascade -m ref -o ./output --epochs 10 --base-model ./models/polar/ref/binary.pth
```

### Dual-decoder architecture
```bash
python train.py -a dual -m ref -o ./output --epochs 10
```

## Model evaluation

### Cascade architecture
```bash
python test.py -a binary -m ./models/polar/ref/binary.pth
```

```bash
python test.py -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth
```

### Dual-decoder architecture
```bash
python test.py -a dual -m ./models/polar/ref/dual.pth
```

## Inference

### Region of Interest (RoI) detection with CenterNet
```bash
python segment.py ./ImagesForSegmentation --centernet ./models/roi/centernet.pth --roi-output-dir ./RoiResults
```

### Cascade segmentation
```bash
python segment.py ./RoiResults -a cascade -m ./models/polar/ref/cascade.pth --base-model ./models/polar/ref/binary.pth -o ./CascadeResults
```

### Dual-decoder segmentation
```bash
python segment.py ./RoiResults -a dual -m ./models/polar/ref/dual.pth -o ./DualResults
```

## Results


