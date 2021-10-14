# **GRSGD**

## **About**

GRSGD is a gradient sparsification method to accelerate the distributed edge learning process.

## Dependencies

Our simulation codes are implemented with Python 3.7.5 and PyTorch 1.3.1.

Before using the repository,  please install the necessary packages with:

```
pip install -r requirements.txt
```

## Simulation

You can change your configurations in python codes.

### Image Classification

For MNIST and Cifar10, run `cv.py` from the repository's root directory like:

```
python cv.py --gpus 0,1 --world-size 8 --epochs 80 --optim grsgd --dataset mnist --model CNNmnist
```

```
python cv.py --gpus 0,1 --world-size 8 --epochs 80 --optim grsgd --dataset cifar --model resnet50
```

For ImageNet, run `imgnet.py` from the repository's root directory like:

```
python imgnet.py --gpus 0,1 --world-size 4 --epochs 80 --optim grsgd
```

### Machine Translation

For Multi30k, run `multi30k.py` from the repository's root directory like:

```
python multi30k.py --gpus 0,1 --world-size 4 --epochs 20 --optim grsgd
```