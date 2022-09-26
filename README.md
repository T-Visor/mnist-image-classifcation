# mnist-image-classification
A basic CNN (Convolutional Neural Network) which can classify handwritten digits from
the MNIST (Modified National Institute of Standards and Technology) dataset.

This work extends on code provided at
https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-from-scratch-for-mnist-handwritten-digit-classification/

## Installation
Run the following command:

```bash
pip3 install -r requirements.txt
```

For conda users:

```bash
conda env create -f environment.yml
```

## Get Started

Run **create-digit-classifier.py** to create a digit classification model and save it to the disk.

```bash
usage: create-digit-classifier.py [-h] -k {2,3,4,5}

options:
  -h, --help            show this help message and exit
  -k {2,3,4,5}, --k_value {2,3,4,5}
                        Determines k-value for k-fold cross validation.
```

Afterward, run **make-prediction.py** to classify a sample image for validation.

![classification-result](https://user-images.githubusercontent.com/25760670/129660012-d76c2d06-b90d-4c4e-8695-9ebe0d4ee3e1.png)
