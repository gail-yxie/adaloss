# AdaLoss: A computationally-efficient and provably convergent adaptive gradient method

This repository is the official implementation of [AdaLoss: A computationally-efficient and provably convergent adaptive gradient method](https://arxiv.org/pdf/2109.08282.pdf) (AAAI 2022).
- 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training (Fine-tuning)

To fine-tuning the pre-trianed deep neural network models---ViT S/16 and ResNet ---on CIFAR100 as the experiments in the paper, run this command:

```train
python train.py -c ./configs/config_cifar100_adaloss.json --b0
```

>📋  
> When running the following training scripts, the code will download CIFAR100 datasets and pretrianed models automatically.

### Source of Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z.

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Comparison of AdaLoss (ours), AdaGrad-Norm, SGD_const, SGD_sqrt on test accuracy on CIFAR100 by fine-tuning on pretrained DNNs, vision transformer ViT-S/16 and ResNet50-swsl. 
Training: 45k, validation: 5k, and test: 10k.

### [Image Classification on CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)

| b_0    | AdaLoss        | AdaGrad-Norm   | SGD_Constant   | SGD_DecaySqrt  |
|--------|----------------|----------------|----------------|----------------|
| 0.01   | $90.77\pm0.12$ | $62.22\pm1.47$ | N/A            | $90.50\pm0.21$ |
| 0.1    | $90.78\pm0.03$ | $86.61\pm0.38$ | N/A            | $90.57\pm0.08$ |
| 1      | $90.65\pm0.26$ | $88.98\pm0.45$ | $82.73\pm1.17$ | $90.61\pm0.11$ |
| 10     | $90.56\pm0.25$ | $90.67\pm0.15$ | $90.46\pm0.26$ | $90.60\pm0.08$ |
| 100    | $89.51\pm0.0$  | $89.80\pm0.12$ | $89.75\pm0.09$ | $89.43\pm0.13$ |
| % 1000 | $76.78\pm1.14$ | $79.87\pm0.94$ | $77.33\pm0.94$ | $77.25\pm1.06$ |


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 

##Bibtex
@article{wu2021adaloss,
title={AdaLoss: A computationally-efficient and provably convergent adaptive gradient method},
author={Wu, Xiaoxia and Xie, Yuege and Du, Simon and Ward, Rachel},
journal={arXiv preprint arXiv:2109.08282},
year={2021}
}