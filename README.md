# PSGAN-PyTorch

**PyTorch** implementation of [**PSGAN: Pose and Expression Robust Spatial-Aware GAN for Customizable Makeup Transfer**](https://arxiv.org/abs/1909.06956), still in construction...

## Related

A makeup-transfer App [**MagicMirror**](https://github.com/Super262/MagicMirror) is developed by [Fengwei Zhang](https://github.com/Super262).

Here are some exemplar results.

When source image and target image are both from the makeup dataset.

![app_example_dataset](https://github.com/DateBro/PSGAN-PyTorch/blob/master/imgs/app_example_dataset.jpg)

When the source image is from the makeup dataset and the target image is from weibo.

![app_example_weibo](https://github.com/DateBro/PSGAN-PyTorch/blob/master/imgs/app_example_weibo.jpg)

When we try to transfer the makeup style from the makeup dataset to an makeup image from weibo.

![app_example1](https://github.com/DateBro/PSGAN-PyTorch/blob/master/imgs/app_example1.jpg)

## Preparation

- **Prerequisites**
    - PyTorch
    - Python 3.x with matplotlib, numpy and scipy

- **Dataset**
Dataset can be found in project page: http://colalab.org/projects/BeautyGAN

## Usage

Before training, you should generate train/test split labels using data_preparation/generate_labels.py. What you should do is just modify the data path in generate_labels.py.

The training setting is the same as [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch). The implementation of PSGAN is still incomplete. I still have some problems in implementing the AMM with 68 landmarks detector.

However, the incomplete results is satisfying.

The training example of 50th epoch is as below:

![PSGAN_training_result](https://github.com/DateBro/PSGAN-PyTorch/blob/master/imgs/PSGAN_49_1259_fake.jpg)

The training example of BeautyGAN in 200th epoch is as below:

![BeautyGAN_training_result](https://github.com/DateBro/PSGAN-PyTorch/blob/master/imgs/BeautyGAN_199_1259_fake.jpg)

Though the implementation of PSGAN is still incomplete, it's obvious that PSGAN is pose and expression robust for makeup transfer.

## Acknowledgement

The code is built upon [BeautyGAN](https://github.com/wtjiang98/BeautyGAN_pytorch), thanks for their excellent work!