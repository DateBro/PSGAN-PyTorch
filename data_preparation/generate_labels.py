import os
import sys
import random

"""Among 3834 images, we randomly select 100 non-makeup images and 250 makeup images for test.
The remaining images are separated into training set and validation set."""

train_non_makeup_labels = 'train_SYMIX.txt'
train_makeup_labels = 'train_MAKEMIX.txt'
test_non_makeup_labels = 'test_SYMIX.txt'
test_makeup_labels = 'test_MAKEMIX.txt'

# data_path = '/home/zhiyong/RemoteServer/data/makeup_dataset/'
# Windows测试文件夹
# data_path = 'E:/Datasets/makeup_dataset/'
data_path = 'F:/zzy/data/makeup_dataset'
makeup_path = 'all/images/makeup/'
non_makeup_path = 'all/images/non-makeup/'

makeup_files = os.listdir(os.path.join(data_path, makeup_path))
non_makeup_files = os.listdir(os.path.join(data_path, non_makeup_path))

# 每次生成的文件都是随机的
random.shuffle(makeup_files)
random.shuffle(non_makeup_files)

test_non_makeup_files = non_makeup_files[:100]
train_non_makeup_files = non_makeup_files[100:]
test_makeup_files = makeup_files[:250]
train_makeup_files = makeup_files[250:]

with open(os.path.join(data_path, train_non_makeup_labels), 'wt') as f:
    for file_name in train_non_makeup_files:
        file_path = os.path.join(non_makeup_path, file_name)
        mask_file_path = file_path.replace('images', 'segs')
        f.write(file_path + ' ' + mask_file_path)
        f.write('\n')

with open(os.path.join(data_path, train_makeup_labels), 'wt') as f:
    for file_name in train_makeup_files:
        file_path = os.path.join(makeup_path, file_name)
        mask_file_path = file_path.replace('images', 'segs')
        f.write(file_path + ' ' + mask_file_path)
        f.write('\n')

with open(os.path.join(data_path, test_non_makeup_labels), 'wt') as f:
    for file_name in test_non_makeup_files:
        file_path = os.path.join(non_makeup_path, file_name)
        mask_file_path = file_path.replace('images', 'segs')
        f.write(file_path + ' ' + mask_file_path)
        f.write('\n')

with open(os.path.join(data_path, test_makeup_labels), 'wt') as f:
    for file_name in test_makeup_files:
        file_path = os.path.join(makeup_path, file_name)
        mask_file_path = file_path.replace('images', 'segs')
        f.write(file_path + ' ' + mask_file_path)
        f.write('\n')