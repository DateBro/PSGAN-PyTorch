import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToPILImage

import faceutils as futils
from ops.histogram_matching import *

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])


def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img


def copy_area(tar, src, lms):
    rect = [int(min(lms[:, 1])) - preprocess_image.eye_margin,
            int(min(lms[:, 0])) - preprocess_image.eye_margin,
            int(max(lms[:, 1])) + preprocess_image.eye_margin + 1,
            int(max(lms[:, 0])) + preprocess_image.eye_margin + 1]
    tar[:, :, rect[1]:rect[3], rect[0]:rect[2]] = \
        src[:, :, rect[1]:rect[3], rect[0]:rect[2]]


def to_var(x, requires_grad=True):
    if requires_grad:
        return Variable(x).float()
    else:
        return Variable(x, requires_grad=requires_grad).float()


def preprocess_image(image: Image):
    face = futils.dlib.detect(image)

    assert face, "no faces detected"

    # face[0]是第一个人脸，给定图片中只能有一个人脸
    face = face[0]
    image, face = futils.dlib.crop(image, face)

    # detect landmark
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round()
    lms_eye_left = lms[42:48]
    lms_eye_right = lms[36:42]
    lms = lms.transpose((1, 0)).reshape(-1, 1, 1)  # transpose to (y-x)
    lms = np.tile(lms, (1, 256, 256))  # (136, h, w)

    # calculate relative position for each pixel
    fix = np.zeros((256, 256, 68 * 2))
    for i in range(256):  # row (y) h
        for j in range(256):  # column (x) w
            fix[i, j, :68] = i
            fix[i, j, 68:] = j
    fix = fix.transpose((2, 0, 1))  # (136, h, w)
    diff = to_var(torch.Tensor(fix - lms).unsqueeze(0), requires_grad=False)

    # obtain face parsing result
    image = image.resize((512, 512), Image.ANTIALIAS)
    mask = futils.mask.mask(image).resize((256, 256), Image.ANTIALIAS)
    mask = to_var(ToTensor(mask).unsqueeze(0), requires_grad=False)
    mask_lip = (mask == 7).float() + (mask == 9).float()
    mask_face = (mask == 1).float() + (mask == 6).float()

    # 需要抠出 mask_eye
    mask_eyes = torch.zeros_like(mask)
    copy_area(mask_eyes, mask_face, lms_eye_left)
    copy_area(mask_eyes, mask_face, lms_eye_right)
    mask_eyes = to_var(mask_eyes, requires_grad=False)

    mask_list = [mask_lip, mask_face, mask_eyes]
    mask_aug = torch.cat(mask_list, 0)  # (3, 1, h, w)
    # 根据给定 size 或 scale_factor，上采样或下采样输入数据input
    mask_re = F.interpolate(mask_aug, size=preprocess_image.diff_size).repeat(1, diff.shape[1], 1,
                                                                              1)  # (3, 136, 64, 64)
    diff_re = F.interpolate(diff, size=preprocess_image.diff_size).repeat(3, 1, 1, 1)  # (3, 136, 64, 64)
    # 这就是论文里计算attention时要求同一个facial region
    diff_re = diff_re * mask_re  # (3, 136, 64, 64)
    # dim=1，求出的norm就是(3, 1, 64, 64)，也就是relative position的范数值
    norm = torch.norm(diff_re, dim=1, keepdim=True).repeat(1, diff_re.shape[1], 1, 1)
    # torch.where()函数的作用是按照一定的规则合并两个tensor类型
    norm = torch.where(norm == 0, torch.tensor(1e10), norm)
    diff_re /= norm

    image = image.resize((256, 256), Image.ANTIALIAS)
    real = to_var(transform(image).unsqueeze(0))
    return [real, mask_aug, diff_re]


def preprocess_makeup_gan(image: Image):
    face = futils.dlib.detect(image)

    assert face, "no faces detected"

    # face[0]是第一个人脸，给定图片中只能有一个人脸
    face = face[0]
    image, face = futils.dlib.crop(image, face)

    # detect landmark
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round()
    lms_eye_left = lms[42:48]
    lms_eye_right = lms[36:42]

    # obtain face parsing result
    image = image.resize((512, 512), Image.ANTIALIAS)
    mask = futils.mask.mask(image).resize((256, 256), Image.ANTIALIAS)
    mask = to_var(ToTensor(mask).unsqueeze(0), requires_grad=False)
    mask_lip = (mask == 7).float() + (mask == 9).float()
    mask_face = (mask == 1).float() + (mask == 6).float()

    # 需要抠出 mask_eye
    mask_eyes = torch.zeros_like(mask)
    copy_area(mask_eyes, mask_face, lms_eye_left)
    copy_area(mask_eyes, mask_face, lms_eye_right)
    mask_eyes = to_var(mask_eyes, requires_grad=False)

    mask_list = [mask_lip, mask_face, mask_eyes]
    mask_aug = mask_list[0]
    for i, temp_mask in enumerate(mask_list):
        if i > 0:
            mask_aug = mask_aug + temp_mask
    # mask_aug = torch.cat(mask_list, 0)  # (3, 1, h, w)

    image = image.resize((256, 256), Image.ANTIALIAS)
    real = to_var(transform(image).unsqueeze(0))
    return [real, mask_aug]


def preprocess_train_image(image: Image, mask, diff_re):
    real = transform(image).unsqueeze(0)
    mask_aug = mask
    diff_re = diff_re

    return [real, mask_aug, diff_re]


def get_mask(image: Image, mask_lip_flag, mask_eye_flag, mask_face_flag):
    face = futils.dlib.detect(image)

    assert face, "no faces detected"

    face = face[0]
    image, face = futils.dlib.crop(image, face)

    # detect landmark
    lms = futils.dlib.landmarks(image, face) * 256 / image.width
    lms = lms.round()
    lms_eye_left = lms[42:48]
    lms_eye_right = lms[36:42]

    # obtain face parsing result
    image = image.resize((512, 512), Image.ANTIALIAS)
    current_mask = futils.mask.mask(image).resize((64, 64), Image.ANTIALIAS)
    current_mask = to_var(ToTensor(current_mask).unsqueeze(0), requires_grad=False)
    mask_lip = (current_mask == 7).float() + (current_mask == 9).float()
    mask_face = (current_mask == 1).float() + (current_mask == 6).float()

    mask_eyes = torch.zeros_like(current_mask)
    copy_area(mask_eyes, mask_face, lms_eye_left)
    copy_area(mask_eyes, mask_face, lms_eye_right)
    mask_eyes = to_var(mask_eyes, requires_grad=False)

    mask_list = []
    if str(mask_lip_flag) == '1':
        print('mask_lip_flag')
        mask_list.append(mask_lip)
    if str(mask_eye_flag) == '1':
        print('mask_eye_flag')
        mask_list.append(mask_eyes)
        if str(mask_face_flag) != '1':
            mask_list.append(mask_face)
    if str(mask_face_flag) == '1':
        print('mask_face_flag')
        mask_list.append(mask_face)

    mask2use = mask_list[0]
    for i, current_mask in enumerate(mask_list):
        if i > 0:
            mask2use = mask2use + current_mask

    return mask2use


def data2img(gan_output):
    fake_A_img = gan_output.cpu().clone().squeeze(0)
    # normalize
    min_, max_ = fake_A_img.min(), fake_A_img.max()
    fake_A_img.add_(-min_).div_(max_ - min_ + 1e-5)
    fake_A_img = ToPILImage()(fake_A_img)
    return fake_A_img


# parameter of eye transfer
preprocess_image.eye_margin = 16
# down sample size
preprocess_image.diff_size = (64, 64)
