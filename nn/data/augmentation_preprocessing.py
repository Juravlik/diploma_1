import albumentations as A
import torch
import random
import numpy as np
import cv2

IMAGE_SIZE = 256


def lock_deterministic(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def add_padding_to_square(x, **kwargs):
    max_side = max(x.shape)

    return A.PadIfNeeded(
        min_height=max_side, min_width=max_side, always_apply=True, border_mode=cv2.BORDER_CONSTANT
    )(image=x)['image']


def _get_validation_augmentation():
    transforms = [
        A.Lambda(image=add_padding_to_square, mask=add_padding_to_square, always_apply=True),
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE, always_apply=True),
    ]

    return A.Compose(transforms)


def _get_training_augmentation():
    transforms = [

        A.Blur(blur_limit=(3, 3), p=0.05),

        A.Cutout(num_holes=6, max_h_size=12, max_w_size=12, fill_value=0, p=0.07),
        A.OneOf(
            [
                A.ISONoise(color_shift=(0.05, 0.01), intensity=(0.1, 0.5), p=0.1),
                A.IAAAdditiveGaussianNoise(p=0.1),
                A.IAAPerspective(p=0.1),
            ], p=0.3
        ),

        A.RandomBrightnessContrast(p=0.1),

        A.RandomShadow(num_shadows_upper=3, p=0.05),

        A.Flip(p=0.25),

        A.ShiftScaleRotate(border_mode=cv2.BORDER_CONSTANT, p=0.2),

        _get_validation_augmentation(),
    ]

    return A.Compose(transforms)


def get_train_aug_preproc(preprocessing_fn):
    return A.Compose([*_get_training_augmentation()] + [*_get_preprocessing(preprocessing_fn)])


def get_valid_aug_preproc(preprocessing_fn):
    return A.Compose([*_get_validation_augmentation()] + [*_get_preprocessing(preprocessing_fn)])


def to_tensor(x, **kwargs):
    return torch.from_numpy(x.transpose(2, 0, 1).astype('float32'))


def _get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor),
    ]
    return A.Compose(_transform)
