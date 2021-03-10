import typing
import numpy as np
import gin
import torch
import torchvision
import cv2
import matplotlib.pyplot as plt

from soccer_robot_perception.utils.detection_utils import det_label_preprocessor
from soccer_robot_perception.utils.segmentation_utils import seg_label_preprocessor


def get_transform(transform_type: str, params: typing.Dict):
    transform_class = {
        "Resize": Resize,
        "RandomOrientation": RandomOrientation,
        "RandomCrop": RandomCrop,
        "NormalizeImage": NormalizeImage,
    }[transform_type]
    return transform_class(**params)


@gin.configurable
def configure_transforms(config: typing.Dict) -> torchvision.transforms:
    transforms_list = []
    for i in range(len(config)):
        transforms_list.append(get_transform(**config[str(i)]))
    transforms_list.append(ToTensor())
    transform = torchvision.transforms.Compose(transforms_list)
    return transform


class Resize(object):
    """Resize the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, list))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = tuple(output_size)

    def __call__(self, sample: typing.Dict) -> typing.Dict:

        image = sample["image"]

        old_h, old_w = image.shape[0], image.shape[1]
        new_h, new_w = self.output_size

        # resize image
        new_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        sample["image"] = new_image

        if "seg_mask" in sample.keys():
            mask = sample["seg_mask"]
            new_mask = cv2.resize(mask, (int(new_w / 4), int(new_h / 4)), interpolation=cv2.INTER_NEAREST)
            sample["seg_mask"] = new_mask

        width_factor = new_w / old_w
        height_factor = new_h / old_h

        if "det_boxcord" in sample.keys():
            # Resize the bounding box coordinates for the image
            new_bounding_box = []
            for box in sample["det_boxcord"]:
                x_min, y_min, x_max, y_max = box
                x_min = int(np.round(x_min * width_factor))
                y_min = int(np.round(y_min * height_factor))
                x_max = int(np.round(x_max * width_factor))
                y_max = int(np.round(y_max * height_factor))
                new_box = [x_min, y_min, x_max, y_max]
                new_bounding_box.append(new_box)
            sample["det_boxcord"] = new_bounding_box

        return sample


class NormalizeImage(object):
    def __init__(self, mean: float, stddev: float):
        assert isinstance(mean, list) and len(mean) == 3
        assert isinstance(stddev, list) and len(stddev) == 3
        self.mean = mean
        self.stddev = stddev

    def normalize(self, image: np.ndarray) -> np.ndarray:
        # normalize data
        (channel_b, channel_g, channel_r) = cv2.split(image)
        channel_b = (channel_b - self.mean[0]) / self.stddev[0]
        channel_g = (channel_g - self.mean[1]) / self.stddev[1]
        channel_r = (channel_r - self.mean[2]) / self.stddev[2]
        image = cv2.merge((channel_b, channel_g, channel_r))
        return image

    def __call__(self, sample: typing.Dict) -> typing.Dict:
        image = sample["image"]

        sample["image"] = self.normalize(image)

        return sample


class ToTensor(object):



    def __call__(self, sample: typing.Dict) -> typing.Dict:


        if "seg_mask" in sample.keys():
            seg_mask = seg_label_preprocessor(sample["seg_mask"])
            sample["target"] = seg_mask

        if "det_boxcord" in sample.keys():
            det_mask = det_label_preprocessor(bb=sample["det_boxcord"], class_name=sample["det_class"])

            sample["det_boxcord"] = torch.tensor(sample["det_boxcord"], dtype=torch.float)
            sample["target"] = torch.tensor(det_mask, dtype=torch.float)

            sample["det_class"] = torch.tensor(sample["det_class"], dtype=torch.int)

        image = sample["image"]
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        sample["image"] = image

        return sample


class RandomOrientation(object):
    pass


class RandomCrop(object):
    pass
