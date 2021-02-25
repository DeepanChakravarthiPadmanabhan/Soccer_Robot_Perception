import os
import typing
import logging
import re
import cv2

import gin
import torch
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


@gin.configurable
class SegmentationDataset(Dataset):
    """"""

    def __init__(
        self,
        root_dir: str,
        transform: typing.Dict,
    ):
        """

        :param root_dir:
        :param transform:
        """

        self.root_dir = root_dir
        LOGGER.info("Root directory read: %s", root_dir)
        self.transform = transform
        self.all_images, self.all_labels = self._get_images_labels_lists(self.root_dir)
        LOGGER.info("Number of samples in detection dataset: %d", len(self.all_images))

    @staticmethod
    def _get_images_labels_lists(root_dir):
        """

        :return:
        """

        image_list = []
        label_list = []
        image_pattern = re.compile("([A-Z0-9a-z_\-]+)\.[pj][pn]g")
        for subdir, directory, files in os.walk(root_dir):
            for document in files:
                image_name = re.search(image_pattern, document)
                if image_name and "/image" in subdir:
                    if os.path.isfile(
                        os.path.join(subdir, image_name.group(1) + ".jpg")
                    ):
                        image_list.append(
                            os.path.join(subdir, image_name.group(1) + ".jpg")
                        )
                    elif os.path.isfile(
                        os.path.join(subdir, image_name.group(1) + ".png")
                    ):
                        image_list.append(
                            os.path.join(subdir, image_name.group(1) + ".png")
                        )

        for file_path in image_list:
            new_file_path = file_path.replace("/image/", "/target/")
            directory, filename = os.path.split(new_file_path)
            name, ext = os.path.splitext(filename)
            if os.path.isfile(os.path.join(directory, name + ".jpg")):
                label_list.append(os.path.join(directory, name + ".jpg"))
            elif os.path.isfile(os.path.join(directory, name + ".png")):
                label_list.append(os.path.join(directory, name + ".png"))

        return image_list, label_list

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx: int):
        # TODO: Fix input and target data structures and format
        image = cv2.imread(self.all_images[idx])
        label = cv2.imread(self.all_labels[idx])

        sample = {
            "image": image,
            "gt_mask": label,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
