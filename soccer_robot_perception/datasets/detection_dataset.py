import os
import typing
import logging
import re
import cv2

import gin
import torch
from torch.utils.data import Dataset
from soccer_robot_perception.utils.detection_utils import read_xml_file

LOGGER = logging.getLogger(__name__)


@gin.configurable
class DetectionDataset(Dataset):
    """"""

    def __init__(
        self,
        root_dir: str,
        transform: typing.Dict,
        detection_sample_count: int,
    ):
        """

        :param root_dir:
        :param transform:
        """

        self.root_dir = root_dir
        LOGGER.info("Root directory read: %s", root_dir)
        self.transform = transform
        self.detection_sample_count = detection_sample_count
        self.all_images, self.all_labels = self._get_images_labels_lists(self.root_dir, self.detection_sample_count)
        LOGGER.info("Number of samples in detection dataset: %d", len(self.all_images))

    @staticmethod
    def _get_images_labels_lists(root_dir, detection_sample_count):
        """

        :return:
        """

        image_list = []
        label_list = []
        image_pattern = re.compile("([A-Z0-9a-z_\-]+)\.[pj][pn]g")
        for subdir, directory, files in os.walk(root_dir):
            for document in files:
                image_name = re.search(image_pattern, document)
                if image_name:
                    if os.path.isfile(
                        os.path.join(subdir, image_name.group(1) + ".jpg")
                    ):
                        image_list.append(
                            os.path.join(subdir, image_name.group(1) + ".jpg")
                        )
                    else:
                        image_list.append(
                            os.path.join(subdir, image_name.group(1) + ".png")
                        )
                    if os.path.isfile(
                        os.path.join(subdir, image_name.group(1) + ".xml")
                    ):
                        label_list.append(
                            os.path.join(subdir, image_name.group(1) + ".xml")
                        )
                    else:
                        image_list.pop()
        image_list = image_list[:detection_sample_count]
        label_list = image_list[:detection_sample_count]
        return image_list, label_list

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx: int):
        # TODO: Fix input and target data structures and format
        image = cv2.imread(self.all_images[idx]) / 255.0
        name, bb_list = read_xml_file(self.all_labels[idx])

        sample = {
            "filename": self.all_images[idx],
            "dataset_class": "detection",
            "image": image,
            "det_boxcord": bb_list,
            "det_class": name,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
