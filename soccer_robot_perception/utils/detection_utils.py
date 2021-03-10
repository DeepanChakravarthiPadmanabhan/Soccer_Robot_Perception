import numpy as np
import matplotlib.pyplot as plt
import cv2
import gin
import torch

import xml.etree.ElementTree as ET
from soccer_robot_perception.utils.constants import CLASS_MAPPING_DETECTION


def read_xml_file(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    bb_list = []
    class_list = []

    for boxes in root.iter("object"):
        filename = root.find("filename").text
        class_list.append(CLASS_MAPPING_DETECTION[boxes.find("name").text])
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        bb_list.append([xmin, ymin, xmax, ymax])

    return class_list, bb_list

@gin.configurable
def det_label_preprocessor(input_width,
                           input_height,
                           channels,
                           bb,
                           class_name,
                           variance=4,
                           scale=4,
                           visualize_label_masks=False):

    label_mask_shrinked = np.ones((int(input_height / scale), int(input_width / scale), channels))
    label_mask = np.ones((input_height, input_width, channels))

    for box, name in zip(bb, class_name):

        if name == CLASS_MAPPING_DETECTION["ball"]:
            point_x = (box[0] + box[2]) / 2
            point_y = (box[1] + box[3]) / 2
            point = (int(point_x), int(point_y))
            label_mask = cv2.circle(label_mask, point, variance, (255, 0, 0), -1, lineType=cv2.LINE_AA)
            label_mask_shrinked = cv2.circle(label_mask_shrinked, (int(point[0] / scale), int(point[1] / scale)), int(variance / scale), (1, 0, 0), -1, lineType=cv2.LINE_AA)

        elif name == CLASS_MAPPING_DETECTION["robot"]:
            point_x = (box[0] + box[2]) / 2
            point_y = (box[3])
            point = (int(point_x), int(point_y))
            label_mask = cv2.circle(label_mask, point, variance, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            label_mask_shrinked = cv2.circle(label_mask_shrinked, (int(point[0] / scale), int(point[1] / scale)), int(variance / scale), (0, 2, 0), -1,
                                             lineType=cv2.LINE_AA)

        elif name == CLASS_MAPPING_DETECTION["goalpost"]:
            point_x = (box[0] + box[2]) / 2
            point_y = (box[3])
            point = (int(point_x), int(point_y))
            label_mask = cv2.circle(label_mask, point, 4, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            label_mask_shrinked = cv2.circle(label_mask_shrinked, (int(point[0] / scale), int(point[1] / scale)), int(variance / scale), (0, 0, 3), -1,
                                             lineType=cv2.LINE_AA)

    if visualize_label_masks:
        plt.subplot(121)
        plt.imshow(label_mask)
        plt.subplot(122)
        plt.imshow(label_mask_shrinked)
        plt.show()

    label_mask_shrinked = torch.tensor(label_mask_shrinked, dtype=torch.float)
    label_mask_shrinked = label_mask_shrinked.permute(2, 0, 1)

    return label_mask_shrinked
