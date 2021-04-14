import numpy as np
import matplotlib.pyplot as plt
import cv2
import gin
import typing
import torch
import imutils
from scipy.stats import multivariate_normal
import xml.etree.ElementTree as ET

from soccer_robot_perception.utils.constants import CLASS_MAPPING_DETECTION
from soccer_robot_perception.utils.metrics import total_variation_loss


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
def det_label_preprocessor(
    input_width,
    input_height,
    channels,
    bb,
    class_name,
    small_variance=6,
    large_variance=12,
    scale=4,
    visualize_label_masks=False,
):

    label_mask_shrinked = np.zeros(
        (channels, int(input_height / scale), int(input_width / scale))
    )

    robot_map = np.ones((int(input_height / scale), int(input_width / scale)))
    ball_map = np.ones((int(input_height / scale), int(input_width / scale)))
    goalpost_map = np.ones((int(input_height / scale), int(input_width / scale)))
    blob_centers = []
    for box, name in zip(bb, class_name):

        box = [x / scale for x in box]
        if name == CLASS_MAPPING_DETECTION["ball"]:
            ball_heatmap = np.dstack(
                np.mgrid[
                    0 : int(input_height / scale) : 1, 0 : int(input_width / scale) : 1
                ]
            )
            point_x = (box[0] + box[2]) / 2
            point_y = (box[1] + box[3]) / 2
            start_x = int(point_x)
            start_y = int(point_y)

            rv = multivariate_normal(mean=[start_y, start_x], cov=small_variance)
            ball_map = ball_map + rv.pdf(ball_heatmap)
            blob_centers.append((start_y, start_x, name))

        elif name == CLASS_MAPPING_DETECTION["robot"]:
            robot_heatmap = np.dstack(
                np.mgrid[
                    0 : int(input_height / scale) : 1, 0 : int(input_width / scale) : 1
                ]
            )
            point_x = (box[0] + box[2]) / 2
            point_y = box[3]
            start_x = int(point_x)
            start_y = int(point_y)

            rv = multivariate_normal(mean=[start_y, start_x], cov=large_variance)
            robot_map = robot_map + rv.pdf(robot_heatmap)
            blob_centers.append((start_y, start_x, name))

        elif name == CLASS_MAPPING_DETECTION["goalpost"]:
            goalpost_heatmap = np.dstack(
                np.mgrid[
                    0 : int(input_height / scale) : 1, 0 : int(input_width / scale) : 1
                ]
            )
            point_x = box[0]
            point_y = box[3]
            start_x = int(point_x)
            start_y = int(point_y)
            rv = multivariate_normal(mean=[start_y, start_x], cov=small_variance)
            goalpost_map = goalpost_map + rv.pdf(goalpost_heatmap)
            blob_centers.append((start_y, start_x, name))

            goalpost_heatmap = np.dstack(
                np.mgrid[
                    0 : int(input_height / scale) : 1, 0 : int(input_width / scale) : 1
                ]
            )
            point_x = box[2]
            point_y = box[3]
            start_x = int(point_x)
            start_y = int(point_y)
            rv = multivariate_normal(mean=[start_y, start_x], cov=small_variance)
            goalpost_map = goalpost_map + rv.pdf(goalpost_heatmap)
            blob_centers.append((start_y, start_x, name))

    if visualize_label_masks:
        plt.imshow(label_mask_shrinked)
        plt.show()

    label_mask_shrinked[0] = ball_map
    label_mask_shrinked[1] = robot_map
    label_mask_shrinked[2] = goalpost_map
    label_mask_shrinked = torch.tensor(label_mask_shrinked, dtype=torch.float)
    return label_mask_shrinked, blob_centers


def center_of_shape(image: np.ndarray, name: int):
    """
    To find centers of the contours in the input image.

    :param image: Image to find the contours.
    :param name: Integer denoting the classes - 1: ball, 2: robot, 3: goalpost

    :return:
    out_centers: All contour centers in the input image.
    """
    centers = []
    blurred = cv2.GaussianBlur(image * 255, (3, 3), 0)
    thresh, im_bw = cv2.threshold(blurred, 250, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(
        im_bw.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        M = cv2.moments(c)
        if M["m00"] > 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cY, cX, name))
    return centers


def plot_blobs(points: typing.List, variance: float):
    """
    Function to plot blobs predicted and preprocessed by the network.

    :param points:
    :param variance:
    :return:
    """
    blob_map = np.ones((120, 160))

    for i in points:
        blob_point = [i[0], i[1]]
        pos = np.dstack(np.mgrid[0:120:1, 0:160:1])
        rv = multivariate_normal(mean=blob_point, cov=variance)
        blob_map = blob_map + rv.pdf(pos)
    return blob_map

def compute_total_variation_loss_det(img, weight: float = 0.0001):
    tv_loss = total_variation_loss(img, weight)
    return tv_loss


def det_image_processor_wandb(input_image, model_det_out, target):
    fig = plt.figure()
    plt.subplot(131)
    new_image = cv2.resize(input_image.permute(1, 2, 0).numpy(), (160, 120),
                           interpolation=cv2.INTER_NEAREST)
    plt.imshow(new_image)
    plt.title('Input')
    plt.subplot(132)
    plt.imshow((model_det_out.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    plt.title('Det out')
    plt.subplot(133)
    plt.imshow((target.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    plt.title('Target')
    plt.close()

    return fig