import logging
import os
import gin
from typing import Dict, Any, List, Union

from torch import Tensor
from torch.utils.data._utils.collate import default_collate


@gin.configurable
def custom_collate_segmentation(
    batch: List[Dict[str, Tensor]], groundtruth: bool = True
) -> Dict[str, Union[Any, List[Tensor]]]:
    """This custom collate function allows variable batch size for all modalities.

    It can be used to overwrite the default collate function in torch.utils.dataloader.Dataloader.
    The function allows us to have variable-sized tensors (e.g. box coordinates) and
    have batch size > 1, by combining them as list of tensors rather than including the
    batch in dimension 0.

    :param batch: list of data samples, where data samples are dictionary
    :param groundtruth: flag if groundtruth is present in batch or not
    :return: batch as dictionary to tensors or lists of tensors
    """

    batch = default_collate(batch)
    return batch


@gin.configurable
def custom_collate_detection(
    batch: List[Dict[str, Tensor]], groundtruth: bool = True
) -> Dict[str, Union[Any, List[Tensor]]]:
    """This custom collate function allows variable batch size for all modalities.

    It can be used to overwrite the default collate function in torch.utils.dataloader.Dataloader.
    The function allows us to have variable-sized tensors (e.g. box coordinates) and
    have batch size > 1, by combining them as list of tensors rather than including the
    batch in dimension 0.

    :param batch: list of data samples, where data samples are dictionary
    :param groundtruth: flag if groundtruth is present in batch or not
    :return: batch as dictionary to tensors or lists of tensors
    """

    bb_list = []
    for element in batch:
        bb_list.append(element.pop("gt_boxcord"))
    batch = default_collate(batch)
    batch["gt_boxcord"] = bb_list
    return batch


@gin.configurable
def custom_collate_alldata(
    batch: List[Dict[str, Tensor]], groundtruth: bool = True
) -> Dict[str, Union[Any, List[Tensor]]]:
    """This custom collate function allows variable batch size for all modalities.
    This collate function combines Gini and ICDAR dataset.

    It can be used to overwrite the default collate function in torch.utils.dataloader.Dataloader.
    The function allows us to have variable-sized tensors (e.g. box coordinates) and
    have batch size > 1, by combining them as list of tensors rather than including the
    batch in dimension 0.

    :param batch: list of data samples, where data samples are dictionary
    :param groundtruth: flag if groundtruth is present in batch or not
    :return: batch as dictionary to tensors or lists of tensors
    """
    box_coordinates_list = []
    class_list = []
    mask_list = []

    for element in batch:
        # Clean Detection Dataset
        if "gt_boxcord" in element.keys():
            box_coordinates_list.append(element.pop("gt_boxcord"))
            class_list.append(element.pop("class"))
            mask_list.append([])

        # Clean Segmentation Dataset
        if "gt_mask" in element.keys():
            box_coordinates_list.append([])
            class_list.append([])
            mask_list.append(element.pop("gt_mask"))

    batch = default_collate(batch)
    batch["gt_boxcord"] = box_coordinates_list
    batch["gt_mask"] = mask_list
    batch["class"] = class_list

    return batch
