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

    box_coordinates_list = []
    class_list = []
    blob_centers_list = []
    for element in batch:
        box_coordinates_list.append(element.pop("det_boxcord"))
        class_list.append(element.pop("det_class"))
        blob_centers_list.append(element.pop("blob_centers"))
    batch = default_collate(batch)
    batch["det_boxcord"] = box_coordinates_list
    batch["det_class"] = class_list
    batch["blob_centers"] = blob_centers_list
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
    blob_centers_list = []
    mask_list = []
    seg_mask_list = []

    for element in batch:
        # Clean Detection Dataset

        if element["dataset_class"] == "detection":
            box_coordinates_list.append(element.pop("det_boxcord"))
            class_list.append(element.pop("det_class"))
            blob_centers_list.append(element.pop("blob_centers"))
            seg_mask_list.append([])

        # Clean Segmentation Dataset
        if element["dataset_class"] == "segmentation":
            box_coordinates_list.append([])
            class_list.append([])
            blob_centers_list.append([])

            seg_mask_list.append(element.pop("seg_mask"))

    batch = default_collate(batch)
    batch["det_boxcord"] = box_coordinates_list
    batch["det_class"] = class_list
    batch["blob_centers"] = blob_centers_list
    batch["seg_mask"] = seg_mask_list

    return batch
