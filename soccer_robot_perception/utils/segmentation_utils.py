import os
import torch
import gin
import typing
from collections import Counter
import pandas as pd


def seg_label_preprocessor(label_mask):

    label_mask = torch.tensor(label_mask, dtype=torch.float)
    label_mask[label_mask == 3] = 1  # Map ball to field. 0 - BG, 1 - Field, 2 - Lines

    return label_mask


@gin.configurable
def calculate_weight(
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    num_classes: int,
    loader_idx: int,
    report_path: str = "model/",
    write_statistics: bool = True,
) -> torch.Tensor:
    """

    Args:
        data_loaders: typing.Tuple[torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader
        ]
            Train dataloader, validation dataloader, test dataloader

        num_classes: int
            Number of classes in the dataset
        loader_idx: int
            Whether to choose train or test loader for loss calculation

        report_path: str
            Path to save the label count and corresponding class weight values
            In training module: The statistics are saved in the path model/
            In evaluation module: The statistics are saved in the path report/

        write_statistics: bool
            Flag to write statistics of the label count and the corresponding class weights

    Returns:
        segmentation_class_weights: torch.Tensor
            Weights for each class for CE loss weighting as per the chargrid paper and
            https://arxiv.org/pdf/1606.02147.pdf (Section: 5.2) for balancing imbalanced training in segmentation

    """
    train_loader = data_loaders[loader_idx]
    c_segmentation_weights = 1.04
    c_segmentation_weights = torch.scalar_tensor(c_segmentation_weights)
    segmentation_class_counts = Counter()

    # To definitely include all classes atleast once
    labels = range(0, num_classes)
    segmentation_class_counts.update(labels)

    for sample in train_loader:
        if sample["dataset_class"][0] == "segmentation":
            labels = sample["target"][0]
            segmentation_class_counts.update(labels.flatten().tolist())
    total_count_segmentation_labels = max(sum(segmentation_class_counts), 1)
    segmentation_class_weights = torch.tensor(
        [
            1
            / torch.log(
                c_segmentation_weights
                + segmentation_class_counts[label] / total_count_segmentation_labels
            )
            for label in sorted(segmentation_class_counts.keys())
        ]
    )
    sum_seg_class_weights = torch.sum(segmentation_class_weights)
    segmentation_class_weights = torch.div(
        segmentation_class_weights, sum_seg_class_weights
    )

    if write_statistics:

        label_count = [
            segmentation_class_counts[label]
            for label in sorted(segmentation_class_counts.keys())
        ]
        df_columns = ["label_count", "weight"]
        df_count_metrics = pd.DataFrame(columns=df_columns)
        df_count_metrics["label_count"] = label_count
        df_count_metrics["weight"] = segmentation_class_weights.tolist()

        if not os.path.exists(os.path.dirname(report_path)):
            os.makedirs(os.path.dirname(report_path))
        report_path_str = os.path.join(report_path, "")
        excel_writer = pd.ExcelWriter(
            os.path.join("report/", "report_statistics.xlsx"), engine="xlsxwriter"
        )
        df_count_metrics.to_excel(excel_writer, sheet_name="frequency-weight")
        excel_writer.save()

    return segmentation_class_weights


def total_variation_loss(img: torch.tensor, weight: float = 0.001):
    bs, num_channels, height, width = img.shape
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).sum()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).sum()
    tv_loss = weight * (tv_h + tv_w) / (bs)
    return tv_loss


def compute_total_variation_loss_seg(img):
    tv_loss = total_variation_loss(img[:, :2, ::])
    return tv_loss

def compute_total_variation_loss_det(img):
    tv_loss = total_variation_loss(img)
    return tv_loss
