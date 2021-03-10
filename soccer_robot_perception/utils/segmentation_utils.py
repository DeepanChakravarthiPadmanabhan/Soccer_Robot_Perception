import os
import torch
import gin
import typing
from collections import Counter
import pandas as pd


def seg_label_preprocessor(label_mask):

    label_mask = torch.sum(torch.tensor(label_mask, dtype=torch.float), dim=2)
    label_mask[label_mask == 0.] = 0  # Background
    label_mask[label_mask == 128.] = 1 # Field
    label_mask[label_mask == 256.] = 2  # Lines

    # label_mask.unsqueeze_(dim=2)
    # label_mask = label_mask.expand(56, 56, 3)

    label_mask = torch.nn.functional.one_hot(label_mask.long(),  num_classes=3)
    label_mask = label_mask.permute(2, 0, 1)
    # label_mask[1] = label_mask[1] * 2 # Field
    # label_mask[2] = label_mask[2] * 3 # Lines

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
        if sample["dataset_class"][0] == 'segmentation':
            labels = sample["target"][0]
            # labels = torch.argmax(labels, dim=1)
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
            os.path.join(report_path_str, "report_statistics.xlsx"), engine="xlsxwriter"
        )
        df_count_metrics.to_excel(excel_writer, sheet_name="frequency-weight")
        excel_writer.save()

    return segmentation_class_weights