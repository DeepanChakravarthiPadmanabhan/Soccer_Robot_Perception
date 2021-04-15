import os
import logging
import gin
import typing
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import git

from soccer_robot_perception.utils.metrics import (
    calculate_seg_metrics,
    get_confusion_matrix,
    calculate_iou,
    calculate_det_metrics,
    iou_metrics_preprocess,
)
from soccer_robot_perception.utils.detection_utils import center_of_shape, plot_blobs


LOGGER = logging.getLogger(__name__)


@gin.configurable
def evaluate_model(
    model_path: str,
    report_output_path: str,
    seg_criterion: torch.nn,
    det_criterion: typing.Callable,
    net: torch.nn.Module,
    seg_data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    det_data_loaders: typing.Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
        ],
    wandb_key,
    loss_factor: float = 0.5,
    num_classes: int = 3,
    visualize: bool = False,
    input_width: int = 640,
    input_height: int = 480,
    run_name="soccer-robot",
) -> None:
    """
    This function evaluates the model trained on a set of test image and provides a report with evaluation metrics.
    The evaluation metrics used are: Precision, Recall and F-score.
    The module also aids in visualizing the predictions and groundtruth labels.

    Args:
        model_path: string
        Path of the model to be used for inference
        report_output_path: string
        Path for writing the inference output report with evaluation metrics and visualization images
        criterion: torch.nn
        Loss type for evaluation
        net: torch.nn.Module
        Network architecture of the model trained
        dataset_class: typing.Union[Dataset, ConcatDataset]
        Specifies the dataset to use for inference.
        The datasets available are: SEGMENTATION, DETECTION provided in gin config
        collate: typing.Callable
        Function in utils custom_collate for gathering the dataset keys
        visualize: bool
        To visualize the model predictions alongside groundtruth prediction
    """
    if not os.path.exists(os.path.dirname(report_output_path)):
        LOGGER.info(
            "Output directory does not exist. Creating directory %s",
            os.path.dirname(report_output_path),
        )
        os.makedirs(os.path.dirname(report_output_path))
    if visualize and (
        not os.path.exists(os.path.join(report_output_path, "output_images"))
    ):
        os.makedirs(os.path.join(report_output_path, "output_images"))
        LOGGER.info(
            "Saving images in the directory: %s",
            os.path.join(report_output_path, "output_images"),
        )


    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    LOGGER.info('Evaluating using the git sha: %s', sha)

    device = torch.device("cpu")
    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    net.eval()

    # instantiate dataset
    train_seg_loader, valid_seg_loader, test_seg_loader = seg_data_loaders
    train_det_loader, valid_det_loader, test_det_loader = det_data_loaders

    LOGGER.info("Evaluating Soccer Robot Perception using the model, %s", model_path)
    LOGGER.info("Results will be written to the path, %s", report_output_path)

    LOGGER.info("Ready to start evaluating!")

    df_seg_columns = [
        "id",
        "seg loss",
        "precision",
        "recall",
        "f1-score",
        "accuracy",
    ]

    df_det_columns = [
        "id",
        "det loss",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "f1-score",
        "accuracy",
        "fdr",
    ]

    df_micro = pd.DataFrame(columns=df_seg_columns)
    df_macro = pd.DataFrame(columns=df_seg_columns)
    df_iou = pd.DataFrame(columns=["id", "bg", "field", "lines"])
    confusion_matrix_array = np.zeros((num_classes, num_classes))
    precision_per_class = np.zeros((num_classes))
    recall_per_class = np.zeros((num_classes))
    f1score_per_class = np.zeros((num_classes))
    accuracy_per_class = np.zeros((num_classes))

    df_det_ball = pd.DataFrame(columns=df_det_columns)
    df_det_robot = pd.DataFrame(columns=df_det_columns)
    df_det_goalpost = pd.DataFrame(columns=df_det_columns)

    test_loader_list = [test_det_loader, test_seg_loader]

    example = 0

    for loader in test_loader_list:
        for data in loader:
            example += 1
            LOGGER.info("Predicting on image: %d at filepath: %s", example, data["filename"][0])

            input_image = data["image"]
            det_out, seg_out = net(input_image)

            # To calculate loss for each data
            if (data["dataset_class"][0]) == "detection":
                det_loss = det_criterion(det_out, data["det_target"])
                seg_loss = torch.tensor(
                    0, dtype=torch.float32, requires_grad=True, device=device
                )
            if (data["dataset_class"][0]) == "segmentation":
                seg_loss = seg_criterion(seg_out, data["seg_target"].long())
                det_loss = torch.tensor(
                    0, dtype=torch.float32, requires_grad=True, device=device
                )

            ball_points = center_of_shape(det_out[0][0].detach().numpy(), 5, 1)
            robot_points = center_of_shape(det_out[0][1].detach().numpy(), 250, 2)
            goalpost_points = center_of_shape(det_out[0][2].detach().numpy(), 120, 3)

            blob_map = np.zeros(
                (3, int(input_height / 4), int(input_width / 4))
            )
            ball_map = plot_blobs(ball_points, 4)
            robot_map = plot_blobs(robot_points, 6)
            goalpost_map = plot_blobs(goalpost_points, 4)
            blob_map[0] = ball_map
            blob_map[1] = robot_map
            blob_map[2] = goalpost_map

            if (data["dataset_class"][0]) == "detection":
                (
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ) = calculate_det_metrics(ball_points, data["blob_centers"][0], 1)

                df_det_ball.loc[len(df_det_ball)] = [
                    example,
                    det_loss.detach().numpy(),
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ]

                (
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ) = calculate_det_metrics(robot_points, data["blob_centers"][0], 2)
                df_det_robot.loc[len(df_det_robot)] = [
                    example,
                    det_loss.detach().numpy(),
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ]

                (
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ) = calculate_det_metrics(goalpost_points, data["blob_centers"][0], 3)
                df_det_goalpost.loc[len(df_det_goalpost)] = [
                    example,
                    det_loss.detach().numpy(),
                    tp,
                    fp,
                    tn,
                    fn,
                    precision,
                    recall,
                    f1,
                    accuracy,
                    fdr,
                ]

            if (data["dataset_class"][0]) == "segmentation":
                seg_out_max = torch.argmax(seg_out, dim=1)
                outputs_seg_flatten = torch.flatten(seg_out_max, start_dim=0).unsqueeze_(0)
                labels_seg_flatten = torch.flatten(
                    data["seg_target"], start_dim=0
                ).unsqueeze_(0)

                (
                    target_bg_iou_map,
                    target_field_iou_map,
                    target_lines_iou_map,
                ) = iou_metrics_preprocess(data["seg_target"])
                (
                    output_bg_iou_map,
                    output_field_iou_map,
                    output_lines_iou_map,
                ) = iou_metrics_preprocess(seg_out_max)

                iou_bg = calculate_iou(target_bg_iou_map, output_bg_iou_map)
                iou_field = calculate_iou(target_field_iou_map, output_field_iou_map)
                iou_lines = calculate_iou(target_lines_iou_map, output_lines_iou_map)
                df_iou.loc[len(df_iou)] = [
                    example,
                    iou_bg.detach().item(),
                    iou_field.detach().item(),
                    iou_lines.detach().item(),
                ]

                precision, recall, f1score, accuracy = calculate_seg_metrics(
                    labels_seg_flatten.detach().numpy(),
                    outputs_seg_flatten.detach().numpy(),
                    False,
                    "micro",
                )
                df_micro.loc[len(df_micro)] = [
                    example,
                    seg_loss.detach().numpy(),
                    precision,
                    recall,
                    f1score,
                    accuracy,
                ]

                precision, recall, f1score, accuracy = calculate_seg_metrics(
                    labels_seg_flatten.detach().numpy(),
                    outputs_seg_flatten.detach().numpy(),
                    False,
                    "macro",
                )
                df_macro.loc[len(df_macro)] = [
                    example,
                    seg_loss.detach().numpy(),
                    precision,
                    recall,
                    f1score,
                    accuracy,
                ]
                image_precision, image_recall, image_f1score, _ = calculate_seg_metrics(
                    labels_seg_flatten.detach().numpy(),
                    outputs_seg_flatten.detach().numpy(),
                    True,
                )
                precision_per_class = precision_per_class + image_precision
                recall_per_class = recall_per_class + image_recall
                f1score_per_class = f1score_per_class + image_f1score

                confusion_matrix_array = confusion_matrix_array + get_confusion_matrix(
                    labels_seg_flatten.detach().numpy(),
                    outputs_seg_flatten.detach().numpy(),
                )

                accuracy_per_class = accuracy_per_class + (
                    confusion_matrix_array.diagonal() / confusion_matrix_array.sum(axis=1)
                )

            loss = seg_loss + det_loss
            LOGGER.info(
                "image: %d, loss: %f, segment loss: %f, regression loss: %f",
                example,
                loss.item(),
                seg_loss.item(),
                det_loss.item(),
            )

            if visualize:
                new_image = input_image[0].permute(1, 2, 0).detach().numpy()
                plt.subplot(231)
                plt.imshow(cv2.resize(new_image, (160, 120), cv2.INTER_NEAREST))
                plt.title("Input")
                plt.subplot(232)
                plt.imshow((det_out[0].detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                plt.title("Det out")
                plt.subplot(233)
                plt.imshow((torch.argmax(seg_out, dim=1)[0].detach().numpy()), cmap="gray")
                plt.title("Seg out")
                if (data["dataset_class"][0]) == "detection":
                    plt.subplot(234)
                    plt.imshow((data["det_target"][0].detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                    plt.title("Det tar")
                else:
                    plt.subplot(234)
                    plt.imshow(np.zeros((120, 160)), cmap='gray')
                    plt.title("Det tar")
                if (data["dataset_class"][0]) == "segmentation":
                    plt.subplot(235)
                    plt.imshow(data["seg_target"][0].numpy(), cmap="gray")
                    plt.title("Seg tar")
                else:
                    plt.subplot(235)
                    plt.imshow(np.zeros((120, 160)), cmap='gray')
                    plt.title("Seg tar")
                plt.subplot(236)
                plt.imshow((np.transpose(blob_map, (1, 2, 0)) * 255).astype(np.uint8))
                plt.title("Blobs")
                plt.savefig(
                    report_output_path
                    + "/output_images/"
                    + str(example)
                    + "_pred_local.jpg"
                )
                plt.close()

                plt.figure(figsize=(14, 5))
                new_image = input_image[0].permute(1, 2, 0).detach().numpy()
                plt.subplot(131)
                plt.imshow(cv2.resize(new_image, (160, 120), cv2.INTER_NEAREST))
                plt.title("Input", fontsize=24)
                plt.subplot(132)
                plt.imshow((torch.argmax(seg_out, dim=1)[0].detach().numpy()), cmap="gray")
                plt.title("Segmentation", fontsize=24)
                plt.subplot(133)
                plt.imshow((np.transpose(blob_map, (1, 2, 0)) * 255).astype(np.uint8))
                plt.title("Detection", fontsize=24)
                plt.tight_layout()
                plt.savefig(
                    report_output_path
                    + "/output_images/"
                    + str(example)
                    + "_pred_final.jpg"
                )

                plt.close()


    df_iou.loc["mean"] = df_iou.mean()
    df_micro.loc["mean"] = df_micro.mean()
    df_macro.loc["mean"] = df_macro.mean()
    df_confusion_matrix = pd.DataFrame(confusion_matrix_array / len(df_micro))
    df_precision_per_class = pd.DataFrame(precision_per_class / len(df_micro))
    df_recall_per_class = pd.DataFrame(recall_per_class / len(df_micro))
    df_f1score_per_class = pd.DataFrame(f1score_per_class / len(df_micro))
    df_accuracy_per_class = pd.DataFrame(accuracy_per_class / len(df_micro))

    df_det_ball.loc["mean"] = df_det_ball.mean()
    df_det_robot.loc["mean"] = df_det_robot.mean()
    df_det_goalpost.loc["mean"] = df_det_goalpost.mean()

    excel_writer = pd.ExcelWriter(
        os.path.join(report_output_path, "report.xlsx"), engine="xlsxwriter"
    )
    df_micro.to_excel(excel_writer, sheet_name="micro")
    df_macro.to_excel(excel_writer, sheet_name="macro")
    df_iou.to_excel(excel_writer, sheet_name="iou")
    df_confusion_matrix.to_excel(excel_writer, sheet_name="normalized_confusion_matrix")
    df_precision_per_class.to_excel(excel_writer, sheet_name="precision_per_class")
    df_recall_per_class.to_excel(excel_writer, sheet_name="recall_per_class")
    df_f1score_per_class.to_excel(excel_writer, sheet_name="f1score_per_class")
    df_accuracy_per_class.to_excel(excel_writer, sheet_name="accuracy_per_class")
    df_det_ball.to_excel(excel_writer, sheet_name="ball_det")
    df_det_robot.to_excel(excel_writer, sheet_name="robot_det")
    df_det_goalpost.to_excel(excel_writer, sheet_name="goalpost_det")

    excel_writer.save()
    LOGGER.info("Results were written to %s", report_output_path)
