import os
import logging
import gin
import typing
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from soccer_robot_perception.utils.metrics import (
    calculate_metrics,
    get_confusion_matrix,
    calculate_iou,
    calculate_det_metrics,
    iou_metrics_preprocess,
)
from soccer_robot_perception.utils.detection_utils import center_of_shape, plot_blobs
from scipy.stats import multivariate_normal


LOGGER = logging.getLogger(__name__)


@gin.configurable
def evaluate_model(
    model_path: str,
    report_output_path: str,
    seg_criterion: torch.nn,
    det_criterion: typing.Callable,
    net: torch.nn.Module,
    data_loaders: typing.Tuple[
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
        torch.utils.data.DataLoader,
    ],
    loss_factor: float = 0.5,
    num_classes: int = 3,
    visualize: bool = False,
    input_width: int = 640,
    input_height: int = 480,
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
        The datasets available are: GINI, ICDAR and GINI_ICDAR provided in gin config
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

    device = torch.device("cpu")
    state_test = torch.load(model_path, map_location=device)
    net.load_state_dict(state_test)
    net.eval()

    # instantiate dataset
    train_loader, valid_loader, test_loader = data_loaders

    LOGGER.info("Evaluating Soccer Robot Perception using the model, %s", model_path)
    LOGGER.info("Results will be written to the path, %s", report_output_path)

    LOGGER.info("Ready to start evaluating!")

    df_seg_columns = [
        "seg loss",
        "precision",
        "recall",
        "f1-score",
        "accuracy",
    ]

    df_det_columns = [
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
    df_iou = pd.DataFrame(columns=["bg", "field", "lines"])
    confusion_matrix_array = np.zeros((num_classes, num_classes))
    precision_per_class = np.zeros((num_classes))
    recall_per_class = np.zeros((num_classes))
    f1score_per_class = np.zeros((num_classes))
    accuracy_per_class = np.zeros((num_classes))

    df_det_ball = pd.DataFrame(columns=df_det_columns)
    df_det_robot = pd.DataFrame(columns=df_det_columns)
    df_det_goalpost = pd.DataFrame(columns=df_det_columns)

    for data in test_loader:
        LOGGER.info("Predicting on image: %d", len(df_micro) + 1)

        input_image = data["image"]
        det_out, seg_out = net(input_image)

        dummy_ball_points = [
            [53.0, 91.0, 1.0],
            [61, 81, 1],
            [0, 19, 1],
        ]

        dummy_robot_points = [
            [50, 50, 2],
            [98.0, 12.0, 2.0],
            [14.0, 65.0, 2.0],
            [89.0, 87.0, 2.0],
        ]

        # dummy_goalpost_points = [[13., 109., 3.],
        #                      [13., 112., 3.],
        #                      [13., 113., 3.],
        #                      ]
        dummy_goalpost_points = []

        if visualize:
            plt.subplot(231)
            new_image = input_image[0].permute(1, 2, 0).detach().numpy()
            plt.imshow(cv2.resize(new_image, (160, 120), cv2.INTER_NEAREST))
            plt.title("Input")
            plt.subplot(232)
            plt.imshow(det_out[0][0].detach().numpy())
            plt.title("Det out")
            plt.subplot(233)
            plt.imshow((torch.argmax(seg_out, dim=1)[0].detach().numpy()), cmap="gray")
            plt.title("Seg out")
            blob_map = plot_blobs(dummy_ball_points, 6)
            plt.subplot(234)
            plt.imshow(blob_map)
            plt.title("Blobs")

        det_out_collected = []
        det_target_collected = []
        seg_out_collected = []
        seg_target_collected = []

        # To calculate loss for each data
        for n, i in enumerate(data["dataset_class"]):
            if i == "detection":
                det_target_collected.append(data["target"][n].unsqueeze_(0))
                det_out_collected.append(det_out[n].unsqueeze_(0))
                if visualize:
                    plt.subplot(235)
                    plt.imshow(data["target"][n][2].detach().numpy())
                    plt.title("Det tar")
            else:
                seg_target_collected.append(data["target"][n].unsqueeze_(0))
                seg_out_collected.append(seg_out[n].unsqueeze_(0))
                if visualize:
                    plt.subplot(236)
                    plt.imshow(data["target"][n].numpy(), cmap="gray")
                    plt.title("Seg tar")
        if visualize:
            plt.savefig(
                report_output_path
                + "/output_images/"
                + str(len(df_micro) + 1)
                + "_pred.jpg"
            )
            plt.close()

        if len(det_target_collected) != 0:
            det_target_tensor = torch.cat(det_target_collected, dim=0)
            det_out_tensor = torch.cat(det_out_collected, dim=0)
            det_loss = det_criterion(det_out_tensor, det_target_tensor)

            ball_points = center_of_shape(det_out[0][0].detach().numpy(), 1)
            robot_points = center_of_shape(det_out[0][1].detach().numpy(), 2)
            goalpost_points = center_of_shape(det_out[0][2].detach().numpy(), 3)

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
            ) = calculate_det_metrics(dummy_ball_points, data["blob_centers"][0], 1)

            df_det_ball.loc[len(df_det_ball)] = [
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
            ) = calculate_det_metrics(dummy_robot_points, data["blob_centers"][0], 2)
            df_det_robot.loc[len(df_det_robot)] = [
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
            ) = calculate_det_metrics(dummy_goalpost_points, data["blob_centers"][0], 3)
            df_det_goalpost.loc[len(df_det_goalpost)] = [
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

        else:
            det_loss = torch.tensor(
                0, dtype=torch.float32, requires_grad=True, device=device
            )

            df_det_ball.loc[len(df_det_ball)] = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]

            df_det_robot.loc[len(df_det_robot)] = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]

            df_det_goalpost.loc[len(df_det_goalpost)] = [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]

        if len(seg_target_collected) != 0:
            seg_target_tensor = torch.cat(seg_target_collected, dim=0)
            seg_out_tensor = torch.cat(seg_out_collected, dim=0)
            seg_loss = seg_criterion(seg_out_tensor, seg_target_tensor.long())

            seg_out_max = torch.argmax(seg_out_tensor, dim=1)
            outputs_seg_flatten = torch.flatten(seg_out_max, start_dim=0).unsqueeze_(0)
            labels_seg_flatten = torch.flatten(
                seg_target_tensor, start_dim=0
            ).unsqueeze_(0)

            (
                target_bg_iou_map,
                target_field_iou_map,
                target_lines_iou_map,
            ) = iou_metrics_preprocess(seg_target_tensor)
            (
                output_bg_iou_map,
                output_field_iou_map,
                output_lines_iou_map,
            ) = iou_metrics_preprocess(seg_out_max)

            iou_bg = calculate_iou(target_bg_iou_map, output_bg_iou_map)
            iou_field = calculate_iou(target_field_iou_map, output_field_iou_map)
            iou_lines = calculate_iou(target_lines_iou_map, output_lines_iou_map)
            df_iou.loc[len(df_iou)] = [
                iou_bg.detach().item(),
                iou_field.detach().item(),
                iou_lines.detach().item(),
            ]

            precision, recall, f1score, accuracy = calculate_metrics(
                labels_seg_flatten.detach().numpy(),
                outputs_seg_flatten.detach().numpy(),
                False,
                "micro",
            )
            df_micro.loc[len(df_micro)] = [
                seg_loss.detach().numpy(),
                precision,
                recall,
                f1score,
                accuracy,
            ]

            precision, recall, f1score, accuracy = calculate_metrics(
                labels_seg_flatten.detach().numpy(),
                outputs_seg_flatten.detach().numpy(),
                False,
                "macro",
            )
            df_macro.loc[len(df_macro)] = [
                seg_loss.detach().numpy(),
                precision,
                recall,
                f1score,
                accuracy,
            ]
            image_precision, image_recall, image_f1score, _ = calculate_metrics(
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

        else:
            seg_loss = torch.tensor(
                0, dtype=torch.float32, requires_grad=True, device=device
            )

            df_iou.loc[len(df_iou)] = [0, 0, 0]

            df_micro.loc[len(df_micro)] = [
                seg_loss.detach().numpy(),
                0,
                0,
                0,
                0,
            ]
            df_macro.loc[len(df_macro)] = [
                seg_loss.detach().numpy(),
                0,
                0,
                0,
                0,
            ]

            precision_per_class = precision_per_class + 0
            recall_per_class = recall_per_class + 0
            f1score_per_class = f1score_per_class + 0
            accuracy_per_class = accuracy_per_class + 0

            confusion_matrix_array = confusion_matrix_array + 0

        loss = seg_loss + det_loss
        LOGGER.info(
            "image: %d, loss: %f, segment loss: %f, regression loss: %f",
            len(df_micro),
            loss.item(),
            seg_loss.item(),
            det_loss.item(),
        )
        # break
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
