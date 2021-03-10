import typing
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix


def calculate_metrics(
    target: np.ndarray,
    output: np.ndarray,
    include_bg_class: bool = False,
    average: str = None,
    num_classes: int = 3,
) -> typing.Tuple:
    """
    Calculates the evaluation metrics precision, recall and f-score for the average
    method passed using sklearn.metrics.precision_recall_fscore_support.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction
        include_bg_class: bool
        Flag to whether include background class in metric calculation
        average: string
        Average argument for sklearn.metrics.precision_recall_fscore_support

    Returns:
        precision, recall, fscore: tuple
        Evaluation metrics: precision, recall and fscore respectively

    """
    if include_bg_class:
        start_label = 0
    else:
        start_label = 1
    metrics = precision_recall_fscore_support(
        target[0],
        output[0],
        labels=list(range(start_label, num_classes)),
        average=average,
    )

    return metrics[0], metrics[1], metrics[2]


def get_confusion_matrix(
    target: np.ndarray,
    output: np.ndarray,
    num_classes: int = 3,
) -> np.ndarray:
    """
    Calculates the confusion matrix normalized for each image with regard to all the character classes
    method passed using sklearn.metrics.confusion_matrix.

    Args:
        target: np.ndarray
        Flattened label prediction
        output: np.ndarray
        Flattened model prediction

    Returns:
        confusion_matrix_: np.ndarray
        Confusion matrix for all the characters

    """

    confusion_matrix_ = confusion_matrix(
        target[0], output[0], labels=list(range(1, num_classes)), normalize="all"
    )
    return confusion_matrix_
