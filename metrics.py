import logging

from ignite.metrics import EpochMetric
import numpy as np


class AUCPR(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def aucpr_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import precision_recall_curve, auc
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            logging.debug(f'AUCPR {y_pred.shape}, {y_true.shape}, {sum(y_true)}')
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            return auc(recall, precision)
        super().__init__(aucpr_compute_fn, output_transform=output_transform)


class AP(EpochMetric):
    def __init__(self, average=None, output_transform=lambda x: x):
        def average_precision_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import average_precision_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            logging.debug(f'AP {y_pred.shape}, {y_true.shape}, {sum(y_true)}')
            return average_precision_score(y_true, y_pred, average=average)
        super().__init__(average_precision_compute_fn, output_transform=output_transform)


class AUROC(EpochMetric):
    def __init__(self, average=None, output_transform=lambda x: x):
        def auroc_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import roc_auc_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return roc_auc_score(y_true, y_pred, average)

        super().__init__(auroc_compute_fn, output_transform=output_transform)


class Output(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def _compute_fn(y_preds, y_targets):

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()
            return y_pred, y_true

        super().__init__(_compute_fn, output_transform=output_transform)


class Accuracy(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def accuracy_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import accuracy_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy().argmax(-1)
            return accuracy_score(y_true, y_pred)

        super().__init__(accuracy_compute_fn, output_transform=output_transform)


class Kappa(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def kappa_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import cohen_kappa_score
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy().argmax(-1)

            return cohen_kappa_score(y_true, y_pred)

        super().__init__(kappa_compute_fn, output_transform=output_transform)


class MAD(EpochMetric):
    def __init__(self, output_transform=lambda x: x):
        def mad_compute_fn(y_preds, y_targets):
            try:
                from sklearn.metrics import mean_absolute_error
            except ImportError:
                raise RuntimeError("This contrib module requires sklearn to be installed.")

            y_true = y_targets.numpy()
            y_pred = y_preds.numpy()

            return mean_absolute_error(y_true, y_pred)

        super().__init__(mad_compute_fn, output_transform=output_transform)
