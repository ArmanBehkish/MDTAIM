from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import os
import copy as cp
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .config import Config
from .utility import md_plot, plot_plotly


class PreprocessData:
    """
    Preprocess the data,
    load the data and the ground truth,
    and provide methods to access and plot the data
    """

    def __init__(self, logger: logging.Logger, config: Config) -> None:
        self.logger: logging.Logger = logger
        self.config: Config = config
        self.data: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.padded_labels: Optional[np.ndarray] = None

    def load_data(self) -> None:
        """
        Load the dataset and the ground truth
        """
        data_config = self.config.get_config()["data"]
        # load the dataset
        # if dataset file extension is .csv & first column is the index
        if data_config["dataset_file_name"].endswith(".csv"):
            try:
                data_path = os.path.join(
                    data_config["dataset_path"], data_config["dataset_file_name"]
                )
                self._df_data = pd.read_csv(data_path, index_col=0)
            except FileNotFoundError:
                self.logger.critical(f"dataset file not found: {data_path}")

            self.logger.info(
                f"dataset {data_config['dataset_file_name']} loaded successfully!"
            )
            self.logger.debug(f"loaded dataset: \n{self._df_data.head()}")
            self.data = self._df_data.values

        # load the ground truth
        # if ground truth file extension is .csv & first column is the index
        try:
            gt_path = os.path.join(
                data_config["dataset_path"], data_config["dataset_gt_file_name"]
            )
            self._df_gt = pd.read_csv(gt_path, index_col=0)
        except FileNotFoundError:
            self.logger.error(f"ground truth file not found: {gt_path}")
        self.logger.info(
            f"ground truth {data_config['dataset_gt_file_name']} loaded successfully!"
        )
        self.logger.debug(f"loaded ground truth: \n{self._df_gt.head()}")
        self.labels = self._df_gt.values

        self.logger.debug(f"data shape: {self.data.shape}")
        self.logger.debug(f"labels shape: {self.labels.shape}")

    def preprocess_data(self):
        pass

    def get_data_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.data is None or self.labels is None:
            self.logger.error("Trying to fetch unloaded data or labels!")
            raise ValueError("Data or labels are not loaded!")
        else:
            return self.data, self.labels

    @staticmethod
    def get_state_intervals(gt: np.ndarray) -> Dict[int, List[List[int]]]:
        """
        gets a 1_D np array of labels
        returns a dict in the form
        {0: [[0, 189], [224, 504], [539, 999]], 1: [[190, 223], [505, 538]]}); specifying the range of 0's and 1's
        """

        diff = np.diff(gt, axis=-1)
        idxs = np.array(np.nonzero(diff))
        idxs = idxs.flatten()
        idxs += 1
        idxs = np.concatenate([[0], idxs, [np.size(gt)]])
        state_intervals = defaultdict(list)
        for p in range(1, len(idxs)):
            state_intervals[gt[idxs[p - 1]]].append([idxs[p - 1], idxs[p] - 1])

        return state_intervals

    def convert_to_standard_label(
        self, raw_label: List[List[int]], num_of_samples: int, num_of_dims: int
    ) -> np.ndarray:
        """
        convert labels like raw_label to starndard numpy array labels
        Input:
        raw_label: List of lists, where each inner list contains a timestamp and dimension indices
                        11810,24,15,28
                        12760,21,26,5
        num_of_samples: Total number of samples in the dataset
        num_of_dims : Total number of dimensions
        pad_size: Size of padding to apply around each label
        Output:
        labels: 2D numpy array of shape (num_of_samples, num_of_dims) containing the padded labels
        """
        data_config = self.config.get_config()["data"]
        pad_size = data_config["label_pad_size"]

        labels = np.zeros((num_of_samples, num_of_dims))
        for i in raw_label:
            idx = i[0]
            dims = i[1:]
            labels[idx - pad_size : idx + pad_size, dims] = 1
            if idx - pad_size < 0:
                labels[0 : idx + pad_size, dims] = 1
            elif idx + pad_size > num_of_samples:
                labels[idx - pad_size :, dims] = 1
        self.labels = labels
        return labels

    def pad_labels(
        self,
        labels: Optional[np.ndarray] = None,
        raw_label: Optional[List[List[int]]] = None,
    ) -> np.ndarray:
        """
        Pads the labels in a multi-dimensional ndarray.

        pads one from beggining and end by pad_size

        Parameters:
        lables (numpy.ndarray): A 2D numpy array of shape (num_of_samples, num_of_dims) containing the labels to be padded.

        pad_size (int): The size of padding to apply around each label.

        raw_label (list of list, optional): If provided, a list of lists where each inner list contains an index and dimension indices. Default is None.

        Returns:
        numpy.ndarray: A 2D numpy array of the same shape as input, with padded labels.
        """
        data_config = self.config.get_config()["data"]
        pad_size = data_config["label_pad_size"]
        self.logger.debug(f"pad_size: {pad_size}")

        if labels is None:
            labels = self.labels
        if labels is None:
            raise ValueError("Labels are not loaded or provided.")

        num_of_samples, num_of_dims = labels.shape
        labels_padded = cp.deepcopy(labels)
        if raw_label is not None:
            return self.convert_to_standard_label(
                raw_label, num_of_samples, num_of_dims
            )
        else:
            for i in np.arange(labels.shape[1]):
                intervals = np.asarray(self.get_state_intervals(labels_padded[:, i])[1])
                self.logger.debug(f"intervals for column {i}: {intervals}")
                for j in intervals:
                    labels_padded[j[0] - pad_size : j[1] + pad_size, i] = 1
                    if j[0] - pad_size < 0:
                        labels_padded[0 : j[1] + pad_size, i] = 1
                    elif j[1] + pad_size > num_of_samples:
                        labels_padded[j[0] - pad_size :, i] = 1
                # ---> what is this intervals for?
                intervals = np.asarray(self.get_state_intervals(labels_padded[:, i])[1])
                self.logger.debug(
                    f"intervals for column {i} after padding: {intervals}"
                )
            self.padded_labels = labels_padded
            return labels_padded

    @staticmethod
    def make_kd_labels(self, labels: np.ndarray, padded: bool = True) -> np.ndarray:
        """Makes labels for KDPs"""
        if labels is None:
            if padded:
                labels = self.padded_labels
            else:
                labels = self.labels
        _, num_of_dims = labels.shape
        overall = np.sum(labels, axis=1)
        kda_gt = np.zeros_like(labels)
        for i in np.arange(1, num_of_dims + 1):
            temp = cp.deepcopy(overall)
            temp[temp < i] = 0
            temp[temp >= i] = 1
            kda_gt[:, i - 1] = temp
        return kda_gt

    def cal_anomaly_rate(self, lables: np.ndarray) -> float:
        """percentage of ones to the whole series"""
        n, m = lables.shape
        num_of_ones = np.sum(lables)
        score = num_of_ones / (n * m)
        return score

    @staticmethod
    def remove_base(data: np.ndarray, quantile: float = 0.75):
        """zero out the entries in each series below specified quantile"""
        num_of_dims = data.shape[1]
        # quantiles of each dimension
        bases = np.abs(np.quantile(data, quantile, axis=0))
        data_scaled = data - bases
        for i in np.arange(num_of_dims):
            temp = cp.deepcopy(data_scaled[:, i])
            temp = temp[temp < 0]
            if temp.size > 0:
                threshold_to_zero_out = -1 * np.min(temp)
            else:
                threshold_to_zero_out = 0
            data_scaled[:, i][data_scaled[:, i] < threshold_to_zero_out] = 0
        return data_scaled

    def plot(
        self,
        regimes=None,
        each_tag_thrs=None,
        title: str = "result",
        plot_box: bool = True,
        save_plot: bool = True,
        idx_tag_name: int = 0,
        idx_name: int = 0,
        idx_string: str = None,
        line_color: str = "gray",
        label_type: Optional[str] = "normal",
        show_plot: bool = False,
    ):
        """Ploting the data in a nice way"""

        plot_config = self.config.get_config()["plot"]

        if label_type == "padded":
            labels = self.padded_labels
        elif label_type == "normal":
            labels = self.labels
        else:
            raise ValueError(f"Invalid label type: {label_type}")

        if self.data is None or labels is None:
            self.logger.error("Trying to plot unloaded data or labels!")
            raise ValueError("Data or labels are not loaded!")

        md_plot(
            self.data,
            labels,
            self.logger,
            plot_config,
            regimes,
            each_tag_thrs,
            title,
            plot_box,
            save_plot,
            idx_tag_name,
            idx_name,
            idx_string,
            line_color,
            show_plot,
        )

    def plot_plotly(
        self,
        title: str = "result",
        save_plot: bool = True,
        line_color: str = "gray",
        label_type: Optional[str] = "normal",
        show_plot: bool = True,
    ):
        """Ploting the data in a nice way"""

        plot_config = self.config.get_config()["plot"]
        subplot_size = plot_config["subplot_size"]

        if label_type == "padded":
            labels = self.padded_labels
        elif label_type == "normal":
            labels = self.labels
        else:
            raise ValueError(f"Invalid label type: {label_type}")

        if self.data is None or labels is None:
            self.logger.error("Trying to plot unloaded data or labels!")
            raise ValueError("Data or labels are not loaded!")

        message = (
            f"DATASET: {title} - LABEL TYPE: {label_type} - SHAPE: {self.data.shape}"
        )

        data = self.data.T

        plot_plotly(
            data,
            labels,
            self.logger,
            plot_config,
            title,
            message,
            save_plot,
            show_plot,
            line_color,
            name="TS",
            subplot_size=subplot_size,
        )
