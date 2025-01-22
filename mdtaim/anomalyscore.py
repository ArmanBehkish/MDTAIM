from abc import ABC, abstractmethod
from typing import Optional
import copy as cp
import os
import logging
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
import matrixprofile as mpx
from tqdm import tqdm
from .config import Config
from .utility import md_plot, plot_plotly


class AnomalyScoring(ABC):
    """
    Abstract base class for anomaly scoring algorithms.
    """

    @abstractmethod
    def __init__(self, logger: logging.Logger, config: Config):
        """
        Initialize the anomaly scoring algorithm with configuration parameters.
        """
        self.config = config
        self.logger = logger

    @abstractmethod
    def calculate_score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate the anomaly score for the given data.

        Args:
            data (np.ndarray): Input data for anomaly detection.

        Returns:
            np.ndarray: Anomaly scores for the input data.
        """
        pass

    @abstractmethod
    def save_to_pickle(self, score: np.ndarray) -> None:
        """
        Save the anomaly score to the pickle file
        """
        pass

    @abstractmethod
    def plot(
        self,
        regimes,
        each_tag_thrs,
        title: str,
        plot_box: bool,
        save_plot: bool,
        idx_tag_name: int,
        idx_name: int,
        idx_string: str,
        line_color: str,
        label_type: Optional[str],
        labels: np.ndarray,
    ) -> None:
        """
        Plot the anomaly score.

        Args:
            score (np.ndarray): Anomaly scores for the input data.
            title (str): Title of the plot.
        """
        pass


class MatrixProfile(AnomalyScoring):
    """
    Calculate the matrix profile of the data
    """

    def __init__(self, logger: logging.Logger, config: Config) -> None:
        super().__init__(logger, config)
        self.mps: np.ndarray = np.empty((0, 0), dtype=np.float64)
        self.file_path: str = ""
        self.mp_seq_len: int = 0

    def calculate_score(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates Matrix Profile of data
        Shape of the output is the same as the input data
        The calculated Matrix Profile is saved in a pickle file!
        The returend Matrix Profile is Quantile Corrected is option is set.
        """
        data_config = self.config.get_config()["data"]
        mp_config = self.config.get_config()["anomalyscoring"]["matrixprofile"]
        itemset_config = self.config.get_config()["itemset_mining_preparation"]
        output_path = data_config["scores_path"]
        ds_title = data_config["dataset_title"]
        self.mp_seq_len = mp_config["subsequence_length"]
        self.file_path = output_path + f"{ds_title}_mps_{self.mp_seq_len}.pkl"

        # if MP already calculated for this dataset and window size
        if os.path.exists(self.file_path):
            self.logger.warning(
                "Matrix profile already calculated for this dataset! Ignore MP calculation time."
            )
            with open(self.file_path, "rb") as f:
                self.mps = pickle.load(f)
                f.close()
            if itemset_config["cut_baseline"]:
                # cut baseline on loaded MP
                q = itemset_config["quantile"]
                self.mps = self.remove_baseline(self.mps, q)
                self.logger.info(f"Baseline correction with quantile: {q} done!")
            return self.mps

        # mp_config = self.config.get_config()["anomalyscoring"]["matrixprofile"]
        mps = []
        # win_size = mp_config["subsequence_length"]

        if np.ndim(data) == 1:
            if mp_config["auto_subsequence_length"]:
                # calculate windows size
                self.mp_seq_len = self.get_win(data)
                self.logger.info(
                    f"estimated window size by median of distance between peaks: {self.mp_seq_len}"
                )
            self.logger.info(
                "calculating MPs with given window size: {self.mp_seq_len}"
            )
            # calculate matrix profile
            profile = mpx.compute(data, self.mp_seq_len)["mp"]
            return np.asarray([profile]).T

        for i in tqdm(np.arange(data.shape[1]), desc="MP calculation"):
            current_ts = data[:, i]
            if mp_config["auto_subsequence_length"]:
                self.mp_seq_len = self.get_win(current_ts)
                self.logger.info(
                    f"estimated window size for dimension {i} is: {self.mp_seq_len}"
                )
            self.logger.debug(
                f"calculating MP for dimension {i} with given window size: {self.mp_seq_len}"
            )
            # calculate matrix profile
            profile = mpx.compute(current_ts, self.mp_seq_len)["mp"]
            pad_size = len(current_ts) - len(profile)
            # pad calculated MP with its min val to the same length of TS
            profile = np.insert(profile, len(profile), [np.min(profile)] * pad_size)
            mps.append(profile)

        self.mps = np.asarray(mps).T
        self.logger.debug(
            f"matrix profile shape is: {self.mps.shape} and its header is: \n{self.mps[:5]}"
        )

        # save MP to file
        self.save_to_pickle(self.mps)

        # Cutting quntile(q) of MP if relevant option is set
        if itemset_config["cut_baseline"]:
            q = itemset_config["quantile"]
            self.mps = self.remove_baseline(self.mps, q)
            self.logger.info(f"Baseline correction with quantile: {q} done!")

        return self.mps

    def get_win(self, ts: np.ndarray) -> int:
        """
        Estimates windows size, by median of distance between peaks
        """
        # scale the TS to [0,2*mean(ts)]
        x = np.squeeze(
            MinMaxScaler((0, np.abs(2 * np.average(ts)))).fit_transform(
                ts.reshape(-1, 1)
            )
        )
        # find local maxima of scaled signal
        peaks = find_peaks(x, prominence=np.abs(np.average(ts) / 2))[0]
        # return median of distance between peaks
        return np.median(np.diff(peaks, axis=-1))

    @staticmethod
    def remove_baseline(data: np.ndarray, quantile: float) -> np.ndarray:
        """zero out the entries in each series below specified quantile"""
        num_of_dims = data.shape[1]
        bases = np.abs(np.quantile(data, quantile, axis=0))
        data_scaled = data - bases
        for i in np.arange(num_of_dims):
            col_i = cp.deepcopy(data_scaled[:, i])
            col_i_negs = col_i[col_i < 0]
            if col_i_negs.size > 0:
                thr_to_zero_out = -1 * np.min(col_i_negs)
            else:
                thr_to_zero_out = 0
            data_scaled[:, i][data_scaled[:, i] < thr_to_zero_out] = 0
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
        labels: np.ndarray = None,
        show_plot: bool = False,
    ):
        """Ploting the data in a nice way"""

        plot_config = self.config.get_config()["plot"]

        if label_type == "padded":
            title = title + "_Padded_Labels"

        if self.mps is None or labels is None:
            self.logger.error("Matrix profile and Labels not Provided or Loaded!")
            raise ValueError("Data or labels are not loaded!")

        name = "MP"

        md_plot(
            self.mps,
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
            name,
        )

    def plot_plotly(
        self,
        title: str = "result",
        save_plot: bool = True,
        line_color: str = "gray",
        label_type: Optional[str] = "normal",
        show_plot: bool = True,
        labels: np.ndarray = np.empty((0, 0), dtype=np.float64),
    ):
        """Ploting the data in a nice way"""

        plot_config = self.config.get_config()["plot"]
        subplot_size = plot_config["subplot_size"]

        if self.mps is None or labels is None:
            self.logger.error("Matrix profile and Labels not Provided or Loaded!")
            raise ValueError("Data or labels are not loaded!")

        message = f"{title} - MP_SUB_LEN: {self.mp_seq_len} LABEL: {label_type} - SHAPE: {self.mps.shape}"

        mps = self.mps.T

        plot_plotly(
            mps,
            labels,
            self.logger,
            plot_config,
            title,
            message,
            save_plot,
            show_plot,
            line_color,
            name="MP",
            subplot_size=subplot_size,
        )

    def save_to_pickle(self, score: np.ndarray) -> None:
        """
        Save matrix profile scores to the pickle file
        It should always save the pure scores, not baseline corrected
        """
        # if not os.path.exists(output_path):
        #     self.logger.error(f"output path {output_path} does not exist!")
        #     raise FileNotFoundError

        if os.path.exists(self.file_path):
            self.logger.warning("Matrix profile already calculated for this dataset!")
            return

        # # remove old files in the output path
        # for file in os.listdir(output_path):
        #     os.remove(os.path.join(output_path, file))

        with open(self.file_path, "wb") as f:
            pickle.dump(score, f)
            f.close()

        self.logger.info(f"matrix profile saved to file {self.file_path} successfully!")

    def load_from_pickle(self) -> np.ndarray:
        """
        NOT USED IN THE CURRENT VERSION
        Load matrix profile from the pickle file
        """
        data_config = self.config.get_config()["data"]
        output_path = data_config["scores_path"]
        file = next(os.scandir(output_path)).name
        file_path = os.path.join(output_path, file)
        with open(file_path, "rb") as f:
            self.mps = pickle.load(f)
            f.close()
        return self.mps
