"""KDP class: n of k anomalies implementation based on TSAD paper!"""

import logging
from .config import Config
from .processdata import PreprocessData
from .utility import md_plot
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


class KDP:
    """all K Dimensional Profiles as in TSADIS paper"""

    def __init__(self, logger: logging.Logger, config: Config) -> None:
        self.logger = logger
        self.config = config
        self.mps = np.empty((0, 0), dtype=np.float64)
        self.kdps = np.empty((0, 0), dtype=np.float64)
        self.kdps_idx = np.empty((0, 0), dtype=np.float64)

    def fast_find_anomalies(self, mps) -> Tuple[np.ndarray, np.ndarray]:
        """get all-kdp-profils (def. 14) by sorting the MPs at each timepoint i (descending), the value at place j is jKP."""
        kdps = np.sort(mps, axis=1)
        self.logger.debug(f"kdps shape: {kdps.shape}")
        self.logger.debug(f"KDPs: {kdps[100:200]}")

        kdps_idx = np.argsort(mps, axis=1)
        self.logger.debug(f"kdps_idx shape: {kdps_idx.shape}")
        self.logger.debug(f"KDPs_idx: {kdps_idx[100:200]}")
        # flip the rows odered just for visualization, otherwise does not impact the result
        self.kdps = np.flip(kdps, axis=1)
        self.kdps_idx = np.flip(kdps_idx, axis=1)
        return self.kdps, self.kdps_idx

    def plot(
        self,
        regimes=None,
        each_tag_thrs=None,
        title: str = "KDP_Profile",
        plot_box: bool = True,
        save_plot: bool = True,
        idx_tag_name: int = 0,
        idx_name: int = 0,
        idx_string: str = None,
        line_color: str = "black",
        label_type: Optional[str] = "padded",
        labels: np.ndarray = None,
        show_plot: bool = False,
    ):
        """Plot and save the KDPs"""

        plot_config = self.config.get_config()["plot"]

        if label_type == "padded":
            title = title + "_Padded_Labels"

        if self.kdps is None or labels is None:
            self.logger.error("KDP profile and Labels not Provided or Loaded!")
            raise ValueError("Data or labels are not loaded!")

        kd_labels = PreprocessData.make_kd_labels(self, labels)
        name = "KDP"

        md_plot(
            self.kdps,
            kd_labels,
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
            line_color=line_color,
            show_plot=show_plot,
            name=name,
        )
