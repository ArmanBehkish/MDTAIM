"""
NOTICE: Main implementation logic has been removed from this file.
"""

from typing import Optional
from copy import deepcopy
from typing import List, Dict
from collections import deque
import logging
import timeit
import os
import pickle
import subprocess
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
from bidict import bidict
from .config import Config
from .utility import plot_plotly
from .transactions import Transactions, Transaction, Anomaly
from .anomalyscore import MatrixProfile
from .processdata import PreprocessData
from .utility import bold, underline, green, blue, red


class ItemSetPreparation:
    """
    convert anomaly scores to transactions for itemset mining
    Methods using different methods for conversion
    """

    def __init__(self, logger: logging.Logger, config: Config):
        self.config = config
        self.logger = logger
        self.anomaly_scores: Optional[np.ndarray] = None
        self.transactions: Transactions = Transactions(config, logger)
        self.all_anomalies: list[Anomaly] = []
        self.temp_thr: Optional[float] = None
        self.num_max: int = 1
        self.test_idx: int = 0

    def set_anomaly_scores(self, anomaly_scores: np.ndarray) -> None:
        """
        passing the scores
        """
        self.anomaly_scores = anomaly_scores

    def load_anomaly_scores(self) -> None:
        """
        load the anomaly scores (matrix profile) from the pickle file
        """
        pass

    def convert_anomalies_to_transactions(self, padded_labels: np.ndarray) -> None:
        """
        convert the anomalies to transactions using the specified method
        for each selected method, a separate conversion function is provided
        """
        pass

    def get_cons_nonzero_len(self, window: np.ndarray) -> int:
        """
        Calculates the longest non-zero sequence in a window
        Returns:
        """
        return 0

    def convert_beta_1(self) -> None:
        """
        Converting MP scores to transactions, ready for itemset mining.
        This function contains both the logic to detect windows containing a discord, and to remove redundant detections.
        """
        pass

    def convert_mean_sigma(self) -> None:
        """
        Converting MP scores to transactions, ready for itemset mining.
        This function contains both the logic to detect windows containing a discord, and to remove redundant detections.
        """

        pass


    def build_transactions(self) -> None:
        """
        Build and save transactions for further processing
        """
        pass

    def cal_anomaly_detec_accuracy(
        self, padded_lbls: np.ndarray, param_search_mode: bool
    ) -> tuple[float, float, float]:
        """
        calculate the accuracy of generated transactions by comparing with passed labels
        count the total number of detected anomalies in transasctions
        matches: number of anomalies that have an interval in common with a label
        is param_search_mode is True, the indexes are shifted by val_idx
        RETURN: f1 score, recall, precision
        """
        return 0.0, 0.0, 0.0

    def plot_detected_anomalies_vs_labels(
        self, padded_lbls: np.ndarray, show_plot: bool
    ) -> None:
        """plot detected anomalies vs labels to better understand detection accuracy"""
        pass


class SPMF:
    """
    class to run the SPMF algorithms using the command line version of SPMF Library
    """

    def __init__(self, logger: logging.Logger, config: Config) -> None:
        self.config = config
        self.logger = logger
        self.transactions: Transactions = Transactions(config, logger)
        self.REQUIRED_JAVA_VERSION: int = 8
        self.tid_map: bidict = bidict()

    def load_transactions(self) -> None:
        """load transactions"""
        self.transactions.load_transactions_from_file()

    def check_java_version(self) -> None:
        """check proper java installation and version to run the jar file
        TASK: find minimum version for SPMF and set the constant"""
        pass

    def run_algorithm(self) -> None:
        """
        run the selected SPMF algorithm
        output file: ALGONAME_out.txt

        IMPLEMETDD ALGORITHMS:

        - Frequent Itemsets:
        NO TID OUTPUT: Apriori, FPGrowth_itemsets,dEclat,HMine,FIN,DFIN,NegFIN,PrePost+,PrePost,LCMFreq
        TID OUTPUT: Eclat, AprioriTID_Bitset

        - Frequent closed Itemsets:
        NO TID OUTPUT: AprioriClose, LCM, FPClose, NAFCP, NEclatClosed
        TID OUTPUT: DCI_Closed, Charm_bitset

        - Frequent Maximal Itemsets:
        NO TID OUTPUT: FPMax
        TID OUTPUT: Charm_MFI

        - High Utility Itemsets:
        NO TID OUTPUT: Two-Phase, FHM, EFIM, CHUI-MinerMax
        TID OUTPUT:

        - Perfectly Rare Itemsets:
        NO TID OUTPUT: AprioriInverse
        TID OUTPUT: AprioriInverse_TID

        - Generator Itemsets:
        NO TID OUTPUT: DefMe, Pascal, Zart
        TID OUTPUT:

        - Minimal Rare Itemsets:
        NO TID OUTPUT: AprioriRare
        """
        pass


    def prepare_transaction_database(self) -> None:
        """
        Prepare context database
        """
        pass

    def print_output(self) -> None:
        """
        NOT USED IN THE CURRENT VERSION
        Read the SPMF output file and formats the output to be human readable
        """
        pass
