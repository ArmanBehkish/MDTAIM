"""
A few helper classes to handle anomalies, transactions, etc.
"""

import os
import pickle
from dataclasses import dataclass
import logging
import numpy as np
from .config import Config
from .utility import bold, blue, red


@dataclass()
class Anomaly:
    """
    An anomaly is identified with its dimension & Transaction Number
    It can have a significance (utility)
    """

    dimension: int
    transaction: int
    nz_len: int
    utility: float
    # window: Optional[np.ndarray] = None
    # win_max: Optional[float] = None
    # win_sum: Optional[float] = None

    @staticmethod
    def calculate_utility(window: np.ndarray, func: str) -> float:
        """calculate the utility of an anomalous window"""
        if func == "max":
            return np.max(window, axis=0)
        elif func == "sum":
            return np.sum(window, axis=0)
        else:
            raise ValueError(f"Unknown function: {func}")

    def __str__(self) -> str:
        return f"Anomaly(DIM:{self.dimension}, TRANS: {self.transaction}, SIGNIFICANCE: {self.utility:.1f}, NZ_LEN: {self.nz_len})"


@dataclass(order=True)
class Transaction:
    """
    A transaction contains List of anomalies
    one transaction is associated with one window
    """

    def __init__(self, win_num: int):
        self.anomalies: list[Anomaly] = []
        # self.is_empty: bool = True
        self.win_num: int = win_num

    def add_anomaly(self, anomaly: Anomaly) -> None:
        """
        add anomaly to the transaction
        """
        self.anomalies.append(anomaly)

    def __str__(self) -> str:
        # if self.is_empty:
        #     return f"Transaction {self.win_num}: Empty."
        return f"Transaction {self.win_num}: " + ":::".join(
            f"\n {anol}" for anol in self.anomalies
        )

    def __iter__(self):
        return iter(self.anomalies)


class Transactions:
    """
    List of transactions for one dataset
    """

    def __init__(
        self,
        config: Config,
        logger: logging.Logger,
    ) -> None:
        self.transactions: list[Transaction] = []
        self.is_empty: bool = True
        self.logger = logger
        self.config = config

    def __str__(self) -> str:
        if self.is_empty:
            self.logger.error("No transactions found!")
            return ""
        return "Dataset Transactions are: \n" + "\n".join(
            f"\n {trs}" for trs in self.transactions
        )

    def __iter__(self):
        return iter(self.transactions)

    def __next__(self):
        return next(self.transactions)

    def __len__(self):
        return len(self.transactions)

    def add_transaction(self, transaction: Transaction):
        """
        add one transaction to the list of transactions
        """
        self.transactions.append(transaction)
        self.is_empty = False

    def sort_transactions(self) -> None:
        """
        sort the transactions based on window number
        """
        self.transactions.sort(key=lambda x: x.win_num, reverse=False)

    def save_transactions_to_file(self) -> None:
        """
        save the current transactions to pickle file
        picks the path to save from config_file: data/transactions_path
        file name: transactions.pkl
        """
        data_config = self.config.get_config()["data"]
        output_path = data_config["transactions_path"]

        if not os.path.exists(output_path):
            self.logger.error(f"output path {output_path} does not exist!")
        # remove old files in the output path
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))
        # interim file name is fixed!
        file_path = output_path + "transactions.pkl"

        try:
            with open(file_path, "wb") as f:
                pickle.dump(self.transactions, f)
                f.close()
        except FileNotFoundError as e:
            self.logger.critical(f"Error in saving transactions to pickle: {e}")
            raise e
        else:
            self.logger.info("current transactions saved to pickle successfully!")
        return

    def load_transactions_from_file(self) -> None:
        """
        load the transactions from pickle file
        picks the path to load from config_file: data/transactions_path
        file name: transactions.pkl
        """
        data_config = self.config.get_config()["data"]
        input_path = data_config["transactions_path"]

        if not os.path.exists(input_path):
            self.logger.error(f"input path {input_path} does not exist!")
            raise FileNotFoundError
        if sum([1 for file in os.listdir(input_path)]) > 1:
            self.logger.error(f"more than one file in the input path {input_path}")
            self.logger.info(
                f"make sure there is only one file in the input path: {input_path}"
            )
            raise FileNotFoundError

        # get the file name in the input path
        file_name = os.listdir(input_path)[0]
        # check if the flename is transactions.pkl
        if file_name != "transactions.pkl":
            self.logger.error(f"file name is not transactions.pkl: {file_name}")
            self.logger.info("make sure the file name is transactions.pkl")
            raise FileNotFoundError

        file_path = input_path + file_name

        try:
            with open(file_path, "rb") as f:
                self.transactions = pickle.load(f)
                f.close()
        except FileNotFoundError as e:
            self.logger.error(f"Error finding file: {e}")
            raise e
        self.logger.info("transaction DB loaded successfully!")

        return
