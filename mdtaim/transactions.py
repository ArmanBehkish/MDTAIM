"""
A few helper classes to handle anomalies, transactions, etc.
"""

import os
import pickle
from copy import deepcopy
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional
import logging
import numpy as np
from .config import Config
from .utility import bold, underline, green, blue, red


@dataclass(order=True)
class Anomaly:
    """
    An anomaly is identified with its dimension number
    It can have an importance (utility)
    """

    dimension: int
    utility: Optional[float] = 0

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
        return f"Anomaly(DIM:{self.dimension}, UTIL: {self.utility})"


@dataclass(order=True)
class Transaction:
    """
    A transaction contains List of anomalies
    one transaction is associated with one window
    """

    def __init__(self, win_num: int):
        self.anomalies: list[Anomaly] = []
        self.is_empty: bool = True
        self.win_num: int = win_num

    def add_anomaly(self, dimension: int, utility: float) -> None:
        """
        add anomaly to the transaction
        """
        self.anomalies.append(Anomaly(dimension=dimension, utility=utility))
        self.is_empty = False

    def remove_anomaly(self, dimension: int) -> None:
        """
        remove anomaly with a specific dimension
        """
        self.anomalies.remove(
            next(
                (
                    anomaly
                    for anomaly in self.anomalies
                    if anomaly.dimension == dimension
                ),
                None,
            )
        )
        if len(self.anomalies) == 0:
            self.is_empty = True

    def get_list_of_anomaly_dims(self) -> list[int]:
        """
        get the list of (distinct) anomalies in the transaction
        """
        return list(set(anomaly.dimension for anomaly in self.anomalies))

    def pop_anomaly(self, dimension: Optional[int] = None) -> Anomaly:
        """
        pop anomaly with a specific dimension/with the lowest dimension if no dimension is specified
        """
        if dimension is None:
            self.anomalies.sort(key=lambda x: x.dimension)
            anomaly = self.anomalies.pop(0)
        else:
            index = next(
                (
                    i
                    for i, anomaly in enumerate(self.anomalies)
                    if anomaly.dimension == dimension
                ),
                None,
            )
            if index is not None:
                anomaly = self.anomalies.pop(index)
            else:
                return None

        if len(self.anomalies) == 0:
            self.is_empty = True

        return anomaly

    def get_anomaly(self, dimension: int) -> Anomaly:
        """
        return the Anomaly object with specific dimension!
        """
        return next(
            (anomaly for anomaly in self.anomalies if anomaly.dimension == dimension),
            None,
        )

    def __str__(self) -> str:
        if self.is_empty:
            return f"Transaction {self.win_num}: Empty."
        return f"Transaction {self.win_num}: " + ":::".join(
            str(anol) for anol in self.anomalies
        )

    def __iter__(self):
        return iter(self.anomalies)


class MergeTable:
    """
    Helper data structure to  merge anomalies in consecutive transactions under certain conditions
    TABLE format:
    [dimensions] , dim_1, dim_2, dim_3, dim_4]
    [cons-win_1, (1,25), (0,0), (1,10), (0,0)]
    [cons-win_2, (0,0), (1,10), (1,10), (0,0)]
    [cons-win_3, (0,0), (1,10), (1,10), (0,0)]
    [cons-win_4, (0,0), (0,0), (0,0), (1,10)]
    [sums         , 1,     2,     0,     1]
    where tuple is (exist, utility)
    """

    entry = namedtuple("entry", ["exist", "utility"])
    table: np.ndarray
    list_of_distinct_anomalies: list[Anomaly]
    list_of_transactions: list[Transaction]

    def __init__(
        self,
        logger: logging.Logger,
        config: Config,
        list_of_transactions: list[Transaction],
    ):
        self.logger = logger
        self.config = config
        self.list_of_transactions = list_of_transactions
        self.num_tran = len(list_of_transactions)
        self.list_of_distinct_dims: list[int] = []
        for trans in list_of_transactions:
            for dim in trans.get_list_of_anomaly_dims():
                if dim not in self.list_of_distinct_dims:
                    self.list_of_distinct_dims.append(dim)
        self.num_dims = len(self.list_of_distinct_dims)
        cell = self.entry(exist=0, utility=0)
        self.table = np.zeros((self.num_tran + 2, self.num_dims + 1), dtype=object)

        for i in range(self.num_tran):
            for j in range(self.num_dims):
                self.table[i + 1, j + 1] = cell

        self.construct_table()

    def construct_table(self) -> None:
        """
        fill merge table with anomaly info
        """
        for i, dim in enumerate(self.list_of_distinct_dims):
            # dimsnion numbers
            self.table[0, i + 1] = dim
            for j, t in enumerate(self.list_of_transactions):
                # transaction numbers
                self.table[j + 1, 0] = t.win_num
                if dim in t.get_list_of_anomaly_dims():
                    self.table[j + 1, i + 1] = self.entry(
                        exist=1, utility=t.get_anomaly(dim).utility
                    )

        # number of anomalies in each dimension
        self.table[-1, 1:] = self.sum_exist_entries()

    def get_transaction(self, win_num: int) -> Transaction:
        """
        returns transaction with a specific window number
        """
        return next(
            (tran for tran in self.list_of_transactions if tran.win_num == win_num),
            None,
        )

    def sum_exist_entries(self) -> list[float]:
        """
        sum of columns in merge table
        """
        temp = self.table[1:-1, 1:]
        return [np.sum([x.exist for x in temp[:, i]]) for i in range(temp.shape[1])]

    def remove_anomaly_except_dim(
        self, dim: int, max_util_tran_wins: list[int]
    ) -> None:
        """
        remove anomalies on given dimension except the one with max utility, in case 2 with highest utility are very close, remove others!
        """

        util_thr = self.config.get_config()["itemset_mining_preparation"][
            "keep_adj_util_diff_thr"
        ]

        if len(max_util_tran_wins) == 1:
            self.logger.error(
                f"Supposed to receive two max utility transaction windows, max_util_tran_wins: {max_util_tran_wins}"
            )
        if len(max_util_tran_wins) == 2:
            max_util_trans = [
                self.get_transaction(win_num) for win_num in max_util_tran_wins
            ]
            a1 = max_util_trans[0].get_anomaly(dim)
            a2 = max_util_trans[1].get_anomaly(dim)
            util_diff = -1

            if a1.utility is not None and a2.utility is not None:
                util_diff = np.abs(a1.utility - a2.utility) / np.abs(a1.utility)
            if a2 is None:
                self.logger.warning(
                    f"a2: {a2} is None, utility difference is not calculated!"
                )
                # remove all except a1
                for tran in self.list_of_transactions:
                    if (
                        tran.get_anomaly(dim) is not None
                        and tran.win_num != max_util_tran_wins[0]
                    ):
                        self.logger.debug(
                            f"removing anomaly on dimension: {dim} from transaction: {tran.win_num}"
                        )
                        tran.remove_anomaly(dimension=dim)
                return

            if util_diff <= util_thr and util_diff > 0:
                # keeping 2 highest utilities
                for tran in self.list_of_transactions:
                    if (
                        tran.get_anomaly(dim) is not None
                        and tran.win_num not in max_util_tran_wins
                    ):
                        tran.remove_anomaly(dimension=dim)
                        self.logger.debug(
                            f"removing anomaly on dimension: {dim} from transaction: {tran.win_num}"
                        )
                return

            if util_diff > util_thr:
                for tran in self.list_of_transactions:
                    if (
                        tran.get_anomaly(dim) is not None
                        and tran.win_num != max_util_tran_wins[0]
                    ):
                        self.logger.debug(
                            f"removing anomaly on dimension: {dim} from transaction: {tran.win_num}"
                        )
                        tran.remove_anomaly(dimension=dim)
                return

    def __str__(self) -> str:
        # print table line by line
        for row in self.table:
            return (
                f"\nHeader: {self.table[0]} \n"
                + "\n".join(f"Row {i}: {row}" for i, row in enumerate(self.table[1:-1]))
                + f"\nSummary: {self.table[-1]}"
            )
        return ""


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
            str(trs) for trs in self.transactions
        )

    def __iter__(self):
        return iter(self.transactions)

    def __next__(self):
        return next(self.transactions)

    def __len__(self):
        return len(self.transactions)

    def clear_transactions(self) -> None:
        """
        Delete current transactions
        """
        self.transactions = []
        self.is_empty = True

    def add_transaction(self, transaction: Transaction):
        """
        add one transaction to the list of transactions
        """
        self.transactions.append(transaction)
        self.is_empty = False

    def count_total_anomalies(self) -> int:
        """
        count the total number of anomalies in all transactions!
        """
        return sum(
            [
                1
                for transaction in self.transactions
                if not transaction.is_empty
                for anomaly in transaction.anomalies
            ]
        )

    def sort_transactions(self) -> None:
        """
        sort the transactions based on window number
        """
        self.transactions.sort(key=lambda x: x.win_num, reverse=False)

    def merge_consecutive_anomalies(self) -> None:
        """
        (Idea: A rare event only happens once in a while, not repeatedly in a short time span!)
        due to sliding window/detection technique, one anomaly (on a dimension) might be detected across two or more consecutive windows. This method merge them all into one with higher utility

        ==> does not marge transactions with a gap between

        TASK (fix): in situation where there are more than set number of consecutive anomalies, might not work!
        """

        # find batches of consecutive transactions up to number set below:
        num_cons_to_check = self.config.get_config()["itemset_mining_preparation"][
            "cons_trans_chk_for_merge"
        ]

        cons_trans: list[Transaction] = []
        i = 0
        while i < len(self.transactions) - 1:
            # Main loop to detect batches of consecutive non-empty transactions
            if self.transactions[i].is_empty and self.transactions[i + 1].is_empty:
                i += 1
                continue
            # on the border of a non-empty transactions
            if self.transactions[i].is_empty and not self.transactions[i + 1].is_empty:
                cons_trans.append(self.transactions[i + 1])
                for j in range(i + 2, i + num_cons_to_check + 3):
                    if j == i + num_cons_to_check + 2:
                        self.logger.error(
                            "Not able to merge transactions, too many consecutive transactions!"
                        )
                        raise ValueError
                    if not self.transactions[j].is_empty:
                        cons_trans.append(self.transactions[j])
                        continue
                    # one series of consecutive transactions ended
                    if self.transactions[j].is_empty:
                        if len(cons_trans) > 1:
                            before = len([x for x in cons_trans if not x.is_empty])
                            # merge consecutive anomalies in this batch
                            processed_trans = self.merge_batch(cons_trans)
                            # replace processed batch!
                            self.transactions[i + 1 : j] = processed_trans
                            # move i to the right place after mergin done
                            i += before + 1
                            self.logger.debug(
                                f"will jump to {i} transaction: {i+1} after this number of transactions: {before}"
                            )
                            cons_trans = []
                            break
                        else:
                            # There was only one transaction!
                            cons_trans = []
                            i = j
                            self.logger.debug(f"will jump to {i} after only one")
                            break

        self.logger.info("consecutive anomalies merged successfully!")

    def merge_batch(self, cons_trans: list[Transaction]) -> list[Transaction]:
        """
        Find the anomalies that needs to be merged in a batch of consecutive transactions using MergeTable data structure!

        It uses the diff of the exist column in merge table to find the anomalies in each dimension that need to be merged (there might be less anomalies than transactions on some dimensions)
        """
        # create merge table
        merge = MergeTable(self.logger, self.config, cons_trans)
        self.logger.debug(f"merge table: {merge}")

        # looping over Dimensions
        for d, dim in enumerate(merge.table[0, 1:], start=1):

            # anomaly on every trans in the batch
            if merge.table[-1, d] == merge.num_tran:
                # find two anomalies with highest utilities
                max_util_indices = np.argsort(
                    [x.utility for x in merge.table[1:-1, d]],
                    axis=0,
                )[::-1][:2]
                max_util_tran_wins = [merge.table[i + 1, 0] for i in max_util_indices]
                self.logger.debug(
                    f"Detected max utility transaction windows: {max_util_tran_wins} on dimension: {dim}"
                )
                merge.remove_anomaly_except_dim(dim, max_util_tran_wins)

            if merge.table[-1, d] == 1 or merge.table[-1, d] == 0:
                # no anomaly to merge
                continue

            # number of anomalies are less than transactions
            if merge.table[-1, d] < merge.num_tran and merge.table[-1, d] > 1:

                # case 1: switches from 0 to 1
                vector = [x.exist for x in merge.table[1:-1, d]]
                diff = np.diff(vector, axis=0)
                one_idxs = np.where(diff == 1)[0]
                m_one_idxs = np.where(diff == -1)[0]
                if len(one_idxs) > 1 or len(m_one_idxs) > 1:
                    # Task: check all possibilities
                    self.logger.warning(
                        f"more than one 1 or -1 in the vector: {vector} for dimension: {dim}, supposing there is no need for merging!"
                    )
                    continue
                else:
                    one_idx: int = int(one_idxs[0] if one_idxs.size != 0 else -1)
                    m_one_idx: int = int(m_one_idxs[0] if m_one_idxs.size != 0 else -1)

                # Case of 1 in diff:
                # one can not be last
                # all diff values to the end after one_idx are 1
                if (
                    one_idx != -1
                    and m_one_idx == -1
                    and one_idx != len(diff) - 1
                    and np.all(np.array(vector[one_idx + 1 :]) == 1)
                ):
                    # keep 2 with highest utilities
                    max_util_indices = np.argsort(
                        [x.utility for x in merge.table[one_idx + 1 : -1, d]],
                        axis=0,
                    )[::-1][:2]
                    max_util_tran_wins = [
                        merge.table[i + 1, 0] for i in max_util_indices
                    ]
                    # remove anomalies from dimension
                    merge.remove_anomaly_except_dim(dim, max_util_tran_wins)

                # cases of -1 in diff:
                # -1 can not be first
                # all diff values until m_one_idx are 1
                if (
                    m_one_idx != -1
                    and one_idx == -1
                    and m_one_idx != 0
                    and np.all(np.array(vector[:m_one_idx]) == 1)
                ):

                    max_util_indices = np.argsort(
                        [x.utility for x in merge.table[1 : m_one_idx + 1, d]],
                        axis=0,
                    )[::-1][:2]
                    max_util_tran_wins = [
                        merge.table[i + 1, 0] for i in max_util_indices
                    ]
                    merge.remove_anomaly_except_dim(dim, max_util_tran_wins)

                # case of both 1 and -1 in diff
                if (
                    one_idx != -1
                    and m_one_idx != -1
                    and one_idx != len(diff) - 1
                    and m_one_idx != 0
                    and np.all(np.array(vector[one_idx + 1 : m_one_idx + 1]) == 1)
                ):

                    max_util_indices = np.argsort(
                        [x.utility for x in merge.table[one_idx + 1 : m_one_idx, d]],
                        axis=0,
                    )[::-1][:2]
                    max_util_tran_wins = [
                        merge.table[i + 1, 0] for i in max_util_indices
                    ]
                    merge.remove_anomaly_except_dim(dim, max_util_tran_wins)

        return merge.list_of_transactions

    def print_transactions(self, to: str) -> None:
        """
        print the transactions
        """
        if to == "logs":
            for trs in self.transactions:
                self.logger.debug(str(trs))
        elif to == "console":
            for trs in self.transactions:
                self.logger.info(str(trs))

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
