import logging
import os
from .config import Config
from .utility import bold, underline, green, blue, red
from .processdata import PreprocessData


class PostProcess:
    """
    Postprocess the output of the SPMF algorithms to produce the output of the MDTAIM pipeline
    """

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self.itemsets: dict[set[int], list[int]] = {}

    def produce_output(self):
        """
        if the algorithm does not return TIDs, find them
        sort the KDAs based on the utility and TIDs
        save the output file
        """
        itemset_config = self.config.get_config()["itemset_mining_preparation"]
        spmf_config = self.config.get_config()["spmf"]
        data_config = self.config.get_config()["data"]

        high_util_enabled = spmf_config["high_utility_itemsets"]
        tid_enabled = spmf_config["show_transaction_ids"]
        win_size = itemset_config["window_size"]
        empty_trans_replacement = spmf_config["empty_trans_replacement"]
        zero_replace = (
            spmf_config["replace_zero"]["replace_zero_with"]
            if spmf_config["replace_zero"]["enable"]
            else None
        )

        spmf_output_f = (
            data_config["spmf_output_path"]
            + next(os.scandir(data_config["spmf_output_path"])).name
        )

        tran_db_f = (
            data_config["transaction_db_path"]
            + next(os.scandir(data_config["transaction_db_path"])).name
        )

        final_output_f = (
            data_config["final_output_path"]
            + data_config["dataset_title"]
            + "_"
            + "FINAL"
            + ".csv"
        )

        # if algorithm is not maximal/closed, scan the database to reduce output to maximal/closed patterns

        # for now only using maximal/closed algoeithms

        if tid_enabled:
            # for algorithms that return TIDs (do we have a high utility itemset with TIDS?)
            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    supp = int(line.split("#SUP:")[1].split("#TID:")[0].strip())
                    tids = [
                        int(tid) + 1 for tid in line.split("#TID:")[1].split(" ")[1:]
                    ]
                    dims = line.split("#SUP:")[0].split(" ")[:-1]
                    self.itemsets[frozenset(dims)] = [supp, 0, tids]

        if not tid_enabled and not high_util_enabled:
            # find TIDs for each frequent itemset
            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    supp = int(line.split("#SUP:")[1].strip())
                    dims = line.split("#SUP:")[0].split(" ")[:-1]
                    self.itemsets[frozenset(dims)] = [supp, 0, []]

                for count, line in enumerate(t_f, start=1):
                    if line.strip() == str(empty_trans_replacement):
                        continue
                    else:
                        tid = count
                        dimensions = line.strip().split(" ")
                        if (
                            # KDA with no TIDs
                            frozenset(dimensions) in self.itemsets
                            and not self.itemsets[frozenset(dimensions)][2]
                        ):
                            # first occurrence of itemset, add tid
                            self.itemsets[frozenset(dimensions)][2].append(tid)

                # remove KDAa which did not appear in a specific place in the
                self.itemsets = {k: v for k, v in self.itemsets.items() if v[2]}
                # sort base on TIDs
                self.itemsets = dict(
                    sorted(self.itemsets.items(), key=lambda x: x[1][2][0])
                )

        if not tid_enabled and high_util_enabled:
            # find TIDS for each high utility itemset
            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    supp = line.split("#SUP:")[1].strip().split("#UTIL:")[0].strip()
                    util = line.split("#UTIL:")[1].strip()
                    dims = line.split("#SUP:")[0].split(" ")[:-2]
                    self.itemsets[frozenset(dims)] = [supp, util, []]

                for count, line in enumerate(t_f, start=1):
                    if line.split(":")[0].strip() == str(empty_trans_replacement):
                        continue
                    else:
                        tid = count
                    dimensions = line.split(":")[0].strip().split(" ")
                    if (
                        # KDA with no TIDs
                        frozenset(dimensions) in self.itemsets
                        and not self.itemsets[frozenset(dimensions)][2]
                    ):
                        # first occurrence of itemset, add tid
                        self.itemsets[frozenset(dimensions)][2].append(tid)

            # remove KDAa which did not appear in a specific place
            self.itemsets = {k: v for k, v in self.itemsets.items() if v[2]}
            # sort base on Utilities, then TIDs
            self.itemsets = dict(
                sorted(
                    self.itemsets.items(),
                    key=lambda x: (x[1][1], x[1][2][0]),
                )
            )

        # save output file
        with open(final_output_f, "w", encoding="utf-8") as f:
            # write header
            if not high_util_enabled:
                f.write("KDA:Location\n")
                for itemset, data in self.itemsets.items():
                    f.write(
                        ",".join(str(item) for item in itemset)
                        + ":"
                        + ",".join(str(int(tid) * win_size) for tid in data[2])
                        + "\n"
                    )
            else:
                f.write("KDA:Location:Importance\n")
                for itemset, data in self.itemsets.items():
                    f.write(
                        ",".join(str(item) for item in itemset)
                        + ":"
                        + ",".join(str(int(tid) * win_size) for tid in data[2])
                        + ":"
                        + str(data[1])
                        + "\n"
                    )

    # def plot(
    #     self,
    #     regimes=None,
    #     each_tag_thrs=None,
    #     title: str = "KDP_Profile",
    #     plot_box: bool = True,
    #     save_plot: bool = True,
    #     idx_tag_name: int = 0,
    #     idx_name: int = 0,
    #     idx_string: str = None,
    #     line_color: str = "black",
    #     label_type: Optional[str] = "padded",
    #     labels: np.ndarray = None,
    #     show_plot: bool = False,
    # ):
    #     """Plot the output of the MDTAIM pipeline"""

    #     plot_config = self.config.get_config()["plot"]

    #     if label_type == "padded":
    #         title = title + "_Padded_Labels"

    #     if self.itemsets is None:
    #         self.logger.error("KDAs  not Provided or Loaded!")
    #         raise ValueError("Data or labels are not loaded!")

    #     kdas = list(self.itemsets.keys())
    #     kd_labels = PreprocessData.make_kd_labels(self, labels)
    #     name = "KDP"

    #     md_plot(
    #         kdas,
    #         kd_labels,
    #         self.logger,
    #         plot_config,
    #         regimes,
    #         each_tag_thrs,
    #         title,
    #         plot_box,
    #         save_plot,
    #         idx_tag_name,
    #         idx_name,
    #         idx_string,
    #         line_color=line_color,
    #         show_plot=show_plot,
    #         name=name,
    #     )
