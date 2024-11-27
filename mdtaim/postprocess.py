from typing import Dict, List, Optional
import logging
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from .config import Config
from .utility import bold, underline, green, blue, red
from .processdata import PreprocessData


class PostProcess:
    """
    processes the output of the SPMF algorithms to produce the output of the MDTAIM pipeline
    """

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        # Itemsets : {frozenset(dimensions) : [support, utility, [tids]]}
        self.itemsets: Dict[frozenset[int], List[int]] = {}

    def produce_output(self):
        """
        if the algorithm does not return TIDs, finds them!
        sorts the KDAs based on the utility and TIDs
        saves the output file
        """
        itemset_config = self.config.get_config()["itemset_mining_preparation"]
        spmf_config = self.config.get_config()["spmf"]
        data_config = self.config.get_config()["data"]

        high_util_enabled = spmf_config["high_utility_itemsets"]
        tid_enabled = spmf_config["show_transaction_ids"]
        win_size = itemset_config["window_size"]
        empty_trans_replacement = spmf_config["empty_trans_replacement"]
        zero_replacement = (
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

        if tid_enabled and not high_util_enabled:
            # for algorithms that return TIDs w/o utilities
            # i.e., AprioriTID_Bitset and alike
            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    # extract SPMF output
                    supp = int(line.split("#SUP:")[1].split("#TID:")[0].strip())
                    tids = [
                        int(tid) + 1 for tid in line.split("#TID:")[1].split(" ")[1:]
                    ]
                    dims = [x for x in line.split("#SUP:")[0].split(" ") if x]
                    self.itemsets[frozenset(dims)] = [supp, 0, tids]

                # keep only KDAa which appeared in the transaction database
                kdas_in_db = []
                for count, line in enumerate(t_f, start=1):
                    if line.strip() == str(empty_trans_replacement):
                        continue
                    else:
                        kdas_in_db.append(frozenset(line.strip().split(" ")))

                self.itemsets = {
                    k: v for k, v in self.itemsets.items() if k in kdas_in_db
                }

                # sort base on TIDs
                # sort base on Utilities, then TIDs
                self.itemsets = dict(
                    sorted(
                        self.itemsets.items(),
                        key=lambda x: (x[1][1], x[1][2][0]),
                    )
                )

            self.logger.debug(
                f"finalized itemsets after TIDs/NO UTILs: {self.itemsets}!"
            )

        if not tid_enabled and not high_util_enabled:
            # i.e., Apriori. and alike

            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    # extract SPMF output
                    supp = int(line.split("#SUP:")[1].strip())
                    dims = [x for x in line.split("#SUP:")[0].split(" ") if x]
                    self.itemsets[frozenset(dims)] = [supp, 0, []]

                # find TIDs for each frequent itemset
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

                # remove KDAa which did not appear in a specific place in the transactions
                self.itemsets = {k: v for k, v in self.itemsets.items() if v[2]}
                # sort base on TIDs
                self.itemsets = dict(
                    sorted(self.itemsets.items(), key=lambda x: x[1][2][0])
                )

            self.logger.debug(
                f"finalized itemsets after NO TIDS/NO UTILs: {self.itemsets}!"
            )

        if not tid_enabled and high_util_enabled:
            # i.e., AprioriTID_Bitset and alike
            # find TIDS for each high utility itemset

            with open(spmf_output_f, "r") as s_f, open(tran_db_f, "r") as t_f:
                for line in s_f:
                    # if the algorithm has support in output (e.g., CHUI-MinerMax):
                    if "#SUP:" in line:
                        supp = line.split("#SUP:")[1].strip().split("#UTIL:")[0].strip()
                        util = line.split("#UTIL:")[1].strip()
                        dims = [x for x in line.split("#SUP:")[0].split(" ") if x]
                        self.itemsets[frozenset(dims)] = [supp, util, []]

                    # if algorithm does not have support in output (e.g., EFIM):
                    if "#SUP:" not in line:
                        supp = None
                        util = line.split("#UTIL:")[1].strip()
                        dims = [x for x in line.split("#UTIL:")[0].split(" ") if x]
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

            self.logger.debug(
                f"finalized itemsets after NO TIDS/with UTILs: {self.itemsets}!"
            )

        # replace zero replacement with 0
        for keys, value in list(self.itemsets.items()):
            new_keys = frozenset(
                key if int(key) != int(zero_replacement) else 0 for key in keys
            )
            del self.itemsets[keys]
            self.itemsets[new_keys] = value

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

    def plot_heatmap(
        self,
        title: str = "MDTAIM",
        save_plot: bool = True,
        line_color: str = "gray",
        label_type: Optional[str] = "normal",
        show_plot: bool = True,
        labels: np.ndarray = np.empty((0, 0), dtype=np.float64),
    ):
        itemset_config = self.config.get_config()["itemset_mining_preparation"]
        spmf_config = self.config.get_config()["spmf"]
        high_util_enabled = spmf_config["high_utility_itemsets"]
        dataset_title = self.config.get_config()["data"]["dataset_title"]
        plot_config = self.config.get_config()["plot"]
        win_size = itemset_config["window_size"]
        title = (
            dataset_title
            + " - "
            + "Multidimensional Anomalies along with their Locations and relative importance."
        )
        box_size = 50

        # Create initial DataFrame from itemsets: : {frozenset(dimensions) : [support, utility, [tids]]}
        df = pd.DataFrame(
            {
                "KDA": [
                    ",".join(sorted(str(x) for x in dims))
                    for dims in self.itemsets.keys()
                ],
                "Location": [
                    tids[0] * win_size for _, _, tids in self.itemsets.values()
                ],
                "Importance": [
                    float(utility) for _, utility, _ in self.itemsets.values()
                ],
            }
        )

        # Add dimension count and sort DataFrame
        df["dim_count"] = df["KDA"].str.count(",") + 1
        df = df.sort_values(["dim_count"], ascending=[False])

        self.logger.debug(f"data frame created for heatmap: {df}\n")

        # Create the heatmap using a scatter plot with square markers
        fig = go.Figure()

        if not high_util_enabled:
            fig.add_trace(
                go.Scatter(
                    x=df["Location"],
                    y=df["KDA"],
                    mode="markers",
                    marker=dict(
                        size=box_size,
                        color="black",
                        symbol="triangle-down",
                    ),
                    showlegend=False,
                )
            )

        if high_util_enabled:
            # create text colors adaptive based on marker color
            min_imp = df["Importance"].min()
            max_imp = df["Importance"].max()
            df["norm_imp"] = (df["Importance"] - min_imp) / (max_imp - min_imp)
            text_colors = ["black" if x > 0.5 else "white" for x in df["norm_imp"]]

            # Add scatter plot with text labels
            fig.add_trace(
                go.Scatter(
                    x=df["Location"],
                    y=df["KDA"],
                    mode="markers+text",
                    text=df["Importance"].round(1),
                    textposition="middle center",
                    textfont=dict(
                        color=text_colors,
                        size=12,
                    ),
                    marker=dict(
                        size=box_size,
                        color=df["Importance"],
                        colorscale="Viridis",
                        showscale=True,
                        colorbar=dict(title="Importance"),
                        symbol="triangle-down",
                    ),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=title,
            xaxis_title="Location",
            yaxis=dict(
                ticktext=df["KDA"].tolist(),
                tickvals=list(range(len(df))),
                title="KDA",
                showticklabels=True,
                automargin=True,
                side="left",
                tickmode="array",
                ticklabelposition="outside",
                tickangle=0,
            ),
            xaxis=dict(side="bottom"),
            height=max(400, 150 * len(df)),
        )

        if show_plot:
            fig.show()

        if save_plot:
            fig.write_html(f"{plot_config['output_path']}/{dataset_title}_heatmap.html")
