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
        show_plot: bool = True,
        labels_df: pd.DataFrame = pd.DataFrame(),
    ):
        itemset_config = self.config.get_config()["itemset_mining_preparation"]
        spmf_config = self.config.get_config()["spmf"]
        high_util_enabled = spmf_config["high_utility_itemsets"]
        dataset_title = self.config.get_config()["data"]["dataset_title"]
        plot_config = self.config.get_config()["plot"]
        win_size = itemset_config["window_size"]
        title = (
            dataset_title.upper()
            + " Dataset - "
            + "Triangles represent Multidimensional Anomalies Found, Lines represent Dataset Labels."
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

        # create combined list of KDAs from labels and results so that they can be shown together/compared in the plot
        def kda_to_set(kda_str):
            return set(map(str.strip, kda_str.split(",")))

        all_kdas = []
        for kda in labels_df["KDA"]:
            all_kdas.append((kda, kda_to_set(kda)))

        for kda in df["KDA"]:
            kda_set = kda_to_set(kda)
            if kda not in [x[0] for x in all_kdas]:
                for i, (existing_kda, existing_set) in enumerate(all_kdas):
                    if kda_set.issubset(existing_set):
                        all_kdas.insert(i + 1, (kda, kda_set))
                        break
                    elif kda_set.issuperset(existing_set):
                        all_kdas.insert(i, (kda, kda_set))
                        break
                else:
                    all_kdas.append((kda, kda_set))

        ordered_kdas = [x[0] for x in all_kdas]

        # create text colors adaptive based on marker color
        min_imp = df["Importance"].min()
        max_imp = df["Importance"].max()
        df["norm_imp"] = (df["Importance"] - min_imp) / (max_imp - min_imp)

        self.logger.debug(f"results dataframe created for heatmap:\n {df}\n")
        self.logger.debug(f"labels dataframe from heatmap function:\n {labels_df}\n")
        self.logger.debug(f"combinedordered_kdas: {ordered_kdas}")

        # Create the heatmap using a scatter plot with triangle markers
        fig = go.Figure()

        # Add traces for results
        for i, row in df.iterrows():
            # parse coordinates for each kda from df
            kda = row["KDA"]
            location = row["Location"]
            importance = round(row["Importance"], 1)
            text_color = "black" if row["norm_imp"] > 0.5 else "white"
            kda_index = ordered_kdas.index(kda)
            # add some offset to lift triangles above the line
            y_position = kda_index + 3 * (box_size / 1000)

            if not high_util_enabled:
                # for frequent itemsets results
                fig.add_trace(
                    go.Scatter(
                        x=[location],
                        y=[y_position],
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

                # for high utility itemsets results
                fig.add_trace(
                    go.Scatter(
                        x=[location],
                        y=[y_position],
                        mode="markers+text",
                        text=importance,
                        textposition="middle center",
                        textfont=dict(
                            color=text_color,
                            size=12,
                        ),
                        marker=dict(
                            size=box_size,
                            color=[importance],
                            cmin=min_imp,
                            cmax=max_imp,
                            colorscale="Viridis",
                            showscale=(i == 0),
                            # showscale=True,
                            colorbar=dict(title="Importance"),
                            symbol="triangle-down",
                        ),
                        showlegend=False,
                    )
                )

        # Add traces for label intervals
        for _, row in labels_df.iterrows():
            # parse coordinates for each kda from labels_df
            kda = row["KDA"]
            loc_start, loc_end = map(int, row["Location"].split(","))
            kda_index = ordered_kdas.index(kda)

            # for label intervals
            fig.add_trace(
                go.Scatter(
                    x=[loc_start, loc_end],
                    y=[kda_index, kda_index],
                    mode="lines+markers",
                    marker=dict(color="brown", size=6, symbol="circle"),
                    line=dict(color="brown", width=2),
                    showlegend=False,
                )
            )

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor="center",
                yanchor="top",
                font=dict(
                    size=20,
                    color="black",
                    family="Arial",
                ),
            ),
            yaxis=dict(
                ticktext=ordered_kdas,
                tickvals=list(range(len(ordered_kdas))),
                title=dict(
                    text="KDA",
                    font=dict(
                        size=16,
                        color="black",
                        family="Arial",
                        weight="bold",
                    ),
                ),
                showticklabels=True,
                automargin=True,
                side="left",
                tickmode="array",
                ticklabelposition="outside",
                tickangle=0,
            ),
            xaxis=dict(
                side="bottom",
                title=dict(
                    text="LOCATION",
                    font=dict(
                        size=16,
                        color="black",
                        family="Arial",
                        weight="bold",
                    ),
                ),
            ),
            height=max(600, 150 * len(labels_df)),
        )

        if show_plot:
            fig.show()

        if save_plot:
            fig.write_html(f"{plot_config['output_path']}/{dataset_title}_heatmap.html")
