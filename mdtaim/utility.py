"""
Utility functions
"""

import logging
import os
from datetime import datetime
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Optional, Dict, Any

# ANSI Color/Style Codes
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"
UNDERLINE = "\033[4m"


def setup_logging(config_path: str) -> logging.Logger:
    """
    Setup project-wide logging,
    log file is created in the logs/ directory (path in config.yaml)
    log file saves all logs
    console logs INFO and above (set in config.yaml)
    """

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        # creating log files using configuration and time
        log_file_prefix = config.get("logging", {}).get("log_file_prefix", "mdtaim")
        log_dir = config.get("logging", {}).get("log_dir", "./logs/")
        log_file = os.path.join(
            log_dir,
            log_file_prefix + "_" + datetime.now().strftime("%Y%m%d_%H%M") + ".log",
        )
        if not os.path.exists(log_file):
            os.makedirs(log_dir, exist_ok=True)
            open(log_file, "w").close()
    except Exception as e:
        print(f"Error creating log file: {e}")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    # set file log level to DEBUG (all logs)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    # set console log level from configuration,
    console_handler.setLevel(
        getattr(logging, config.get("logging", {}).get("console_log_level", "WARNING"))
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def clear_old_log_files(config_path: str) -> None:
    """
    Clear log files in the logs directory
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_dir = config.get("logging", {}).get("log_dir", "./logs/")
    for file in os.listdir(log_dir):
        # if file date is before today, delete it
        if file.split("_")[1] < datetime.now().strftime("%Y%m%d"):
            os.remove(os.path.join(log_dir, file))


def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def underline(text: str) -> str:
    return f"{UNDERLINE}{text}{RESET}"


def red(text: str) -> str:
    return f"{RED}{text}{RESET}"


def green(text: str) -> str:
    return f"{GREEN}{text}{RESET}"


def blue(text: str) -> str:
    return f"{BLUE}{text}{RESET}"


def md_plot(
    data: np.ndarray,
    labels: np.ndarray,
    logger: logging.Logger,
    plot_config: Dict[str, Any],
    regimes=None,
    each_tag_thrs=None,
    title: str = "result",
    plot_box: bool = True,
    save_plot: bool = True,
    idx_tag_name: int = 0,
    idx_name: int = 0,
    idx_string: str = None,
    line_color: str = "gray",
    show_plot: bool = False,
    name: str = "TS",
):
    """General Plotting function"""

    if data is None or labels is None:
        logger.error("data or labels are not provided!")
        raise ValueError("data or labels are not provided!")

    num_subplots = data.shape[1]
    colors = [
        "black",
        "blue",
        "red",
        "cyan",
        "magenta",
        "orange",
        "darkgreen",
        "tomato",
        "darkblue",
        "darkred",
    ]
    colors = colors * 100
    figsize = (30, min(num_subplots * 0.5, 40))
    names = [f"{name}{i}" for i in np.arange(data.shape[0])]
    if name == "KDP":
        names = [f"{i} DP" for i in np.arange(1, data.shape[1] + 1)]
    font_size = max(6, 22 - num_subplots // 5)

    max_data = np.max(data)
    min_data = np.min(data)

    x = np.arange(idx_name, data.shape[0] + idx_name)

    plt.rcParams.update({"font.size": font_size})

    fig, axes = plt.subplots(data.shape[1], 1, figsize=figsize, sharex=True)
    if num_subplots == 1:
        axes = [axes]

    from mdtaim.processdata import PreprocessData

    for i, ax in enumerate(axes):
        if idx_string is not None:
            ax.plot(
                x,
                data[:, i],
                color=line_color,
                label=f"{idx_tag_name+i}{idx_string}",
                linewidth=0.8,
            )
        elif names is not None:
            ax.plot(x, data[:, i], color=line_color, label=f"{names[i]}", linewidth=0.8)
        else:
            ax.plot(
                x, data[:, i], color=line_color, label=idx_tag_name + i, linewidth=0.8
            )
        # anchor the axes legend approximately right top corner
        ax.legend(
            fontsize=font_size - 5,
            loc="upper right",
            bbox_to_anchor=(0.9990, 0.9),
            borderaxespad=0.0,
        )
        if labels is not None:
            current_anomalies = np.asarray(
                PreprocessData.get_state_intervals(labels[:, i])[1]
            )
            for anom in current_anomalies:
                ax.axvspan(
                    anom[0] + idx_name,
                    anom[1] + idx_name,
                    facecolor=colors[i],
                    alpha=0.3,
                )
        if regimes is not None:
            for ii in regimes:
                ax.axvspan(
                    ii[0] + idx_name, ii[1] + idx_name, facecolor="gray", alpha=0.3
                )
        if each_tag_thrs is not None:
            ax.axhline(
                y=each_tag_thrs[i], color="r", alpha=0.4, linestyle="--", dashes=(5, 1)
            )

        ax.tick_params(axis="x", labelsize=font_size - 3)
        ax.tick_params(
            axis="y", labelsize=font_size - 4, labelleft=True, labelright=False
        )
        ax.set_ylim(bottom=min_data, top=max_data)

    plt.autoscale()
    if save_plot:
        plt.savefig(
            plot_config["output_path"] + f"{title}.svg", dpi=800, bbox_inches="tight"
        )
    if show_plot:
        plt.show()


def plot_plotly(
    data: np.ndarray,
    labels: np.ndarray,
    logger: logging.Logger,
    plot_config: Dict[str, Any],
    title: str = "result",
    message: str = "",
    save_plot: bool = True,
    show_plot: bool = False,
    line_color: str = "gray",
    y_common_range: bool = True,
    name: str = "TS",
    subplot_size: int = 100,
) -> None:

    from mdtaim.processdata import PreprocessData

    global_min = np.min(data)
    global_max = np.max(data)

    subplot_bgcolor = "white"
    line_colors = ["lightgray", "gray", "darkgray", "dimgray", "slategray"]

    fig = make_subplots(
        rows=data.shape[0],
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.005,
        # vertical_spacing=0.5,
    )

    for i, ts in enumerate(data, start=1):
        fig.add_trace(
            go.Scatter(
                x=np.arange(data.shape[1]),
                y=ts,
                mode="lines",
                name=f"Time Series {i-1}",
                showlegend=False,
                line=dict(color=line_colors[i % len(line_colors)]),
            ),
            row=i,
            col=1,
        )
        # Add annotation (label) to the right of each subplot
        fig.add_annotation(
            text=f"{name}{i-1}" if name != "KDP" else f"{i} DP",
            xref=f"x{i+1} domain",
            yref=f"y{i+1} domain",
            x=1.003,
            y=0.5,
            showarrow=False,
            font=dict(size=15),
            xanchor="left",
            yanchor="middle",
            row=i,
            col=1,
        )
        if y_common_range:
            fig.update_yaxes(range=[global_min, global_max], row=i, col=1)

        fig.update_yaxes(
            showgrid=False,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="black",
            row=i,
            col=1,
            color="black",
        )
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor="lightgray",
            row=i,
            col=1,
            color="black",
        )

        this_lbls = np.asarray(PreprocessData.get_state_intervals(labels[:, i - 1])[1])
        for label in this_lbls:
            fig.add_shape(
                type="rect",
                x0=label[0],
                x1=label[1],
                y0=global_min,
                y1=global_max,
                xref=f"x{i}",
                yref=f"y{i}",
                fillcolor="yellow",
                opacity=0.85,
                layer="below",
                line_width=0,
                row=i,
                col=1,
            )

    fig.update_layout(
        plot_bgcolor=subplot_bgcolor,
        height=data.shape[0] * subplot_size,
        title=message,
    )
    if show_plot:
        fig.show()
    if save_plot:
        fig.write_html(plot_config["output_path"] + f"{title}.html")
        # fig.write_image(plot_config["output_path"] + f"{title}.jpeg")
