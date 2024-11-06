from typing import Optional, Dict
import timeit
import numpy as np
from mdtaim.config import Config
from mdtaim.utility import setup_logging, clear_old_log_files, bold, green
from mdtaim.processdata import PreprocessData
from mdtaim.anomalyscore import MatrixProfile
from mdtaim.kdp import KDP
from mdtaim.itemset import ItemSetPreparation, SPMF
from mdtaim.postprocess import PostProcess


class Pipeline:
    """
    Pipeline for the MDTAIM pipeline
    """

    def __init__(self, dataset_title: str) -> None:
        """
        Initialize the pipeline
        """
        self.dataset_title = dataset_title
        conf_file_name = f"config_{dataset_title}.yaml"
        self.logger_obj = setup_logging(f"config/{conf_file_name}")
        self.config_obj = Config(
            self.logger_obj, config_path=f"config/{conf_file_name}"
        )
        self.config_obj.load_config()
        self.logger_obj.info("current project config is: \n%s", self.config_obj)
        clear_old_log_files(f"config/{conf_file_name}")
        self.data: np.ndarray = np.array([])
        self.labels: np.ndarray = np.array([])
        self.padded_labels: np.ndarray = np.array([])
        self.mp_scores: np.ndarray = np.array([])
        self.timers: Dict[str, float] = {}

    def load_data(self) -> None:
        """
        Load the data
        """
        data_obj = PreprocessData(self.logger_obj, self.config_obj)
        data_obj.load_data()
        self.data, self.labels = data_obj.get_data_and_labels()

        data_obj.plot(title=self.dataset_title, label_type="normal", show_plot=False)
        self.logger_obj.info(
            "anomaly rate of labels:%s", data_obj.cal_anomaly_rate(self.labels)
        )

        self.padded_labels = data_obj.pad_labels()

        self.logger_obj.info(
            "anomaly rate of padded labels: %s",
            data_obj.cal_anomaly_rate(self.padded_labels),
        )

    def cal_anomaly_score(self) -> None:
        """
        Calculate the anomaly score (Matrix Profile)
        """
        mp_start_time = timeit.default_timer()
        mp_obj = MatrixProfile(self.logger_obj, self.config_obj)
        self.mp_scores = mp_obj.calculate_score(self.data)
        mp_end_time = timeit.default_timer()
        self.timers["mp"] = mp_end_time - mp_start_time

        self.logger_obj.info(
            bold(
                green(
                    f"MP calculation done! total time to calculate Matrix Profile: {self.timers['mp']:.3f} seconds"
                )
            )
        )

        mp_obj.plot(
            title=f"{self.dataset_title}_MATRIX_PROFILE",
            line_color="blue",
            label_type="padded",
            labels=self.padded_labels,
            show_plot=False,
        )

    def cal_kdp(self) -> None:
        """
        Calculate the KDP
        """
        kdp_start_time = timeit.default_timer()
        kdp_obj = KDP(self.logger_obj, self.config_obj)
        kdp_obj.fast_find_anomalies(self.mp_scores)
        kdp_end_time = timeit.default_timer()
        self.timers["kdp"] = kdp_end_time - kdp_start_time

        self.logger_obj.info(
            bold(
                green(
                    f"KDP calculation done! total time to calculate KDP: {self.timers['kdp']:.3f} seconds"
                )
            )
        )

        kdp_obj.plot(
            title=f"{self.dataset_title}_KDP",
            line_color="brown",
            label_type="padded",
            labels=self.padded_labels,
            show_plot=False,
        )

    def convert_anomalies_to_transactions(self) -> None:
        """
        Convert Matrix Profile Discords to Transactions readable by SPMF
        """
        itemsetp_obj = ItemSetPreparation(self.logger_obj, self.config_obj)
        itemsetp_obj.load_anomaly_scores()

        convert_start_time = timeit.default_timer()
        itemsetp_obj.convert_anomalies_to_transactions()
        itemsetp_obj.transactions.merge_consecutive_anomalies()
        convert_end_time = timeit.default_timer()
        self.timers["convert"] = convert_end_time - convert_start_time

        itemsetp_obj.print_transactions(to="logs")
        itemsetp_obj.cal_anomaly_detec_accuracy(self.padded_labels)

        itemsetp_obj.transactions.save_transactions_to_file()

    def perform_itemset_mining(self) -> None:
        """
        Perform Itemset Mining using SPMF
        """
        spmf_obj = SPMF(self.logger_obj, self.config_obj)
        spmf_obj.load_transactions()
        spmf_obj.prepare_transaction_database()

        spmf_start_time = timeit.default_timer()
        spmf_obj.run_algorithm()
        spmf_end_time = timeit.default_timer()
        self.timers["spmf"] = spmf_end_time - spmf_start_time

        self.logger_obj.info(
            bold(
                green(
                    f"MP conversion to transactions took {self.timers['convert']:.3f} seconds and SPMF execution took {self.timers['spmf']:.3f} seconds"
                )
            )
        )

    def perform_postprocessing(self) -> None:
        """
        Perform Postprocessing
        """
        postprocess_obj = PostProcess(self.config_obj, self.logger_obj)
        postprocess_obj.produce_output()
