from mdtaim.pipeline import Pipeline


def main():
    ### Set the dataset title: "toy" or
    # "mscred" or "pvsystem" or "mgab" or
    # "diff_types_0" or
    # "Synth_vd_20k_10d_20a" or
    # "Synth_van_50k_20d_60a" or
    # "Synth_G_5k_15d_75a"
    # "Thesis_example_1"
    # dataset_title = "mscred"
    dataset_title = "mscred"
    ###

    pipeline = Pipeline(dataset_title)
    pipeline.load_data()
    pipeline.cal_anomaly_score()
    pipeline.cal_kdp()
    pipeline.convert_anomalies_to_transactions()
    pipeline.perform_itemset_mining()
    # pipeline.perform_postprocessing()


if __name__ == "__main__":
    main()
