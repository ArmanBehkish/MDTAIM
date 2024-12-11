from mdtaim.pipeline import Pipeline


def main():
    ### Set the dataset title: "toy" or "mscred" or "pvsystem"
    dataset_title = "pvsystem"
    ###

    pipeline = Pipeline(dataset_title)
    pipeline.load_data()
    pipeline.cal_anomaly_score()
    pipeline.cal_kdp()
    pipeline.convert_anomalies_to_transactions()
    pipeline.perform_itemset_mining()
    pipeline.perform_postprocessing()


if __name__ == "__main__":
    main()
