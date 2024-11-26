<!-- omit in toc -->
# ⭐ MDTAIM: Multi-Dimensional Time-Series Anomaly Detection and Itemset Mining


[![Python Version](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Matrix Profile](https://img.shields.io/badge/matrixprofile-1.1.10-blue?style=flat)](https://pypi.org/project/matrixprofile/)
[![SPMF](https://img.shields.io/badge/spmf-2.62-blue?style=flat)](https://www.philippe-fournier-viger.com/spmf/)
[![GitHub stars](https://img.shields.io/github/stars/armanbehkish/mdtaim)](https://github.com/armanbehkish/mdtaim/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/armanbehkish/mdtaim)](https://github.com/armanbehkish/mdtaim/network)
[![GitHub issues](https://img.shields.io/github/issues/armanbehkish/mdtaim)](https://github.com/armanbehkish/mdtaim/issues)
[![GitHub license](https://img.shields.io/github/license/armanbehkish/mdtaim)](https://github.com/armanbehkish/mdtaim/blob/master/LICENSE)


<!-- omit in toc -->
## TABLE OF CONTENTS
- [🔍 About](#-about)
- [🛠️ Installation](#️-installation)
- [📊 Dataset](#-dataset)
- [📝 Configuration](#-configuration)
- [📚 Libraries](#-libraries)
- [📚 Documentation](#-documentation)
- [📜 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)
- [📧 Contact](#-contact)
- [📊 Project Status](#-project-status)


## 🔍 About

This repository is an implementation of a research done to detect multi-dimensional time-series anomalies using a novel method. It builds on top of existing anomaly detection scoring funcitons (Primarily matrix profile) and leveragtes the output as transactions to mine frequent itemsets. We try to investigate the possibility of using current efficient frequent itemset mining algorithms as a fast way to detect multi-dimensional anomalies.

## 🛠️ Installation

```bash
# make sure you have poetry installed
poetry --version

# Clone the repository
git clone https://github.com/armanbehkish/mdtaim.git

# Navigate to the directory
cd mdtaim

# install dependencies
poetry install

# activate the virtual environment
poetry shell

# make sure to double check the configuration file in config/config_<dataset>.yaml

# run the code
python main.py

# check the ouputs in /data/output/final
# check the plots in /plots
```

<!-- ## 💻 Usage

Here's a simple example of how to use your project:

```python
from myproject import MyClass

# Create an instance
obj = MyClass()

# Do something
result = obj.do_something()
``` -->

<!-- ## 📊 Demo

![Demo GIF](path/to/demo.gif)

You can try the live demo [here](https://demo-link.com) -->

## 📊 Dataset

Dataset location: `data/input/raw/<dataset_name>`
- create if doesn't exist
- Put the CSV files of the dataset and the ground truth in this directory!


## 📝 Configuration

Project configuration file: `config/config_<dataset>.yaml`
- create the file if doesn't exist (you can use config_toy.yaml as a template)
- fill in the parameters according to the documentation  
  

    
**GENERAL CONFIGURATION SECTIONS:** 

***Anomaly scoring:*** settings to calculate anomaly scores
```yaml
anomalyscoring:
  which: matrixprofile
  matrixprofile: 
    subsequence_length: 10 
    auto_subsequence_length: False 
  iForest:
    num_trees: 100    
```

***itemset mining:*** settings to convert anomaly scores into transactions

```yaml
itemset_mining_preparation:
  window_size: 10
  ignore_win_smaller_than: 0.5
  windowing_method: energy 
  enable_threshold_tuning: True 
  train_size: 150
  threshold_tuning_step: 0.2
  custom_threshold: 2  
  compare_to_train_for_detection: True 
  cut_baseline: False 
  quantile: 0.8
  utility_function: max  
  cons_trans_chk_for_merge: 4  
```

***SPMF Settings:*** settings to configure the choosen SPMF freauent itemset mining algorithm, not all settings are used for all algorithms, refer to documentation for more details!

```yaml
spmf:
  algorithm: Apriori 
  min_support: 0.5%
  max_support: 1%
  min_support_count: 1 
  max_pattern_length: 3
  min_pattern_length: 1
  show_transaction_ids: False  
  high_utility_itemsets: False
  min_utility: 1
  empty_trans_replacement: 1000
  sort_input_items:
    enable: True 
    ascending: True
  replace_zero:  
    enable: True 
    replace_zero_with: 99
  jar_file: ./lib/spmf.jar
```

***Data Settings:*** settings to prepare the data, set the dataset name here!

```yaml
data:
  dataset_title: toy
  dataset_file_name: toy_data.csv
  dataset_gt_file_name: toy_data_GT.csv
  ground_truth_type: range
  dataset_path: ./data/raw/toy/
  spmf_output_path: ./data/output/spmf/
  final_output_path: ./data/output/final/
  processed_data_path: ./data/processed/
  transactions_path: ./data/processed/saved_transactions/
  scores_path: ./data/processed/saved_scores/
  transaction_db_path: ./data/processed/transaction_databases/
  label_pad_size: 10
```

***Log Settings:*** settings used to configure the logging, Log level and location!

```yaml
logging:
  console_log_level: INFO
  log_dir: ./logs/
  log_file_prefix: dev
```

***Plot Settings:***: configure the plotting output.

```yaml
plot:
  output_path: ./plots/
  subplot_size: 160
```

## 📚 Libraries

We use [MATRIX PROFILE](https://matrixprofile.docs.matrixprofile.org/index.html)  to extract anomaly scores and [SPMF](https://www.philippe-fournier-viger.com/spmf/) for varous itemset mining algorithms. Also, some code excerpts were used in the anomaly scoring module from the [TSADIS](https://sites.google.com/view/tsadis) project.  
 
 


## 📚 Documentation

<!-- Check the comprehensive documentation at [documentation](link-to-docs). -->

<!-- ## 🤝 Contributing

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request -->

## 📜 License

<!-- This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. -->

## 👏 Acknowledgments

<!-- - Person/Library 1 for [what they did]
- Person/Library 2 for [what they did] -->

## 📧 Contact

- Email -  [arman dot behkish at gmail dot com]


<!-- ## 🗺️ Roadmap

- [x] Feature 1
- [ ] Feature 2
- [ ] Feature 3 -->

## 📊 Project Status

Project is: _in progress_