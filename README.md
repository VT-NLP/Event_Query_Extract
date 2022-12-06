<h1 align="center">Query and Extract: Refining Event Extraction as Type Oriented Binary Decoding
</h1>

## Table of Contents (Optional)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Trigger Detection](#trigger-detection)
- [Argument Detection](#argument-detection)
- [Citation](#citation)
- [License](#license)

## Installation
To install the dependency packages, run
```
conda create --name query_extract_EE python=3.8
conda activate query_extract_EE
pip install -r requirements.txt
```

## Data Preparation 

1. Follow https://github.com/wilburOne/ACE_ERE_Scripts to process the raw data and save to ./data/ace_en/processed_data 
and ./data/ere_en/processed_data respectively
2. Save the event data into .txt files, process the .txt file and save as Torch TensorDataset
```angular2html
./setup.sh
```
## Trigger Detection
To train the trigger detection model, run
```
python scripts/run_trigger_detection.py --tr_dataset=${PATH_TO_TRAIN_DATASET} --dev_dataset=${PATH_TO_DEV_DATASET} 
```
To evaluate the trigger detection performance, run
```
python scripts/run_trigger_detection.py --EPOCH=0 --te_dataset=${PATH_TO_TEST_DATASET} 
```



## Argument Detection
To train the argument detection model, run
```
python scripts/run_argument_detection.py --train_file_pt=${PATH_TO_TRAIN_FILE} --dev_file_pt=${PATH_TO_DEV_FILE}
```
To evaluate the argument detection model, run
```
python scripts/eval.py
```



## Citation
If you find this repo useful, please cite the following paper:
```
@inproceedings{wang2022query,
    title={Query and Extract: Refining Event Extraction as Type-oriented Binary Decoding},
    author={Wang, Sijia and Yu, Mo and Chang, Shiyu and Sun, Lichao and Huang, Lifu},
    booktitle={Findings of the 2022 Association for Computational Linguistics},  
    year={2022}
}
```


<!-- LICENSE -->
## License

Distributed under the MIT License.

