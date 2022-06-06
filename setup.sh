mkdir error_visualizations
mkdir logs
mkdir saved_models
mkdir outputs
mkdir data/ace_en/pt/
mkdir data/ere_en/pt/

#conda create --name query_extract_EE python=3.8
#conda activate query_extract_EE
#pip install -r requirements.txt


export ProjDir="$(pwd)"

# process raw data and write it into .txt file
python preprocess/ace/read_args_with_entity_ace.py
python preprocess/ere/read_args_with_entity_ere.py

# save to .pt file
python preprocess/save_dataset.py --ace --data_folder=${DATA_FOLDER_ACE}
python preprocess/save_dataset.py --ere --data_folder=${DATA_FOLDER_ERE}
