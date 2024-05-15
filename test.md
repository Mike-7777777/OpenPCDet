
# TITLE

"/openpcdet-docker/data/tum_data"

"/OpenPCDet/data/custom/training"

python3 -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml

pip install kornia==0.6.8

python3 train.py --cfg_file "/OpenPCDet/tools/cfgs/custom_models/pp.yaml"

AttributeError: 'EasyDict' object has no attribute 'BACKUP_DB_INFO'
yaml augment

https://github.com/open-mmlab/OpenPCDet/issues/1300
