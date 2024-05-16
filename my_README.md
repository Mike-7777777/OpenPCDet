
# A Hands-on Guide to Training PointPillars Models on SCC with OpenPCDet

Group members: QS, Leo, Ivan, Alex.

Responsible for this article: QS aka @Mike-7777777

- [Prepare the Data](https://github.com/Mike-7777777/OpenPCDet/blob/master/my_README.md#prepare-the-dataset)
- [Config the HPC system](https://github.com/Mike-7777777/OpenPCDet/blob/master/my_README.md#config-the-goe-scc-hpc-systemscc)
- [The OUTPUT of our model](https://github.com/Mike-7777777/OpenPCDet/blob/master/my_README.md#the-output-folder-the-Results)
- The Data Analysis...

## Prepare the Dataset

Before use this repo, you should already have the custom datasets:

```
- custom
    -- calib
        --- 000000.txt
        --- ...
    -- image_2
        --- 000000.png
        --- ...
    -- ImageSets
        --- test.txt
        --- train.txt
        --- trainval.txt
        --- val.txt
    -- label_2
        --- 000000.txt
        --- ...
    -- velodyne
        --- 000000.bin
        --- ...
```

If the datasets isn't ready, please use the dev-kit(https://github.com/Mike-7777777/tum-traffic-dataset-dev-kit) to prepare it. Or you can just download the processed datasets here: https://1drv.ms/f/s!AoH0J35av8mcnKM6V6xOHcqx17LOzg?e=R3jaAF

## Config the Goe-SCC HPC system(SCC)

### Conn to SCC

Please follow the instructions:
https://docs.gwdg.de/doku.php?id=en:services:application_services:high_performance_computing:connect_with_ssh

You should:

- Get into GoNET using Goe-VPN(if you are not in the campus network)
- Use SSH to connect the SCC system

SSH config file example:

```
Host scc-ss24-smart-city
	Hostname login-mdc.hpc.gwdg.de
	User u11423 # replace by your project username
	IdentityFile .ssh/goe24 # replace by your private key file location

Host scc-transfer-ss24-smart-city
	Hostname transfer-scc.hpc.gwdg.de
	User u11423 # replace by your project username
	IdentityFile .ssh/goe24 # replace by your private key file location
```

### Transfer TUMTraf Dataset to SCC

Example:

```bash
scp -r "I:/tumtraf/custom" u11423@transfer-scc.hpc.gwdg.de:/user/sun.qumeng/u11423/OpenPCDet/data/custom
```

### Build the env

1. Init the env:

```bash
git clone https://github.com/Mike-7777777/OpenPCDet.git
cd OpenPCDet

module load anaconda3
module load cuda/11.8

conda create -n openpcd python=3.8 -y
conda init 
source ~/.bashrc
conda activate openpcd
```

2. Edit the bashrc file:

```bash
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
module load gcc
module load anaconda3
module load cuda/11.8

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/sw/rev/23.12/linux-scientific7-haswell/gcc-11.4.0/anaconda3-2023.09-0-npsf7i/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/sw/rev/23.12/linux-scientific7-haswell/gcc-11.4.0/anaconda3-2023.09-0-npsf7i/etc/profile.d/conda.sh" ]; then
        . "/opt/sw/rev/23.12/linux-scientific7-haswell/gcc-11.4.0/anaconda3-2023.09-0-npsf7i/etc/profile.d/conda.sh"
    else
        export PATH="/opt/sw/rev/23.12/linux-scientific7-haswell/gcc-11.4.0/anaconda3-2023.09-0-npsf7i/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source activate openpcd
export TORCH_CUDA_ARCH_LIST="7.5" # for RTX5000 https://developer.nvidia.com/cuda-gpus
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
```

3. Install the OpenPCDet(https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)

```bash
source ~/.bashrc
pip install -r requirements.txt

pip install spconv-cu118
python setup.py develop
```

Note: This may take some time, less than ten minutes in my case.

If you have syntax error, try `pip3 install pynvml --upgrade`.

More information: https://gitlab-ce.gwdg.de/hpc-team-public/deep-learning-with-gpu-cores/-/tree/main?ref_type=heads

### Useful commands

```bash
# check the disk space
du -ah --max-depth=1 ./ | sort -rh | head -n 10

# check the gpus
sinfo -o "%25N  %5c  %10m  %32f  %10G %18P " | grep gpu
```

## Use OpenPCDet in SCC

### Create Dataset

Please ensure that the files and folders in the `custom` directory have been moved to `/OpenPCDet/data/custom`.

```bash
cd OpenPCDet
python -m pcdet.datasets.custom.custom_dataset create_custom_infos tools/cfgs/dataset_configs/custom_dataset.yaml
```

You will find the data is ready in `data/custom`.

### Train the model

```bash
cd tools
sbatch submit_train.sh
```

## The OUTPUT folder, the Results

### Overview

This folder contains the results and checkpoints of the training and evaluation phases of the model. The information is saved in the `\output\user\sun.qumeng\u11423\OpenPCDet\tools\cfgs\custom_models\pps\default` directory, which includes various subdirectories and files.

Download link: https://1drv.ms/u/s!AoH0J35av8mcnOcGV5VgGaWtJrnbWA?e=8cKpQw

### Directory Structure

- **`ckpt`**:
  - This folder contains the checkpoint files from the 50th to the 80th epochs. Due to space constraints, we uploaded the folder on OneDrive rather than GitHub.
  
- **`eval/eval_with_train`**:
  - This directory holds the results of the evaluation phase.
    - **`epoch_80/val/result.pkl`**: This file is used to save data for assessment results. You can load this file using the pickle library to view the details of the assessment results.
    - **`tensorboard_val`**: This folder stores data files related to TensorBoard, a visualization tool for displaying metrics generated during the evaluation.

- **`tensorboard`**:
  - This folder contains TensorBoard data files for the training phase, similar to `tensorboard_val` but for training metrics.

### Detailed Description

#### Checkpoints

- **`ckpt/`**:
  - Checkpoints are saved periodically during training to allow for recovery and continuation of training from a specific state. 
  - You can load the checkpoint in your training script to resume training or for inference purposes.

#### Evaluation Results

- **`eval/eval_with_train/epoch_80/val/result.pkl`**:
  - This pickle file contains the detailed results of the evaluation after the 80th epoch.
  - To view the contents of this file, you can use the following Python code:
    ```python
    import pickle
    
    with open('result.pkl', 'rb') as f:
        data = pickle.load(f)
        print(data)
    ```
  - The evaluation metrics include various performance indicators of the model on the validation dataset.

- **`eval/eval_with_train/tensorboard_val`**:
  - This directory contains TensorBoard event files that store metrics and other information generated during the evaluation phase.
  - You can visualize these metrics by running TensorBoard:
    ```bash
    tensorboard --logdir=path_to_tensorboard_val_directory
    ```
  - Then, open your web browser and go to `http://localhost:6006` to view the evaluation metrics.

#### Training Metrics

- **`tensorboard/`**:
  - Similar to `tensorboard_val`, this directory contains TensorBoard event files for the training phase.
  - To visualize the training metrics, run TensorBoard:
    ```bash
    tensorboard --logdir=path_to_tensorboard_directory
    ```
  - Open your web browser and go to `http://localhost:6006` to view the training metrics.

### Note

The data is too large to upload in its entirety. Hence, it has been saved on a personal computer and will be uploaded to OneDrive for sharing. A link will be provided for access to the full dataset. 

This readme file is partially completed by GPT-4o. Thanks OpenAI.

### Contact

For any issues or questions regarding this dataset, please contact @Mike-7777777