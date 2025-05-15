# Anomaly Detection in RL (ANOMRL)

This repository contains code for training agents, collecting & using datasets and training & evaluating anomaly detectors for RL.

Note: for the standalone anomaly-gym environments see: https://github.com/anonymous-3-141592653/anomaly-gym


## Installation


- Recommended python version: 3.10
- Recommended OS: Ubuntu 22.04
- Recommended Hardware: Nvidia-GPU with at least 24GB of GPU-Memory, e.g.  RTX 3090 or later (Image observations and larger models e.g. PredNet require more VRAM)

we recommend using uv(https://docs.astral.sh/uv/). If you don't have it, install it with: 

    $ curl -LsSf https://astral.sh/uv/install.sh | sh

then run :

    uv sync --all-extras

activate venv with 
   
    source .venv/bin/activate


## Usage


### Using anomaly-gym environments

Using anomaly gym environments is as simple as:

````python
import gymnasium
import anomaly_gym
gymnasium.make("Anom_Sape-Goal1", anomaly_type="obs_noise", anomaly_strength="strong") 
````



### Using pre-generated datasets


download our datasets:

    curl -L -o anomaly-gym.zip https://www.kaggle.com/api/v1/datasets/download/anonymous31459/anomaly-gym && unzip anomaly-gym.zip -d data/datasets/ && rm anomaly-gym.zip

Load data in python
```python
from anomrl.data.datamodules import VectorObsDataModule
data_module = VectorObsDataModule(env_id="URMujoco-Reach", data_dir="data/datasets/AGYMv1")
train_set = data_module.train_set
anomaly_set = data_module.get_split("action_offset_extreme")

# this should print
# EpisodeData(
#   observations:torch.Size([201, 13]), 
#   actions:torch.Size([200, 4]), 
#   reward:-185.37, 
#   infos:dict_keys(['cost', 'is_anomaly', 'is_critical', 'is_success']) 
# )
```



### Using the whole pipeline

Anomaly Detection in RL is a three stage process. 

#### Stage 1) - training an agent

Use `python anomrl/exp/train_agent.py` and hydra cli.

```bash
python anomrl/exp/train_agent.py +agent=tqc_multi +env.env_id=URMujoco-Reach
```

Pre-trained agents are found under `./data/trained_agents/*`

Info on trained agents:
     
- CarlaEnv: trained with default TQC params
- UR-Envs: all trained with tqc_multi.yaml
- Mujoco-Envs: all trained with tqc_large.yaml
- Sape-Envs: all trained with standard tqc.yaml

#### Stage 2) - collect data

Collect the data for training the detector. For simplicty, we use `anomrl/exp/collect_all_datasets.py` to collect train, validation, test and anomalous data in one go. 

first set some env variables

    $ export DATASETS_DIR=$(pwd)/data/datasets
    $ export TRAINED_AGENTS_DIR=$(pwd)/data/trained_agents

then run

```bash
python anomrl/exp/collect_all_datasets.py --env URMujoco-Reach --data_tag my-dataset --n_episodes 10 --anomaly_onset random --record_img_obs --seed 123
```


#### Stage 3) - evaluate detector

Finally evaulate the detector. For simplicity, we train and evaluate in one single script(`anomrl/exp/eval_detector.py`), using the data created in the step before.

```bash
python anomrl/exp/eval_detector.py +data_module.env_id=URMujoco-Reach +data_module.data_dir=data/datasets/TQC-my-dataset-onset_random-seed_123 +detector=knn
```


