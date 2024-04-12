# GPT Training Multi-GPU

This repository is used to train Large Language Models on a multi-GPU setup.

### Model
Standard GPT architecture with RoPE. All code in the repo is hackable for experiments and improvements.

### Training
- Utilizes [DeepSpeed](https://github.com/microsoft/DeepSpeed) to enhance training performance across multiple GPUs.
- Launch training using `deepspeed train/train.py`.
- Configuration for both training and model is available at [ds_config.yml](ds_config.yml).
- The official DeepSpeed documentation for its configurations can be found [here](https://www.deepspeed.ai/docs/config-json/).
- The Trainer supports `checkpoints` and monitoring using [Weights & Biases](https://wandb.ai/), with configurations for both available in the config file.
- The Trainer supports `gradient accumulation`, managing the necessary adaptations for checkpoints and logging when using this strategy.
- The Trainer optionally supports [Flash Attention 2](https://github.com/Dao-AILab/flash-attention), but it is not mandatory for operation.

### Data
- The Trainer is compatible with any binary data file in `uint16` format containing stacked tokenized samples.
- You can download a small-sized sample tokenized data directly from S3 using `python train/download_data.py --object slimpajama/train/tokenized/worker_1.bin --file_path data/train.bin` (1.3 billion tokens), or a medium-sized one with the full script available at [download_data.sh](download_data.sh) (11 billion tokens).
- The data is sampled from [Slimpajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B), tokenized using the GPT-NeoX tokenizer. The entire dataset contains 627 billion tokens, suitable for training models in this repo.

### Getting Started
- Clone the repo and run `pip install -r requirements.txt`.
- Your setup needs to support multi-GPU intercommunication.
- Launch training using `deepspeed train/train.py`.
