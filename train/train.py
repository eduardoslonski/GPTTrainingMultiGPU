import os
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from model import Model
from utils import format_magnitude
from dataset import ModelDataset
from trainer import Trainer
from torch.optim.lr_scheduler import LambdaLR
import deepspeed
import argparse
import yaml

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

def get_train_obj(config):
    model = Model(config).to(config["device"])
    #optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer"]["params"]["lr"], betas=(config["beta1"], config["beta2"]))
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    return model, loss_fn

def get_datasets(config):
    train_perc = 1
    val_perc = 0.05
    
    print("loading dataset train")
    dataset_train = ModelDataset(path_data="data/train.bin", path_data_importance=None, context_length=config["context_length"])
    dataset_train = Subset(dataset_train, range(int(train_perc * len(dataset_train))))
    print(f"loaded dataset train len {len(dataset_train):,}")

    if rank == 0:
        print(f"{format_magnitude(len(dataset_train) * config['context_length'])} training tokens")

    if config["use_val_set"]:
        dataset_val = ModelDataset("data/validation.bin", config["context_length"])
        dataset_val = Subset(dataset_val, range(int(val_perc * len(dataset_val))))

        if rank == 0:
            print(f"{format_magnitude(len(dataset_val) * config['context_length'])} training tokens")
    else:
        dataset_val = None

    return dataset_train, dataset_val

def main():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank passed from distributed launcher')

    with open("ds_config.yml", 'r') as fr:
        config = yaml.load(fr, Loader=yaml.FullLoader)
    config["vocab_size"] = len(tokenizer.vocab)

    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    deepspeed.init_distributed(dist_backend='nccl')

    model, loss_fn = get_train_obj(config)
    dataset_train, dataset_val = get_datasets(config)

    std = 0.008 #math.sqrt(2/(config["hidden_size"]*3)) / math.sqrt(2.0 * config["num_layers"])
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, torch.nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    model_engine, optimizer_engine, dataloader_train, lr_scheduler = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     training_data=dataset_train,
                                                     model_parameters=model.parameters(),
                                                     config=config)

    if rank == 0:
        model.num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("===============================================================")
        print(f"{format_magnitude(model.num_parameters)} parameters")
        print(f"{format_magnitude(len(dataset_train) * config['context_length'])} training tokens")
        print(f"{format_magnitude((len(dataloader_train) // config['gradient_accumulation_steps']) * config['num_epochs'])} training steps")
        print("===============================================================")

    trainer = Trainer(model_engine, optimizer_engine, loss_fn, lr_scheduler, config, dataloader_train, dataloader_val=None, tokenizer=tokenizer)
    trainer.train()

if __name__ == "__main__":
    main()