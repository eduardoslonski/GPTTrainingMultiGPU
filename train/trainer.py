import os
import torch
import wandb
from dataclasses import asdict
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from utils import format_magnitude, format_lr
import numpy as np

world_size = int(os.environ['WORLD_SIZE'])
rank = int(os.environ['RANK'])

class Trainer():
    def __init__(self, model, optimizer, loss_fn, lr_scheduler, config, dataloader_train, dataloader_val, tokenizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.tokenizer = tokenizer

    def train(self):
        if self.config["use_wandb"] and rank == 0:
            wandb.init(
            project = "gpt_training",
            name = self.config["wandb_run_name"],
            config = {"config": self.config},
            notes = f"{self.model.num_parameters:,} parameters"
            )
            #wandb.watch(self.model)

        if rank == 0:
            pbar = tqdm("Training", total=((len(self.dataloader_train)) // self.config["gradient_accumulation_steps"])*self.config["num_epochs"])
            tokens_seen = 0

        for i_epoch in range(self.config["num_epochs"]):
            for i_batch, batch in enumerate(self.dataloader_train):
                input_ids = batch["input_ids"].to(self.config["device"])
                labels = batch["labels"].to(self.config["device"])

                logits = self.model(input_ids)
                loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))

                if "importance" in batch:
                    value_loss_compare = 0.8
                    importance = batch["importance"].to(self.config["device"]).view(-1)
                    if rank == 0:
                        perc_tokens_usage = torch.sum(importance >= value_loss_compare) / torch.sum(labels.ne(-100)).item()
                    importance = torch.where(importance < value_loss_compare, torch.zeros_like(importance), importance)
                    loss_importance = (loss * importance).float()
                    if rank == 0:
                        loss_importance_var = (torch.std(loss_importance, unbiased=False) / torch.mean(loss_importance)).item()
                    loss_importance_sum_fp32 = loss_importance.sum()
                    loss_sum_fp32 = loss.float().sum()
                    loss_normalized_importance = (loss_importance / loss_importance_sum_fp32 * loss_sum_fp32).half()

                    loss_final = loss_normalized_importance.mean()
                else:
                    loss_final = loss.mean()

                if rank == 0:
                    tokens_seen += (torch.sum(labels.ne(-100)).item() * world_size)

                self.model.backward(loss_final)
                self.model.step()

                if (i_batch/self.config["gradient_accumulation_steps"] % self.config["checkpoint_every"] == 0 and i_batch > 0) or (i_batch == len(self.dataloader_train)-1 and i_epoch == self.config["num_epochs"]-1):
                    self._save_checkpoint(i_batch*(i_epoch+1))

                if i_batch % self.config["gradient_accumulation_steps"] == 0 and rank == 0:
                    if self.config["use_wandb"]:
                        wandb_dict = {}

                        wandb_dict["loss_train"] = loss_final.item()
                        #wandb_dict["loss_importance_var"] = loss_importance_var
                        wandb_dict["lr"] = self.model.get_lr()[0]
                        wandb_dict["tokens_seen"] = tokens_seen

                        wandb.log(wandb_dict)
                    pbar.update(1)
                    pbar.set_description(
                        f"[{i_epoch+1}/{self.config['num_epochs']}][{(i_batch//self.config['gradient_accumulation_steps'])+1}/{len(self.dataloader_train)//self.config['gradient_accumulation_steps']}] "
                        f"Loss {loss_final.item():.4f} "
                        f"Tokens seen {format_magnitude(tokens_seen)} "
                        #f"Perc tokens usage {perc_tokens_usage*100:.2f}% "
                        f"LR {format_lr(self.model.get_lr()[0])}"
                    )

    def _save_checkpoint(self, i):
        path = "saved_checkpoints"
        if rank == 0:
            print("saving checkpoint")
            if not os.path.exists(path):
                os.makedirs(path)
        self.model.save_checkpoint(f"{path}/checkpoint_{i//self.config['gradient_accumulation_steps']}")
        if rank == 0:
            print("checkpoint saved")