from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from src.model import Model
from src.params import Parameters
from src.vocab import Vocab


class Trainer:
    def __init__(self, model: Model, params: Parameters, vocab: Vocab, optimizer, train_data, val_data):
        self.model = model
        self.params = params
        self.vocab = vocab
        self.optimizer = optimizer
        self.train_data = train_data
        self.val_data = val_data
        self.loss = self.params.LOSS
        self.epoch_loss = {"train": [], "validation": []}

        self.model.to(self.params.DEVICE)
        self.loss.to(self.params.DEVICE)

    def train(self):
        for epoch in range(self.params.NUM_EPOCH):
            print("Starting epoch ", epoch+1)
            self.train_data_loader = DataLoader(self.train_data, batch_size= self.params.BATCH_SIZE, shuffle= False, num_workers=4)
            self.val_data_loader = DataLoader(self.val_data, batch_size= self.params.BATCH_SIZE, shuffle= False, num_workers=4)
            self.train_epoch()
            self.validate_epoch()
            print(f"""Epoch: {epoch+1}/{self.params.NUM_EPOCH}\n""", 
            f"""    Train Loss: {self.epoch_loss["train"][-1]:.4}\n""",
            f"""    Valid Loss: {self.epoch_loss["validation"][-1]:.4}\n""",
            """\n""")

            if (epoch+1) == self.params.NUM_EPOCH:
                self.save(checkpoint=False)
            else:
                self.save(checkpoint=True, epoch=epoch)

    def train_epoch(self):
        self.model.train()
        running_loss = []

        print("Started training.")
        for batch in tqdm(self.train_data_loader):
            if len(batch[0]) == 0:
                continue

            inputs = batch[0].to(self.params.DEVICE)
            targets = batch[1].to(self.params.DEVICE)

            pos_ground_truth = torch.ones([targets.shape[0], 1])
            neg_ground_truth = torch.zeros([targets.shape[0], targets.shape[1] - 1])
            ground_truths = torch.cat([pos_ground_truth, neg_ground_truth], dim=1).to(self.params.DEVICE)

            self.optimizer.zero_grad()
            outputs = self.model(inputs, targets).to(self.params.DEVICE)
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)
            loss_value = self.loss(outputs, ground_truths)
            loss_value.backward()
            self.optimizer.step()

            running_loss.append(loss_value.item())

        self.epoch_loss["train"].append(np.mean(running_loss))

    def validate_epoch(self):
        self.model.eval()
        running_loss = []

        print("Started validation.")
        with torch.no_grad():
            for batch in tqdm(self.val_data_loader):
                if len(batch[0]) == 0:
                    continue
               
                inputs = batch[0].to(self.params.DEVICE)
                targets = batch[1].to(self.params.DEVICE)

                pos_ground_truth = torch.ones([targets.shape[0], 1]).to(self.params.DEVICE)
                neg_ground_truth = torch.zeros([targets.shape[0], targets.shape[1] - 1]).to(self.params.DEVICE)
                ground_truths = torch.cat([pos_ground_truth, neg_ground_truth], dim=1).to(self.params.DEVICE)

                preds = self.model(inputs, targets).to(self.params.DEVICE)
                if preds.dim() == 1:
                    preds = preds.unsqueeze(0)
                loss_value = self.loss(preds, ground_truths)

                running_loss.append(loss_value.item())
            
            self.epoch_loss["validation"].append(np.mean(running_loss))

    def save(self, checkpoint, epoch=0):
        if checkpoint:
            checkpoint = {"epoch": epoch+1, 
                        "model_state_dict": self.model.state_dict(), 
                        "optimizer_state_dict": self.optimizer.state_dict(), 
                        "last_train_loss": self.epoch_loss["train"][-1], 
                        "last_validation_loss": self.epoch_loss["validation"][-1]}
            torch.save(checkpoint, self.params.CHECKPOINT_PATH)
            print("Model checkpoint saved.")
        else:
            torch.save(self.model, self.params.MODEL_PATH)
            print("Final model saved.")

        torch.save(self.model.target_embed_matrix.weight.data, self.params.EMBED_PATH)
        print("Embeddings are saved.")
