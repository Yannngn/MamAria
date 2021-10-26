import torch
import numpy as np
from utils import save_predictions_as_imgs

class EarlyStopping:

    def __init__(self, patience=10, mode="min", delta=0.001, wait = 50):

        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.wait = wait

        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf


    def __call__(self, epoch_score, checkpoint, checkpoint_path, epoch):

        if self.mode == "min":
            score = -1.0 * epoch_score
        else:
            score = np.copy(epoch_score)

        if self.best_score is None:
            self.best_score = score
            self.best_checkpoint(epoch_score, checkpoint, checkpoint_path)
        elif score < self.best_score + self.delta:
            if epoch >= self.wait:
                self.counter += 1

                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_checkpoint(epoch_score, checkpoint, checkpoint_path)
            self.counter = 0


    def best_checkpoint(self, epoch_score, checkpoint, checkpoint_path):

        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:

            print(f"Validation score improved ({self.val_score} --> {epoch_score}). Saving model!")

            torch.save(checkpoint, checkpoint_path)

        self.val_score = epoch_score