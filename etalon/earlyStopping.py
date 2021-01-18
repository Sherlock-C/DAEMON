import torch
import numpy as np
import os


class EarlyStopping:

    def __init__(self, opt, patience=7, verbose=False, delta=0):

        '''

        :param opt:
        :param patience (int): How long to wair after last time validation loss improved

        :param verbose: If true, prints the information of validation loss improvement

        :param delta: Minimum change in the monitored quantity to qualify as an improvement
        '''

        self.opt = opt
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model_G, model_D_rec, model_D_lat):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score

            self.save_checkpoint(val_loss, model_G, model_D_rec, model_D_lat)

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Earlystopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score

            self.save_checkpoint(val_loss, model_G, model_D_rec, model_D_lat)

            self.counter = 0

    def save_checkpoint(self, val_loss, model_G, model_D_rec, model_D_lat):

        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        save_dir = os.path.join(self.opt.outf, self.opt.model, self.opt.dataset)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(model_G.state_dict(),
                   os.path.join(save_dir, self.opt.model + "_folder_" + str(self.opt.filename) + '_G.pkl'))
        # torch.save(self.D_rec.state_dict(),
        #            os.path.join(save_dir, self.opt.model + "_folder_" + str(self.opt.folder) + '_D_rec.pkl'))
        # torch.save(self.D_lat.state_dict(),
        #            os.path.join(save_dir, self.opt.model + "_folder_" + str(self.opt.folder) + '_D_lat.pkl'))