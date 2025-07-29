import re
import os
import torch
import numpy as np
import operator
from statistics import mean,stdev
import fnmatch

import shutil
from datetime import datetime


def make_save_dir(args):
    if args.stage == "single_modal" and args.baseline_model in ["mTand_txt", 'MIMIC4_TextModel']:
        save_dir = args.save_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                   args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ns_embed_dim)
    elif args.stage == "single_modal" and args.baseline_model in ["UTDE_MIMIC3", "UTDE_MIMIC4"]:
        save_dir = args.save_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                   args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ts_embed_dim)
    elif args.stage == "pretrain" or args.stage == "finetune":
        now = datetime.now()
        save_dir = args.save_dir + args.stage + "/" + args.exp_name + "_seed_" + str(args.seed) + "_batch_" + str(
            args.train_batch_size) + "_ts_model_" + str(args.ts_model) + "_ns_model_" + str(
            args.ns_model) + now.strftime("%Y%m%d")
    args.save_dir=save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    print(args.save_dir)

def make_log_dir(args):
    if args.stage == "single_modal" and args.baseline_model in ["mTand_txt", 'MIMIC4_TextModel']:
        log_dir = args.log_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                  args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ns_embed_dim)
    elif args.stage == "single_modal" and args.baseline_model in ["UTDE_MIMIC3", "UTDE_MIMIC4"]:
        log_dir = args.log_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                   args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ts_embed_dim)
    elif args.stage == "pretrain" or args.stage == "finetune":
        now = datetime.now()
        log_dir = args.log_dir + args.stage + "/" + args.exp_name + "_seed_" + str(args.seed) + "_batch_" + str(
            args.train_batch_size) + "_ts_model_" + str(args.ts_model) + "_ns_model_" + str(
            args.ns_model) + now.strftime("%Y%m%d")
    args.log_dir = log_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

def make_tensorboard_dir(args):
    if args.stage == "single_modal" and args.baseline_model in ["mTand_txt", 'MIMIC4_TextModel']:
        tensorboard_dir = args.tensorboard_dir + args.task + "/" + args.exp_name + "_seed_" + str(
            args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                          args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ns_embed_dim)
    elif args.stage == "single_modal" and args.baseline_model in ["UTDE_MIMIC3", "UTDE_MIMIC4"]:
        tensorboard_dir = args.tensorboard_dir + args.task + "/" + args.exp_name + "_seed_" + str(args.seed) + "_epoch_" + str(
            args.num_train_epochs) + "_batch_" + str(args.train_batch_size) + "_" + \
                   args.model_name + str(args.max_length) + "_emb_dim_" + str(args.ts_embed_dim)
    elif args.stage == "pretrain" or args.stage == "finetune":
        now = datetime.now()
        tensorboard_dir = args.tensorboard_dir + args.stage + "/" + args.exp_name + "_seed_" + str(args.seed) + "_batch_" + str(
            args.train_batch_size) + "_ts_model_" + str(args.ts_model) + "_ns_model_" + str(
            args.ns_model) + now.strftime("%Y%m%d")
    args.tensorboard_dir = tensorboard_dir
    if not os.path.exists(args.tensorboard_dir):
        os.makedirs(args.tensorboard_dir)


def save_pretrain_ckpt(save_path, model):
    torch.save(model.state_dict(), save_path)

def load_full_model(args, device):
    return torch.load(args.full_model, map_location=device)

def load_model(model_path, device):
    return torch.load(model_path, map_location=device)

def load_best_full_model(args, best_epoch, device):
    save_dir = args.save_dir
    best_ck = save_dir + '/' + str(best_epoch) + '.pth'
    return torch.load(best_ck, map_location=device)


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_dir=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_dir = save_dir
        self.best_epoch = -1

    def __call__(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None, global_step=0):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, time_predictor, decoder, epoch, global_step)
            if epoch is not None:
                self.best_epoch = epoch
        elif score <= self.best_score + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience} ({self.val_loss_min:.6f} --> {val_loss:.6f})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, classifier, time_predictor, decoder, epoch, global_step)
            if epoch is not None:
                self.best_epoch = epoch
            self.counter = 0

    def save_checkpoint(self, val_loss, model, classifier=None, time_predictor=None, decoder=None, epoch=None, global_step=0):
        '''
        Saves model when validation loss decrease.
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        model_state_dict = model.state_dict()
        if classifier is not None:
            classifier_state_dict = classifier.state_dict()
        else:
            classifier_state_dict = None

        if self.save_dir is not None:
            save_path = f"{self.save_dir}/{epoch}_{global_step}.pth"
            torch.save({
                'model_state_dict': model_state_dict,
                'classifier': classifier_state_dict,
            }, save_path)
        else:
            print("no path assigned")

        self.val_loss_min = val_loss


if __name__ == "__main__":
    dst='test/'
    copy_file(dst, src=os.getcwd())
