import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import time

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, optimizers, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    elif args.lradj =='int':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.channel_fintune:
            for i in range(args.c_out // args.cseg_len):
                for param_group in optimizers[i].param_groups:
                    param_group['lr'] = lr                
        else:
            for i in range(args.pred_len // args.cseg_len):
                for param_group in optimizers[i].param_groups:
                    param_group['lr'] = lr

        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, args, verbose=False, delta=0):
        self.patience = args.patience
        self.verbose = verbose
 
        if args.channel_fintune:
            self.counter = [ 0 for _ in range(args.c_out)]
            self.best_avg_score = [np.Inf for _ in range(args.c_out)]
            self.early_stop = [ 0 for _ in range(args.c_out)]
        else:
            self.counter = [ 0 for _ in range(args.pred_len)]
            self.best_avg_score = [np.Inf for _ in range(args.pred_len)]
            self.early_stop = [ 0 for _ in range(args.pred_len)]            
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, vali_average_loss, model, tun_models, args, start_time, path, number):
        args_tun = args.tun_model
        args_len = args.pred_len
        args_cslen = args.cseg_len
        channel_fintune = args.channel_fintune
        c_out = args.c_out
        score = vali_average_loss

        if self.best_avg_score[0] == None:
            self.best_avg_score = score

        if self.early_stop[number] == 0 :
            if score < self.best_avg_score[number]:
                torch.save(tun_models[number].state_dict(), f"{path}/tun_model{number}.pth")
                self.counter[number] = 0
                self.best_avg_score[number] = score
            else:
                self.counter[number] += 1
            if self.counter[number] == self.patience:
                self.early_stop[number] = 1
                print('stop count {0} cost:{1}'.format(number, time.time()-start_time))
           
        return self.early_stop, self.counter
        

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)