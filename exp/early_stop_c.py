from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools_plug_stop import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from models.tuning import SimpleFCNet_channel,SimpleFCNet2,SimpleFCNet_traffic,SimpleFCNet_ett,SimpleFCNet_exchange,SimpleFCNet,SimpleFCNet_timestep
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
import time
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        sequence_length = self.args.pred_len
        in_features = self.args.c_out

        model = self.model_dict[self.args.model].Model(self.args).float()
        tun_models = []
        if self.args.channel_fintune:
            for i in range(self.args.c_out // self.args.cseg_len):
                tun_models.append(SimpleFCNet_channel(self.args, sequence_length=self.args.pred_len*self.args.cseg_len))        
        else:   
            for i in range(self.args.pred_len // self.args.cseg_len):
                tun_models.append(SimpleFCNet_timestep(self.args, sequence_length=self.args.c_out*self.args.cseg_len))        
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model, tun_models

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        tun_optim_models = []
        if self.args.channel_fintune:
            for i in range(self.args.c_out // self.args.cseg_len):
                tun_optim_models.append(optim.Adam(self.tun_models[i].parameters(), lr=self.args.learning_rate))
        else:
            for i in range(self.args.pred_len // self.args.cseg_len):
                tun_optim_models.append(optim.Adam(self.tun_models[i].parameters(), lr=self.args.learning_rate))
        return model_optim, tun_optim_models

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.tun_model:
            if self.args.channel_fintune:
                early_stop = [ 0 for _ in range(self.args.c_out // self.args.cseg_len)]
                plug_num = self.args.c_out // self.args.cseg_len
            else:
                early_stop = [ 0 for _ in range(self.args.pred_len // self.args.cseg_len)]     
                plug_num = self.args.pred_len // self.args.cseg_len       

        path = os.path.join(self.args.checkpoints, setting)
        load_path = path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'
        
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(args = self.args, verbose=True)

        model_optim , tun_optim_models = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # load socket model and freeze
        if self.args.tun_model:
            self.model.load_state_dict(torch.load(load_path))
            for param in self.model.parameters():
                param.requires_grad = False
        
        # plug fintune
        for plug in range(plug_num):
            plug_time = time.time()
            self.model.train()
            if self.args.tun_model: 
                self.tun_models[plug].train()
            
            for epoch in range(self.args.train_epochs):
                epoch_loss = []
                epoch_time = time.time()
                iter_count = 0
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    iter_count +=1 
                    model_optim.zero_grad()
                    for j in range(plug_num):
                        tun_optim_models[j].zero_grad()

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                        batch_x_mark = None
                        batch_y_mark = None
                    else:
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                    # encoder - decoder
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            if self.args.output_attention:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                            loss = criterion(outputs, batch_y)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs.to(self.device)     

                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    plug_result = self.tun_models[plug](outputs[:, :, plug*self.args.cseg_len:(plug+1)*self.args.cseg_len])  

                    loss = criterion(plug_result, batch_y[:, :, plug*self.args.cseg_len:(plug+1)*self.args.cseg_len])
                    epoch_loss.append(loss.item())

                    if (i + 1) % 500 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()
                    
                    loss.backward()
                    model_optim.step()
                    for i in range(plug_num):
                        if early_stop[i] == 0:
                            tun_optim_models[i].step()

                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

                epoch_loss = np.average(epoch_loss)
                
                vali_loss = self.vali(vali_data, vali_loader, criterion, plug)
                test_loss = self.vali(test_data, test_loader, criterion, plug)
                

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, epoch_loss, vali_loss, test_loss))
                
                # tun model
                early_stop, count = early_stopping(vali_loss, self.model, self.tun_models, self.args, start_time, path, plug)

                if self.args.tun_model :
                    if count[plug] == self.args.patience :
                        print("Early stopping")
                        break            

                adjust_learning_rate(model_optim, tun_optim_models, epoch + 1, self.args) 
            print('plug:{} cost time:{}'.format(plug + 1, time.time() - plug_time))                 

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # load new plug 
        if self.args.tun_model:
            for i in range(plug_num):
                tun_model_path = f"{path}/tun_model{i}.pth" 
                self.tun_models[i].load_state_dict(torch.load(tun_model_path))  

        return self.model, self.tun_models

    def vali(self, vali_data, vali_loader, criterion, number):
        if self.args.channel_fintune:
            total_loss = [[] for _ in range(self.args.c_out// self.args.cseg_len)]
        else:
            total_loss = [[] for _ in range(self.args.pred_len // self.args.cseg_len)]
        val_loss = []

        self.model.eval()
        if self.args.tun_model:
            self.tun_models[number].eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                
                # if self.args.channel_fintune:
                #     true = [None] * (self.args.c_out // self.args.cseg_len)
                #     pred = [None] * (self.args.c_out // self.args.cseg_len)
                # else:
                #     true = [None] * (self.args.pred_len // self.args.cseg_len)
                #     pred = [None] * (self.args.pred_len // self.args.cseg_len)                   

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # plug fintune 
                if self.args.tun_model:
                    plug_result = self.tun_models[number](outputs[:, :, number*self.args.cseg_len:self.args.cseg_len*(number+1)])

                pred_cpu = plug_result.detach().cpu()
                true_cpu = batch_y[:, :, number*self.args.cseg_len:self.args.cseg_len*(number+1)].detach().cpu()

                loss = criterion(pred_cpu, true_cpu)
                val_loss.append(loss)

        val_loss = np.average(val_loss)

        self.model.train()
        if self.args.tun_model:
            self.tun_models[number].train()
        return val_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'))
            if self.args.tun_model:
                path2 = os.path.join(self.args.checkpoints, setting)
                # path2 = '/home/hyq/my_task/Time-Series-Library-main/checkpoints/long_term_forecast_solar_96_720_TSMixer_Solar_ftM_sl96_ll96_pl720_dm512_nh8_el2_dl1_df32_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_1_1'
                if self.args.channel_fintune:
                    for i in range(self.args.c_out // self.args.cseg_len):
                        tun_model_path = f"{path2}/tun_model{i}.pth"
                        self.tun_models[i].load_state_dict(torch.load(tun_model_path))  
                else:
                    for i in range(self.args.pred_len // self.args.cseg_len):
                        tun_model_path = f"{path2}/tun_model{i}.pth"
                        self.tun_models[i].load_state_dict(torch.load(tun_model_path))                  
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.args.tun_model:
            if self.args.channel_fintune:
                for i in range(self.args.c_out // self.args.cseg_len):
                    self.tun_models[i].eval()
            else:
                for i in range(self.args.pred_len // self.args.cseg_len):
                    self.tun_models[i].eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # tunmodel
                output_finnal = []
                if self.args.tun_model:
                    if self.args.channel_fintune:
                        for j in range(self.args.c_out // self.args.cseg_len):
                            output_finnal.append(self.tun_models[j](outputs[:, :, j*self.args.cseg_len:self.args.cseg_len*(j+1)]))
                        outputs = torch.cat(output_finnal, dim=2).detach().cpu()
                    else:
                        for j in range(self.args.pred_len // self.args.cseg_len):
                            output_finnal.append(self.tun_models[j](outputs[:, j*self.args.cseg_len:self.args.cseg_len*(j+1), :]))
                        outputs = torch.cat(output_finnal, dim=1).detach().cpu()

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
