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
import threading
import torch.multiprocessing as mp
from torch.multiprocessing import Manager
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
        self.args.plug_num = self.args.c_out // self.args.cseg_len
        model = self.model_dict[self.args.model].Model(self.args).float()
        tun_models = []
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_models.append(SimpleFCNet_channel(self.args, sequence_length=self.args.pred_len*self.args.cseg_len))               
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model, tun_models

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        tun_optim_models = []

        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_optim_models.append(optim.Adam(self.tun_models[i].parameters(), lr=self.args.learning_rate))

        return model_optim, tun_optim_models

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    

    def train_one_plug(self, plug, start_time, path, early_stop, count, lock):
        
        train_data, train_loader = self._get_data(flag='train')
        train_steps = len(train_loader)
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        early_stopping = EarlyStopping(args=self.args, verbose=True)
        model_optim, tun_optim_models = self._select_optimizer()
        criterion = self._select_criterion()
        
        
        
        self.model = self.model.to(self.device)
        if self.args.tun_model:
            self.tun_models[plug] = self.tun_models[plug].to(self.device)

        plug_time = time.time()
        self.model.train()
        if self.args.tun_model:
            self.tun_models[plug].train() 

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            epoch_loss = []
            iter_count = 0
            time_now = time.time()  # 修复time_now未定义的问题

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                tun_optim_models[plug].zero_grad()

                # 将数据移至对应设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Forward pass
                with torch.set_grad_enabled(True):
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    plug_result = self.tun_models[plug](
                        outputs[:, :, plug*self.args.cseg_len:(plug+1)*self.args.cseg_len]
                    )
                    loss = criterion(
                        plug_result, 
                        batch_y[:, :, plug*self.args.cseg_len:(plug+1)*self.args.cseg_len]
                    )
                
                # Backward pass
                loss.backward()
            
                # 使用锁保护共享资源访问
                model_optim.step()
                if early_stop[plug] == 0:
                    tun_optim_models[plug].step()
            
                epoch_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time}")
        
            # 计算验证损失
            epoch_loss = np.average(epoch_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion, plug)
            test_loss = self.vali(test_data, test_loader, criterion, plug)
        
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | "
                  f"Train Loss: {epoch_loss:.7f} Vali Loss: {vali_loss:.7f} "
                  f"Test Loss: {test_loss:.7f}")

            # 早停检查
            with lock:
                early_stop, count = early_stopping(
                    vali_loss, self.model, self.tun_models, 
                    self.args, start_time, path, plug
                )
            
                if self.args.tun_model and count[plug] >= self.args.patience:
                    print("Early stopping")
                    break

                adjust_learning_rate(model_optim, tun_optim_models, epoch + 1, self.args)

        print(f'plug:{plug + 1} cost time:{time.time() - plug_time}')
        return

    def train(self, setting):
        start_time = time.time()
        path = os.path.join(self.args.checkpoints, setting)
        load_path = path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'
        
        if not os.path.exists(path):
            os.makedirs(path)

        # 加载和冻结模型
        if self.args.tun_model:
            self.model.load_state_dict(torch.load(load_path))
            self.model = self.model.to(self.device)
            for param in self.model.parameters():
                param.requires_grad = False

        # 使用线程锁
        lock = threading.Lock()
        early_stop = [0] * self.args.plug_num
        count = [0] * self.args.plug_num
        
        # 创建线程池
        with ThreadPoolExecutor(max_workers=self.args.plug_num) as executor:
            futures = []
            for plug in range(self.args.plug_num):
                future = executor.submit(
                    self.train_one_plug,
                    plug, start_time, path, early_stop, count, lock
                )
                futures.append(future)
            
            # 等待所有线程完成
            for future in futures:
                future.result()

        # 加载最佳模型
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_model_path = f"{path}/tun_model{i}.pth"
                self.tun_models[i].load_state_dict(torch.load(tun_model_path))

        return self.model, self.tun_models

    def vali(self, vali_data, vali_loader, criterion, number):
        if self.args.tun_model:
            total_loss = [[] for _ in range(self.args.plug_num)]

        val_loss = []

        self.model.eval()
        if self.args.tun_model:
            self.tun_models[number].eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 修复 self.self.device 为 self.device
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    # 修复 self.self.device 为 self.device
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                # 修复 self.self.device 为 self.device
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # 修复 self.self.device 为 self.device
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
                batch_x = batch_x.float().to(self.self.device)
                batch_y = batch_y.float().to(self.self.device)

                if 'PEMS' in self.args.data or 'Solar' in self.args.data:
                    batch_x_mark = None
                    batch_y_mark = None
                else:
                    batch_x_mark = batch_x_mark.float().to(self.self.device)
                    batch_y_mark = batch_y_mark.float().to(self.self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.self.device)
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
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.self.device)
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

