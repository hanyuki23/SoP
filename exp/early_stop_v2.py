from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools_early_stop import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.tuning import SimpleFCNet_channel,SimpleFCNet2,SimpleFCNet_traffic,SimpleFCNet_ett,SimpleFCNet_exchange,SimpleFCNet,SimpleFCNet_timestep2
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
    
    def calculate_channel_losses(self, outputs, batch_y, tun_models, early_stop):
        criterion = self._select_criterion()
        
        # 创建活跃通道掩码
        active_mask = torch.tensor(
            [not bool(es) for es in early_stop], 
            device=self.device
        )
        
        # 批量处理所有未早停通道
        B, T, _ = outputs.shape
        outputs_reshaped = outputs.view(B, T, self.args.plug_num, -1)
        batch_y_reshaped = batch_y.view(B, T, self.args.plug_num, -1)
        
        active_indices = torch.where(active_mask)[0]
        
        if len(active_indices) > 0:
            losses = []
            outputs_processed = []
            
            # 并行处理活跃通道
            for idx in active_indices:
                channel_out = tun_models[idx](
                    outputs_reshaped[:, :, idx]
                )
                outputs_processed.append(channel_out)
                losses.append(
                    criterion(
                        channel_out, 
                        batch_y_reshaped[:, :, idx]
                    )
                )
            
            total_loss = torch.stack(losses).mean()
            channel_losses = torch.zeros(
                self.args.plug_num, 
                device=self.device
            )
            channel_losses[active_indices] = torch.stack(losses)
        else:
            total_loss = torch.tensor(0.0, device=self.device)
            channel_losses = torch.zeros(
                self.args.plug_num, 
                device=self.device
            )
            
        return total_loss, channel_losses.detach().cpu().numpy().tolist()

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

    def train(self, setting):
        start_time = time.time()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if self.args.tun_model:
            early_stop = [ 0 for _ in range(self.args.plug_num)]          

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

        # freeze model
        if self.args.tun_model :
            self.model.load_state_dict(torch.load(load_path))
            for param in self.model.parameters():
                param.requires_grad = False

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.args.tun_model:
                self.model.eval()
                for i in range(self.args.plug_num):
                    self.tun_models[i].train()

            epoch_star_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                model_optim.zero_grad()
                if self.args.tun_model:
                    for j in range(self.args.plug_num):
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
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    outputs = outputs.to(self.device)
                    
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    output_finnal = []
                    # true = [None] * self.args.c_out
                    # pred = [None] * self.args.c_out
                    # tun_model
                    if self.args.tun_model:
                        if self.args.channel_fintune:
                            total_loss, channel_losses = self.calculate_channel_losses(
                                        outputs, batch_y, self.tun_models,early_stop  # 添加early_stop参数
                                                    )
                        else:
                            for j in range(self.args.pred_len // self.args.cseg_len):    
                                output_finnal.append(self.tun_models[j](outputs[:, j*self.args.cseg_len:self.args.cseg_len*(j+1), :]))                      
                            outputs = torch.cat(output_finnal, dim=1).to(self.device)
                    

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, total_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    if self.args.tun_model:
                        if self.args.channel_fintune:
                            for i in range(self.args.c_out // self.args.cseg_len):
                                scaler.step(tun_optim_models[i])
                        else:
                            for i in range(self.args.pred_len // self.args.cseg_len):
                                scaler.step(tun_optim_models[i])

                    scaler.update()
                else:
                    total_loss.backward()
                    train_loss.append(total_loss.item())

                    model_optim.step()
                    if self.args.tun_model:
                        for i in range(self.args.plug_num):
                            if early_stop[i] == 0:
                                tun_optim_models[i].step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_star_time))

            train_loss = np.average(train_loss)

            vali_average_loss = self.vali(vali_data, vali_loader, criterion, early_stop)
            vali_avg = sum(vali_average_loss) / len(vali_average_loss)
            test_average_loss = self.vali(test_data, test_loader, criterion, early_stop)
            test_avg = sum(test_average_loss) / len(test_average_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_avg, test_avg))
            
            # tun model
            early_stop, count = early_stopping(vali_average_loss, self.model, self.tun_models, self.args, start_time, path, early_stop)
            if self.args.channel_fintune :
                if np.sum(count) == self.args.patience * (self.args.plug_num) :
                    print("Early stopping")
                    break          

            adjust_learning_rate(model_optim, tun_optim_models, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        # tun model 
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_model_path = f"{path}/tun_model{i}.pth" 
                self.tun_models[i].load_state_dict(torch.load(tun_model_path))     

        return self.model, self.tun_models

    def vali(self, vali_data, vali_loader, criterion, early_stop):
        if self.args.tun_model:
            total_loss = [[] for _ in range(self.args.plug_num)]
        val_loss = []

        self.model.eval()
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                self.tun_models[i].eval()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

                output_finnal = []
                if self.args.channel_fintune:
                    true = [None] * (self.args.c_out // self.args.cseg_len)
                    pred = [None] * (self.args.c_out // self.args.cseg_len)
                else:
                    true = [None] * (self.args.pred_len // self.args.cseg_len)
                    pred = [None] * (self.args.pred_len // self.args.cseg_len)                   

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                # tun_model
                if self.args.tun_model:
                    if self.args.channel_fintune:
                        for j in range(self.args.c_out // self.args.cseg_len):
                            output_finnal.append(self.tun_models[j](outputs[:, :, j*self.args.cseg_len:self.args.cseg_len*(j+1)]))
                            true[j] = batch_y[:, :, j*self.args.cseg_len:(j+1)*self.args.cseg_len].detach().cpu()
                            pred[j] = output_finnal[j].detach().cpu()
                            total_loss[j].append(criterion(pred[j], true[j]))
                        outputs = torch.cat(output_finnal, dim=2).detach().cpu()
                    else:
                        for j in range(self.args.pred_len // self.args.cseg_len):
                            output_finnal.append(self.tun_models[j](outputs[:, j*self.args.cseg_len:self.args.cseg_len*(j+1), :]))
                            true[j] = batch_y[:, j*self.args.cseg_len:self.args.cseg_len*(j+1), :].detach().cpu()
                            pred[j] = output_finnal[j].detach().cpu()
                            total_loss[j].append(criterion(pred[j], true[j]))                            
                        outputs = torch.cat(output_finnal, dim=1).detach().cpu()

                pred_cpu = outputs.detach().cpu()
                true_cpu = batch_y.detach().cpu()

                loss = criterion(pred_cpu, true_cpu)
                val_loss.append(loss)

        average_losses = [sum(sublist) / len(sublist) for sublist in total_loss]
        val_loss = np.average(val_loss)


        self.model.train()
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                self.tun_models[i].train()
        return average_losses

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'))
            if self.args.tun_model:
                path2 = os.path.join(self.args.checkpoints, setting)
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
