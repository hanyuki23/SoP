from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools_inner_early_stop import EarlyStopping, SocketEarlyStopping, adjust_learning_rate, visual
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
import types 
from utils.augmentation import run_augmentation,run_augmentation_single
import time
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):

        if self.args.channel_fintune:
            self.args.plug_num = self.args.c_out // self.args.cseg_len
        else:
            self.args.plug_num = self.args.pred_len// self.args.cseg_len

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

        count = [ 0 for _ in range(self.args.plug_num)]        

        path = os.path.join(self.args.checkpoints, setting)
        load_path = path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(args = self.args, verbose=True)
        origin_early_stopping = SocketEarlyStopping(args = self.args, verbose=True)

        
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # load freeze model
        if self.args.tun_model:
            self.model.load_state_dict(torch.load(load_path))
            # 获取所有层
            all_layers = []
            for name, module in self.model.named_modules():
                # if isinstance(module, nn.Linear):  # 只选择线性层
                all_layers.append((name, module))
            
            # 从后向前选择要解冻的层
            target_layers = all_layers[-self.args.unfreeze_layers:]
            last_layer_name, last_layer = target_layers[-1]  # 最后一层
            other_layers = target_layers[:-1]  # 其他要解冻的层
            
            # 1. 冻结所有参数
            self.model.requires_grad_(False)
            
            # 2. 处理其他解冻层 - 只解冻不分段
            for layer_name, layer in target_layers:
                print(f"Unfreezing layer (without segmentation): {layer_name}")
                for param in layer.parameters():
                    param.requires_grad = True       

            # 3. 处理最后一层 - 解冻并分段
            print(f"Unfreezing and segmenting last layer: {last_layer_name}")
            in_features = last_layer.in_features
            out_features = last_layer.out_features
            print(f"Last layer input features: {in_features}, output features: {out_features}")
            print(f"Plug number: {self.args.plug_num}")
            # 确保输出维度可以被plug_num整除
            assert out_features % self.args.plug_num == 0, "输出维度必须能被plug_num整除"
            segment_size = out_features // self.args.plug_num     

            # 为最后一层创建分段
            last_layer.segment_layers = []
            last_layer.plug_num = self.args.plug_num
            segment_layers = []     

            # 创建分段并初始化
            for i in range(self.args.plug_num):
                start_idx = i * segment_size
                end_idx = (i + 1) * segment_size
                
                segment_layer = nn.Linear(in_features, segment_size).to(self.device)
                with torch.no_grad():
                    segment_layer.weight.copy_(last_layer.weight[start_idx:end_idx, :])
                    if last_layer.bias is not None:
                        segment_layer.bias.copy_(last_layer.bias[start_idx:end_idx])
                
                segment_layers.append(segment_layer)
            
            last_layer.segment_layers = segment_layers

            # 修改最后一层的forward方法
            def new_forward(self, x):
                outputs = []
                for i in range(self.plug_num):
                    segment_output = self.segment_layers[i](x)
                    outputs.append(segment_output)
                return torch.cat(outputs, dim=-1)
            
            last_layer.forward = types.MethodType(new_forward, last_layer)

            # 创建优化器
            # 1. 为其他解冻层创建普通优化器
            other_params = []
            for _, layer in other_layers:
                other_params.extend(layer.parameters())
            
            # 2. 为最后一层的分段创建单独的优化器
            segment_optimizers = []
            for segment_layer in segment_layers:
                segment_layer.requires_grad_(True)
                # 创建包含两组参数的优化器
                optimizer = optim.Adam([
                    {'params': other_params, 'lr': self.args.learning_rate},  # 其他层参数
                    {'params': segment_layer.parameters(), 'lr': self.args.learning_rate}  # 当前分段参数
                ])
                segment_optimizers.append(optimizer)
        else:
            model_optim , tun_optim_models = self._select_optimizer()

        for variable in range(self.args.plug_num):
            self.model.train()
            variable_start_time = time.time()
            iter_count = 0
            
            for epoch in range(self.args.train_epochs):
                self.model.train()
                train_loss = [] 

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):

                    iter_count += 1

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

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs = outputs.to(self.device)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                                      
                    # 计算分段损失
                    segment_optimizers[variable].zero_grad()
                    if self.args.channel_fintune:
                        segment_output = outputs[:, :, variable:variable + 1]
                        segment_target = batch_y[:, :, variable:variable + 1]
                    else:
                        segment_output = outputs[:, variable:variable + 1,: ]
                        segment_target = batch_y[:, variable:variable + 1,: ]                        
                    
                    segment_loss = criterion(segment_output, segment_target)
                    train_loss.append(segment_loss.item())

                    # 反向传播
                    segment_loss.backward()
                    # 优化分段参数
                    segment_optimizers[variable].step()

                    if (i + 1) % 500 == 0:
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, segment_loss.item()))
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                        print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                        iter_count = 0
                        time_now = time.time()

                val_loss= self.vali(vali_data, vali_loader, criterion, variable)
                test_loss= self.vali(test_data, test_loader, criterion, variable)

                print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(epoch + 1, np.mean(train_loss), val_loss, test_loss))
                
                count = early_stopping(val_loss, self.args, self.model, path, variable)

                if count[variable] == self.args.patience:
                    print("Early stopping")
                    break

            adjust_learning_rate(segment_optimizers[variable], epoch + 1, self.args)
            print('variable [{}/{}], cost time{}'.format(variable + 1, self.args.plug_num, time.time() - variable_start_time))
        
        print('total cost time:', time.time() - start_time)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))  

        return self.model

    def vali(self, vali_data, vali_loader, criterion, variable):
        val_loss = []

        self.model.eval()
        
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

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred_cpu = outputs.detach().cpu()
                true_cpu = batch_y.detach().cpu()
                if self.args.channel_fintune:
                    loss = criterion(pred_cpu[:, :, variable:variable+1], true_cpu[:, :, variable:variable+1])
                else:
                    loss = criterion(pred_cpu[:, variable:variable+1, :], true_cpu[:, variable:variable+1, :])

                val_loss.append(loss.item())

        val_loss = np.average(val_loss)

        return val_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, setting)

        if test:
            print('loading model')   
            # self.model.load_state_dict(torch.load(path + '/' + 'checkpoint.pth'))              
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
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
                        # inner_fintune model
                        output_finnal = []
                        for j in range(self.args.plug_num):
                            variable_model_path = f"{path}/{j}_inner_paprams.pth"
                            self.model.load_state_dict(torch.load(variable_model_path))  
                            if self.args.channel_fintune:
                                output_finnal.append(self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:,:,j:j+1])
                            else:
                                output_finnal.append(self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[:,j:j+1,:])
                        if self.args.channel_fintune:
                            outputs = torch.cat(output_finnal, dim=2).detach().cpu()
                        else:
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

