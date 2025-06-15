from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools_early_stop import EarlyStopping, SocketEarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.tuning import SimpleFCNet_channel,SimpleFCNet2,SimpleFCNet_traffic,SimpleFCNet_ett,SimpleFCNet_exchange,SimpleFCNet
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from sklearn.cluster import AgglomerativeClustering
import time
warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 添加梯度记录器
        self.gradient_recorder = {
            'directions': [],
            'similarity_matrices': [],
            'steps': []
        }

    def _build_model(self):

        self.args.plug_num = self.args.c_out // self.args.cseg_len

        model = self.model_dict[self.args.model].Model(self.args).float()
        tun_models = []

        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_models.append(SimpleFCNet_channel(self.args, sequence_length=self.args.pred_len*self.args.cseg_len))                 
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model, tun_models

    def set_variate_clusters(self, x = None):

        x = x.T
        # x shape (n_variates,seq_len)
        inds = np.arange(0,self.args.enc_in)

        mapping = {i:int(ind) for i,ind in enumerate(inds)}
        corr = np.corrcoef(x)

        distance_threshold = 0.5
        n_clusters = None

        '''
        指定聚类的连接方式。'complete' 表示使用“完全连接”（Complete Linkage），即在合并两个聚类时，考虑两个聚类中距离最远的点之间的距离。其他选项包括：
        'single'：单连接，考虑距离最近的点之间的距离。
        'average'：平均连接，考虑两个聚类中所有点的平均距离。
        '''
        clustering = AgglomerativeClustering(compute_full_tree =True,
                                                                    n_clusters=n_clusters,
                                                                    distance_threshold = distance_threshold,
                                                                    affinity = 'precomputed',

                                                                    #metric = 'precomputed',
                                                                    linkage='complete',
                                                                    compute_distances =True).fit(-corr+1)
        '''
        相关系数为 1（完全正相关）的点，转换后的距离为 0。
        相关系数为 -1（完全负相关）的点，转换后的距离为 2。
        相关系数为 0（无相关性）的点，转换后的距离为 1。
        '''                                                  
        clustering_labels = clustering.labels_
        k_tasks = clustering.n_clusters_
        cl_labels = [np.where(clustering_labels == k)[0].tolist() for k in range(0,k_tasks)]
        cl_labels = [[mapping[ind] for ind in cl] for cl in cl_labels]
        return cl_labels

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

        early_stop = [ 0 for _ in range(self.args.plug_num)]          

        path = os.path.join(self.args.checkpoints, setting)
        load_path = path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        # train_cluster_h = self.set_variate_clusters(train_data.data_x)
        # print("train_cluster:", train_cluster_h)
        # train_cluster_f = self.set_variate_clusters(train_data.data_y)
        # print("train_cluster:", train_cluster_f)
        

        early_stopping = EarlyStopping(args = self.args, verbose=True)
        origin_early_stopping = SocketEarlyStopping(args = self.args, verbose=True)

        model_optim , tun_optim_models = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # freeze model
        if self.args.tun_model:
            self.model.load_state_dict(torch.load(load_path))
            for param in self.model.parameters():
                param.requires_grad = False

        def gradient_hook(grad):
            """记录最后一层的梯度方向"""
            if grad is not None:
                # 获取每个通道的梯度
                grad_np = grad.detach().cpu().numpy()
                n_channels = grad_np.shape[0]
                
                # 计算通道间梯度方向的相似度矩阵
                similarity_matrix = np.zeros((n_channels, n_channels))
                for i in range(n_channels):
                    for j in range(n_channels):
                        # 归一化梯度向量
                        gi = grad_np[i] / (np.linalg.norm(grad_np[i]) + 1e-8)
                        gj = grad_np[j] / (np.linalg.norm(grad_np[j]) + 1e-8)
                        # 计算余弦相似度
                        similarity_matrix[i,j] = np.dot(gi, gj)
                
                self.gradient_recorder['directions'].append(grad_np)
                self.gradient_recorder['similarity_matrices'].append(similarity_matrix)
                self.gradient_recorder['steps'].append(len(self.gradient_recorder['steps']))
        
        # 在训练循环中注册梯度钩子
        if hasattr(self.model, 'projection'):
            self.model.projection.weight.register_hook(gradient_hook)
        else:
            self.model.layers[-1].weight.register_hook(gradient_hook)

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
                # train_cluster_h = self.set_variate_clusters(batch_x.view(-1, batch_x.shape[-1]).detach().cpu().numpy())
                # print("train_cluster_h:", train_cluster_h)

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

                    # train_cluster_f1 = self.set_variate_clusters(outputs.reshape(-1, outputs.shape[-1]).detach().cpu().numpy())
                    # print("train_cluster_f1:", train_cluster_f1) 

                    # plug calibration
                    if self.args.tun_model:
                        output_finnal = []
                        if self.args.channel_fintune:
                            # loss, channel_losses = self.calculate_channel_losses(
                            #             outputs, batch_y, self.tun_models,early_stop )
                            for j in range(self.args.c_out // self.args.cseg_len):    
                                output_finnal.append(self.tun_models[j](outputs[:, :, j*self.args.cseg_len:self.args.cseg_len*(j+1)]))                      
                            outputs = torch.cat(output_finnal, dim=2).to(self.device)        
                        else:
                            for j in range(self.args.pred_len // self.args.cseg_len):    
                                output_finnal.append(self.tun_models[j](outputs[:, j*self.args.cseg_len:self.args.cseg_len*(j+1), :]))                      
                            outputs = torch.cat(output_finnal, dim=1).to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                    else:
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                        
                # train_cluster_f2 = self.set_variate_clusters(outputs.reshape(-1, outputs.shape[-1]).detach().cpu().numpy())
                # print("train_cluster_f2:", train_cluster_f2)                    

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    if self.args.tun_model:
                        for j in range(self.args.plug_num):
                            scaler.step(tun_optim_models[i])
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    if self.args.tun_model:
                        for j in range(self.args.plug_num):
                            if early_stop[j] == 0:
                                tun_optim_models[j].step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_star_time))

            train_loss = np.average(train_loss)

            vali_average_loss ,val_loss= self.vali(vali_data, vali_loader, criterion)
            test_average_loss ,test_loss= self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, val_loss, test_loss))
            
            # tun model
            if self.args.tun_model :
                early_stop, count = early_stopping(vali_average_loss, self.tun_models, self.args, start_time, path, early_stop)
            else:
                count = origin_early_stopping(val_loss, self.model, path)

            if self.args.tun_model :
                if np.sum(count) == self.args.patience * (self.args.plug_num):
                    print("Early stopping")
                    break 
            else:
                if count == self.args.patience:
                    print("Early stopping")
                    break                            

            adjust_learning_rate(model_optim, tun_optim_models, epoch + 1, self.args)

            if (epoch + 1) % 10 == 0:  # 每10个epoch可视化一次
                self.visualize_gradient_conflicts(setting, epoch)
        
        # tun model 
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                tun_model_path = f"{path}/tun_model{i}.pth" 
                self.tun_models[i].load_state_dict(torch.load(tun_model_path))  
        else:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))  

        return self.model, self.tun_models

    def vali(self, vali_data, vali_loader, criterion):
        if self.args.tun_model:
            total_loss = [[] for _ in range(self.args.plug_num)]
        val_loss = []
        average_losses = []

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

        if self.args.tun_model:
            average_losses = [sum(sublist) / len(sublist) for sublist in total_loss]
        val_loss = np.average(val_loss)


        self.model.train()
        if self.args.tun_model:
            for i in range(self.args.plug_num):
                self.tun_models[i].train()
        return average_losses, val_loss
    
    def visualize_layer_params(self, setting):
        """可视化最后一层参数在不同输出通道上的分布"""
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        import seaborn as sns
        
        # 加载模型参数
        path = os.path.join(self.args.checkpoints, setting)
        self.model.load_state_dict(torch.load(path[:-3] + '0' + path[-2:] + '/' + 'checkpoint.pth'))
        
        # 获取最后一层参数
        if hasattr(self.model, 'projection'):
            last_layer_weights = self.model.projection.weight.detach().cpu().numpy()
        else:
            last_layer_weights = self.model.layers[-1].weight.detach().cpu().numpy()
        
        # 按输出通道拆分参数
        n_channels = last_layer_weights.shape[0]
        channel_params = [last_layer_weights[i] for i in range(n_channels)]
        
        # 计算通道间参数相似度
        similarity_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(n_channels):
                similarity = np.dot(channel_params[i], channel_params[j]) / \
                            (np.linalg.norm(channel_params[i]) * np.linalg.norm(channel_params[j]))
                similarity_matrix[i, j] = similarity
        
        # 可视化相似度矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title('Channel Parameters Direction Similarity')
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        
        # 保存结果
        save_path = f'./visualization/{setting}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(os.path.join(save_path, 'channel_direction_similarity.png'), dpi=300)
        plt.close()

    def analyze_params_similarity(self, similarity_matrix, threshold=0.8):
        """分析参数相似度矩阵"""
        n_channels = similarity_matrix.shape[0]
        
        # 找出高度相似的通道对
        high_sim_pairs = []
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                if abs(similarity_matrix[i,j]) > threshold:
                    high_sim_pairs.append((i, j, similarity_matrix[i,j]))
        
        # 找出独立的通道
        independent_channels = []
        for i in range(n_channels):
            if all(abs(similarity_matrix[i,j]) < threshold for j in range(n_channels) if i != j):
                independent_channels.append(i)
        
        # 计算平均相似度
        avg_sim = np.mean(np.abs(similarity_matrix - np.eye(n_channels)))
        
        # 打印分析结果
        print(f"\n参数相似度分析 (阈值={threshold}):")
        print(f"平均相似度: {avg_sim:.3f}")
        print(f"\n高度相似的通道对:")
        for i, j, sim in high_sim_pairs:
            print(f"通道 {i} 和 {j}: 相似度 = {sim:.3f}")
        
        print(f"\n独立通道 (与其他通道相似度都低于阈值):")
        print(independent_channels)
        
        # 计算每个通道的"独立性"得分
        independence_scores = []
        for i in range(n_channels):
            other_channels_sim = [abs(similarity_matrix[i,j]) for j in range(n_channels) if i != j]
            independence_scores.append(1 - np.mean(other_channels_sim))
        
        return {
            'avg_similarity': avg_sim,
            'high_sim_pairs': high_sim_pairs,
            'independent_channels': independent_channels,
            'independence_scores': independence_scores
        }

    def visualize_gradient_conflicts(self, setting, epoch):
        """可视化梯度冲突"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        save_path = f'./visualization/{setting}/gradient_conflicts/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # 1. 绘制最新的梯度方向相似度矩阵
        plt.figure(figsize=(10, 8))
        latest_sim_matrix = self.gradient_recorder['similarity_matrices'][-1]
        sns.heatmap(
            latest_sim_matrix,
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            cbar_kws={'label': 'Gradient Direction Similarity'}
        )
        plt.title(f'Gradient Direction Conflicts (Epoch {epoch})')
        plt.xlabel('Channel Index')
        plt.ylabel('Channel Index')
        plt.savefig(os.path.join(save_path, f'grad_conflicts_epoch_{epoch}.png'))
        plt.close()
        
        # 2. 计算并展示冲突统计
        conflict_pairs = []
        for i in range(latest_sim_matrix.shape[0]):
            for j in range(i+1, latest_sim_matrix.shape[0]):
                if latest_sim_matrix[i,j] < -0.5:  # 相似度小于-0.5认为存在冲突
                    conflict_pairs.append((i, j, latest_sim_matrix[i,j]))
        
        # 打印冲突统计信息
        print(f"\nEpoch {epoch} 梯度冲突分析:")
        print(f"发现 {len(conflict_pairs)} 对存在冲突的通道")
        if conflict_pairs:
            print("\n最严重的冲突对:")
            sorted_conflicts = sorted(conflict_pairs, key=lambda x: x[2])
            for i, j, sim in sorted_conflicts[:5]:
                print(f"通道 {i} 和 {j}: 相似度 = {sim:.3f}")

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

        self.visualize_layer_params(setting)

        return
