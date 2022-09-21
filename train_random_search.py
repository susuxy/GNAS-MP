from cgi import test
import os
import sys
import dgl
import yaml
import torch
import argparse
import numpy as np
# import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from models.model_train import *
from utils.utils import *
from tensorboardX import SummaryWriter
from utils.record_utils import record_run

import logging
# from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
import warnings
warnings.filterwarnings('ignore')

from models.model_search import Model_Search
from search import Searcher
import copy
from scipy import stats
import pickle

def geno_to_hash(geno):
    geno_list = []
    for key1 in geno:
        list1 = geno[key1]
        for each_dict in list1:
            for each_link_dict in each_dict['topology']:
                for each_key in each_link_dict:
                    geno_list.append(each_link_dict[each_key])
    return tuple(geno_list)


def uniform_score(batch_data_loader, optimizer, model, num_classes, device, max_feat_num=10000):
    desc = '=> DE-NAS scores'
    all_feats = torch.zeros((0, num_classes), dtype=torch.float32)
    with torch.no_grad():
        with tqdm(batch_data_loader, desc = desc, leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. preparing datasets
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)  # num of nodes: dim: 3935
                batch_targets = batch_targets.to(device)  
                # dim: 3935

                #! 2. optimizing model parameters
                optimizer.zero_grad()
                input        = {'G': G, 'V': V}
                batch_scores = model(input)
                # batch_scores = model.forward1(input)

                all_feats = torch.cat((all_feats, batch_scores), dim=0)

                if all_feats.shape[0] > max_feat_num:
                    break

    nas_score = torch.pdist(all_feats, p=2).mean()
    return float(nas_score.detach().cpu())


class Random_Search:
    def __init__(self, args):
        self.args = args
        self.model_searcher = Searcher(args)
        self.model = self.model_searcher.model
        self.trainer = Trainer(args)

        if os.path.exists(self.args.geno_dict_path):
            a_file = open(self.args.geno_dict_path, "rb")
            self.geno_dict = pickle.load(a_file)
        else:
            self.geno_dict = {}
    

    def generate_random(self):
        self.model.cell_arch_para = self.model.init_cell_arch_para()
        # 每次初始化随机一下architecture的edge选择参数
        arch_paras = self.model.group_arch_parameters()
        arch_topos = self.model.cell_arch_topo
        geno = genotypes(
            args       = self.args,
            arch_paras = arch_paras,
            arch_topos = arch_topos,
        )
        return geno

    
    def run(self):
        all_train = []
        all_nas = []
        for i_epoch in range(self.args.num_models):
            geno = self.generate_random()
            model_path = f'archs/random/{self.args.data}/arch/{i_epoch}.yaml'
            with open(model_path, "w") as f: 
                yaml.dump(geno, f)
            
            # generate a random model
            self.trainer.random_model(model_path)

            # generate denas scores
            self.trainer.random_parameters()  # randomly set all the paramters of the model
            nas_scores = uniform_score(self.trainer.train_queue, self.trainer.optimizer, 
            self.trainer.model, self.args.nb_classes, self.args.device)

            all_nas.append(nas_scores)

            # generate training scores
            geno_tuple = geno_to_hash(geno)
            if geno_tuple in self.geno_dict:
                train_scores = self.geno_dict[geno_tuple]
            else:
                train_scores = self.trainer.run()
                self.geno_dict[geno_tuple] = train_scores
                a_file = open(self.args.geno_dict_path, "wb")
                pickle.dump(self.geno_dict, a_file)
                a_file.close()


            all_train.append(train_scores)
            

            
            

            corr, pvalue = stats.spearmanr(all_train, all_nas)
            self.args.logger.info(f'epoch number {i_epoch}')
            self.args.logger.info(f'train scores {train_scores:.4f}')
            self.args.logger.info(f'nas scores {nas_scores:.4f}')
            self.args.logger.info(f'spearman correlation {corr:.4f}')
        corr, pvalue = stats.spearmanr(all_train, all_nas)
        return corr
        

class Trainer(object):

    def __init__(self, args):
        
        # self.args = args
        self.args = copy.deepcopy(args)
        # self.console = Console()

        # self.console.log('=> [0] Initial TensorboardX')
        self.args.logger.info('=> [0] Initial TensorboardX')
        self.writer = SummaryWriter(comment = f'Task: {args.task}, Data: {args.data}, Geno: {args.load_genotypes}')

        # self.console.log('=> [1] Initial Settings')
        self.args.logger.info('=> [1] Initial Settings')
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        # torch.cuda.manual_seed(args.seed)
        # cudnn.enabled   = True

        # # self.console.log('=> [2] Initial Models')
        # self.args.logger.info('=> [2] Initial Models')
        # self.load_model(self.args.load_genotypes)

        # self.console.log(f'=> [3] Preparing Dataset')
        self.args.logger.info(f'=> [3] Preparing Dataset')
        self.load_dataloader()

        # # self.console.log(f'=> [4] Initial Optimizers')
        # self.args.logger.info(f'=> [4] Initial Optimizers')
        # self.load_optimizer()

    def random_model(self, file_path):
        self.load_model(file_path)
        self.load_optimizer()



    def load_model(self, file_path):
        # file_path = self.args.load_genotypes
        if not os.path.isfile(file_path):
            raise Exception('Genotype file not found!')
        else:
            with open(file_path, "r") as f:
                geno      = yaml.safe_load(f)
                self.args.nb_layers = len(geno['Genotype'])
                self.args.nb_nodes  = len({ edge['dst'] for edge in geno['Genotype'][0]['topology'] })
        
        
        self.metric    = load_metric(self.args)
        self.loss_fn   = get_loss_fn(self.args).to(self.args.device)
        trans_input_fn = get_trans_input(self.args)
        self.model     = Model_Train(self.args, geno['Genotype'], trans_input_fn, self.loss_fn).to(self.args.device)
        # self.console.log(f'[red]=> Subnet Parameters: {count_parameters_in_MB(self.model)}')
        self.args.logger.info(f'[red]=> Subnet Parameters: {count_parameters_in_MB(self.model)}')


    def load_dataloader(self):
        self.dataset    = load_data(self.args)
        if self.args.pos_encode > 0:
            #! load - position encoding
            # self.console.log(f'==> [3.1] Adding positional encodings')
            self.args.logger.info(f'==> [3.1] Adding positional encodings')
            self.dataset._add_positional_encodings(self.args.pos_encode)
        self.train_data = self.dataset.train
        self.val_data   = self.dataset.val
        self.test_data  = self.dataset.test
        
        num_train = int(len(self.train_data) * self.args.data_clip)
        indices   = list(range(num_train))

        self.train_queue = torch.utils.data.DataLoader(
            dataset     = self.train_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
        )

        num_valid = int(len(self.val_data) * self.args.data_clip)
        indices   = list(range(num_valid))

        self.val_queue  = torch.utils.data.DataLoader(
            dataset     = self.val_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False
        )

        num_test  = int(len(self.test_data) * self.args.data_clip)
        indices   = list(range(num_test))

        self.test_queue = torch.utils.data.DataLoader(
            dataset     = self.test_data,
            batch_size  = self.args.batch,
            pin_memory  = True,
            sampler     = torch.utils.data.sampler.SubsetRandomSampler(indices),
            num_workers = self.args.nb_workers,
            collate_fn  = self.dataset.collate,
            shuffle     = False,
        )
    

    def load_optimizer(self):
        if self.args.optimizer == 'SGD':
            self.optimizer   = torch.optim.SGD(
                params       = self.model.parameters(),
                lr           = self.args.lr,
                momentum     = self.args.momentum,
                weight_decay = self.args.weight_decay,
            )
            
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR( 
                optimizer  = self.optimizer,
                T_max      = float(self.args.epochs),
                eta_min    = self.args.lr_min
            )

        elif self.args.optimizer == 'ADAM':
            self.optimizer   = torch.optim.Adam(
                params       = self.model.parameters(),
                lr           = self.args.lr,
                weight_decay = self.args.weight_decay,
            )

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer = self.optimizer,
                mode      = 'min',
                factor    = 0.5,
                patience  = self.args.patience,
                verbose   = True
            )
        else:
            raise Exception('Unknown optimizer!')


    def random_parameters(self):
        def initialize_params(m):
            for m_i in m.children():
                if isinstance(m_i, nn.Linear):
                    torch.nn.init.xavier_uniform(m_i.weight)
                    if m_i.bias is not None:
                        m_i.bias.data.fill_(0.01)
                initialize_params(m_i)
        initialize_params(self.model)

    def scheduler_step(self, valid_loss):

        if self.args.optimizer == 'SGD':
            self.scheduler.step()
            lr = scheduler.get_lr()[0]
        elif self.args.optimizer == 'ADAM':
            self.scheduler.step(valid_loss)
            lr = self.optimizer.param_groups[0]['lr']
            if lr < 1e-5:
                # self.console.log('=> !! learning rate is smaller than threshold !!')
                self.args.logger.info('=> !! learning rate is smaller than threshold !!')
        return lr
    

    def run(self):
        # self.console.log(f'=> [5] Train Genotypes')
        self.args.logger.info(f'=> [5] Train Genotypes')
        self.lr = self.args.lr


        best_val_result = 0
        patience = 3
        counter = 0


        for i_epoch in range(self.args.epochs):
            #! training
            train_result = self.train(i_epoch, 'train')
            # self.console.log(f"[green]=> train result [{i_epoch}] - loss: {train_result['loss']:.4f} - metric : {train_result['metric']:.4f}")
            self.args.logger.info(f"[green]=> train result [{i_epoch}] - loss: {train_result['loss']:.4f} - metric : {train_result['metric']:.4f}")
            with torch.no_grad():
                #! validating
                val_result   = self.infer(i_epoch, self.val_queue, 'val')
                # self.console.log(f"[yellow]=> valid result [{i_epoch}] - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")
                self.args.logger.info(f"[yellow]=> valid result [{i_epoch}] - loss: {val_result['loss']:.4f} - metric : {val_result['metric']:.4f}")
                

                
                # early stopping
                if val_result['metric'] > best_val_result:
                    best_val_result = val_result['metric']
                    counter = 0 
                else:
                    counter += 1
                    if counter >= patience:  # perform early stop
                        test_result  = self.infer(i_epoch, self.test_queue, 'test')
                        self.args.logger.info(f"[underline][red]=> test  result [{i_epoch}] - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")
                        self.lr = self.scheduler_step(val_result['loss'])
                        self.args.logger.info(f'=> Finished! Genotype')
                        return test_result['metric']

        
        with torch.no_grad():
            #! testing
            test_result  = self.infer(i_epoch, self.test_queue, 'test')
        # self.console.log(f"[underline][red]=> test  result [{i_epoch}] - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")
        self.args.logger.info(f"[underline][red]=> test  result [{i_epoch}] - loss: {test_result['loss']:.4f} - metric : {test_result['metric']:.4f}")
        self.lr = self.scheduler_step(val_result['loss'])
        
        # self.console.log(f'=> Finished! Genotype = {args.load_genotypes}')
        self.args.logger.info(f'=> Finished! Genotype')
        
        return test_result['metric']

    @record_run('train')
    def train(self, i_epoch, stage = 'train'):

        self.model.train()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> training'
        device       = self.args.device

        with tqdm(self.train_queue, desc = desc, leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. preparing datasets
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)  # num of nodes: dim: 3935
                batch_targets = batch_targets.to(device)  
                # dim: 3935

                #! 2. optimizing model parameters
                self.optimizer.zero_grad()
                input        = {'G': G, 'V': V}
                batch_scores = self.model(input)
                # dim: 3935 * 6
                loss         = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)
                loss.backward()
                self.optimizer.step()

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : loss_avg, 'metric' : metric_avg}
                t.set_postfix(lr = self.lr, **result)
                
        return result


    @record_run('infer')
    def infer(self, i_epoch, dataloader, stage = 'infer'):

        self.model.eval()
        epoch_loss   = 0
        epoch_metric = 0
        desc         = '=> inferring'
        device       = self.args.device

        with tqdm(dataloader, desc = desc, leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)

                batch_targets = batch_targets.to(device)
                input        = {'G': G, 'V': V}
                batch_scores = self.model(input)
                loss         = self.loss_fn(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets, graph = batch_graphs, stage = stage)

                loss_avg   = epoch_loss / (i_step + 1)
                metric_avg = epoch_metric / (i_step + 1)

                result = {'loss' : epoch_loss / (i_step + 1), 'metric' : metric_avg}
                t.set_postfix(**result)

        return result
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train_from_Genotype')
    parser.add_argument('--task',           type = str,             default = 'node_level')
    parser.add_argument('--data',           type = str,             default = 'SBM_CLUSTER')
    parser.add_argument('--extra',          type = str,             default = '')
    parser.add_argument('--in_dim_V',       type = int,             default = 7)
    parser.add_argument('--node_dim',       type = int,             default = 70)
    parser.add_argument('--nb_layers',      type = int,             default = 4)
    parser.add_argument('--nb_nodes',       type = int,             default = 2)
    parser.add_argument('--nb_classes',     type = int,             default = 6)
    parser.add_argument('--leaky_slope',    type = float,           default = 1e-2)
    parser.add_argument('--batchnorm_op',   default = False,        action = 'store_true')
    parser.add_argument('--nb_mlp_layer',   type = int,             default = 4)
    parser.add_argument('--dropout',        type = float,           default = 0.2)
    parser.add_argument('--pos_encode',     type = int,             default = 0)

    parser.add_argument('--data_clip',      type = float,           default = 1.0)
    parser.add_argument('--nb_workers',     type = int,             default = 0)
    parser.add_argument('--seed',           type = int,             default = 41)
    parser.add_argument('--epochs',         type = int,             default = 100)
    parser.add_argument('--batch',          type = int,             default = 32)
    parser.add_argument('--lr',             type = float,           default = 0.025)
    parser.add_argument('--lr_min',         type = float,           default = 0.001)
    parser.add_argument('--momentum',       type = float,           default = 0.9)
    parser.add_argument('--weight_decay',   type = float,           default = 3e-4)
    parser.add_argument('--optimizer',      type = str,             default = 'ADAM')
    parser.add_argument('--patience',       type = int,             default = 10)
    # parser.add_argument('--load_genotypes', type = str,             required = True)
    parser.add_argument('--load_genotypes', type = str, default = 'archs/folder5/SBM_CLUSTER/49.yaml')

    parser.add_argument('--portion', type=float, default=0.5)
    parser.add_argument('--unrolled',           action = 'store_true',  default = False)
    parser.add_argument('--search_mode',        type = str,             default = 'train')
    parser.add_argument('--arch_lr',            type = float,           default = 3e-4)
    parser.add_argument('--arch_weight_decay',  type = float,           default =1e-3)
    parser.add_argument('--report_freq',        type = int,             default = 1)
    parser.add_argument('--arch_save',          type = str,             default = './save_arch')

    parser.add_argument('--log_name', type=str, default='node_cluster_train.log')
    parser.add_argument('--geno_dict_path', type=str, default='geno_dict.pkl')

    parser.add_argument('--num_models', type=int, default=300)

    # console = Console()
    # args    = parser.parse_args()
    args = parser.parse_known_args()[0]

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger = logging.getLogger(__name__)
    handler = logging.FileHandler(args.log_name)
    logger.addHandler(handler)

    args.logger = logger
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(args.log_name, 'w') as f:
        pass

    title   = "[bold][red]Training from Genotype"
    vis     = '\n'.join([f"{key}: {val}" for key, val in vars(args).items()])
    vis = Syntax(vis, "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis, title = title)
    # console.print(richPanel)
    args.logger.info(richPanel)
    # Trainer(args).run()

    random_search = Random_Search(args)
    score = random_search.run()

    
    # - end - #
