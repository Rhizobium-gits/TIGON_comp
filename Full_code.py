# 1.Driveに接続
from google.colab import drive
drive.mount('/content/drive')

# 2.環境構築
pip install torch==1.13.1 # Visualizationで使う
pip install scipy==1.10.1  # pytorchのpyは消す
pip install TorchDiffEqPack==1.0.1
pip install torchdiffeq==0.2.3
pip install numpy==1.23.5  # AEで使う
pip install seaborn==0.12.2
pip install matplotlib==3.5.3
pip install scanpy==1.9.3  # AEで使う
pip install utility  # Visualizationで使う

# 3.AEの読み込み(AE.ipynb)
import scanpy as sc
import sys
import os
os.chdir('/content/drive/MyDrive/TIGON/')
import AE
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import scanpy as sc
from matplotlib.pyplot import rc_context
import os
from pathlib import Path
import matplotlib.pyplot as plt
import gc
import sys
sys.path.append('Path to AE folder')
from AE import AutoEncoder, Trainer


def load_data(dataset:str,path_to_data):

    if dataset=='EMT':
        data = pd.read_csv(path_to_data+'AE_EMT_normalized.csv', index_col=0).transpose()
        y = pd.read_csv(path_to_data+'AE_EMT_time.csv', index_col=0)
        row_order = data.index
        y_reordered = y.loc[row_order]
        adata = sc.AnnData(data)
        adata.obs['time'] = y_reordered


        X=adata.X
    elif dataset=='iPSC':
        filename = path_to_data+'data.xlsx'
        df = pd.read_excel(filename, header=[0, 1], index_col=0)
        times = df.columns.get_level_values(0)
        times = times.to_list()
        df.columns = df.columns.droplevel(0)
        df = df.transpose()
        adata = sc.AnnData(df)
        adata.obs['time'] = times

        X=adata.X

    else:
        raise NotImplementedError
    return adata,X
def folder_dir(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str = 'relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         batch_size: int=32,):
    folder=Path('results/'+dataset+'_'+str(seed)+\
           '_'+str(n_latent)+'_'+str(n_layers)+'_'+str(n_hidden)+\
           '_'+str(dropout)+'_'+str(weight_decay)+'_'+str(lr)+'_'+str(batch_size)+'/')
    return folder
def generate_plots(folder,model, adata,seed,n_neighbors=10,min_dist=0.5,plots='umap'):
    model.eval()
    with torch.no_grad():
        X_latent_AE=model.get_latent_representation(torch.tensor(adata.X).type(torch.float32).to('cpu'))
    adata.obsm['X_AE']=X_latent_AE.detach().cpu().numpy()
    sc.pp.neighbors(adata, n_neighbors=n_neighbors,use_rep='X_AE')

    if dataset in ['EMT','iPSC']:
        color=['time']
    else:
        raise  NotImplementedError
    if plots=='umap':
        sc.tl.umap(adata,random_state=seed,min_dist=min_dist)
        with rc_context({'figure.figsize': (8, 8*len(color))}):
            sc.pl.umap(adata, color=color,
                       legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
        plt.savefig(str(folder) + '/umap.pdf')
        plt.close()
    elif plots=='embedding':
        with rc_context({'figure.figsize': (8*len(color), 8)}):
            sc.pl.embedding(adata, 'X_AE',color=color,
                       # legend_loc='on data',
                       legend_fontsize=12,
                       legend_fontoutline=2, )
            plt.legend(frameon=False)
            plt.xticks([plt.xlim()[0], 0., plt.xlim()[1]])
            plt.yticks([plt.ylim()[0], 0., plt.ylim()[1]])
        plt.savefig(str(folder) + '/embedding.pdf')
        plt.close()

def loss_plots(folder,model):
    fig,axs=plt.subplots(1, 1, figsize=(4, 4))
    axs.set_title('AE loss')
    axs.plot(model.history['epoch'], model.history['train_loss'])
    axs.plot(model.history['epoch'], model.history['val_loss'])
    plt.yscale('log')
    axs.legend(['train loss','val loss'])
    plt.savefig(str(folder)+'/loss.pdf')
    plt.close()

def main(dataset:str='EMT',
         seed:int=42,
         n_latent:int=6,
         n_hidden:int=300,
         n_layers: int=1,
         activation: str='relu',
         dropout:float=0.2,
         weight_decay:float=1e-4,
         lr:float=1e-3,
         max_epoch:int=500,
         batch_size: int=32,
         mode='training',
         path_to_data='Path to data'
         ):
    adata,X = load_data(dataset,path_to_data)

    model=AutoEncoder(in_dim=X.shape[1],
                      n_latent=n_latent,
                      n_hidden=n_hidden,
                      n_layers=n_layers,
                      activate_type=activation,
                      dropout=dropout,
                      norm=True,
                      seed=seed,)

    trainer=Trainer(model,X=X,
                    test_size=0.1,
                    lr=lr,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                    seed=seed)

    folder=folder_dir(dataset=dataset,
         seed=seed,
         n_latent=n_latent,
         n_hidden=n_hidden,
         n_layers=n_layers,
         dropout=dropout,
         activation=activation,
         weight_decay=weight_decay,
         lr=lr,
         batch_size=batch_size,)

    if mode=='training':
        print('training the model')
        trainer.train(max_epoch=max_epoch,patient=30)

        # model.eval()
        if not os.path.exists(folder):
            folder.mkdir(parents=True)
        torch.save({
            'func_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'loss_history':trainer.model.history,
        }, os.path.join(folder,'model.pt'))
    elif mode=='loading':
        print('loading the model')
        check_pt = torch.load(os.path.join(folder, 'model.pt'))

        model.load_state_dict(check_pt['func_state_dict'])
        trainer.optimizer.load_state_dict(check_pt['optimizer_state_dict'])
        model.history=check_pt['loss_history']
    return model,trainer,adata,folder


import importlib
# ─── 編集後は必ず dataset → 依存モジュール の順で reload ─── #
importlib.reload(AE)        # ① まず dataset.py をリロード

seed=4232
n_layers = 1
batch_size=128


dataset='EMT'

lr=1e-3
n_hidden=300
n_latent = 10


model,trainer, adata,folder=main(dataset=dataset,seed=seed,
                          n_layers=n_layers,n_latent=n_latent,n_hidden=n_hidden,
                          activation='relu',
                          lr=lr,batch_size=batch_size,
                          max_epoch=500,
                          mode='training',
                          path_to_data = 'EMT_data/')

model=model.to('cpu')

generate_plots(folder,model, adata,seed,n_neighbors=20,min_dist=0.5,plots='embedding')
loss_plots(folder,model)

# 4.Trainingの読み込み(Training.ipynb)
import utility
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('Path to utility.py')
from utility import *

args = create_args()

if __name__ == '__main__':

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:' + str(args.gpu)
                            if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(args,device)
    integral_time = args.timepoints

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]

    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)
    func.apply(initialize_weights)

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())

            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))

            if itr % 500 == 0:
                ckpt_path = os.path.join(args.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))

# 5.Visualizationの読み込み(Visualization.ipynb)
import utility
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import sys
sys.path.append('Path to utility.py')
from utility import *

args = create_args()

if __name__ == '__main__':

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda:' + str(args.gpu)
                            if torch.cuda.is_available() else 'cpu')
    # load dataset
    data_train = loaddata(args,device)
    integral_time = args.timepoints

    time_pts = range(len(data_train))
    leave_1_out = []
    train_time = [x for i,x in enumerate(time_pts) if i!=leave_1_out]

    # model
    func = UOT(in_out_dim=data_train[0].shape[1], hidden_dim=args.hidden_dim,n_hiddens=args.n_hiddens,activation=args.activation).to(device)
    func.apply(initialize_weights)

    # configure training options
    options = {}
    options.update({'method': 'Dopri5'})
    options.update({'h': None})
    options.update({'rtol': 1e-3})
    options.update({'atol': 1e-5})
    options.update({'print_neval': False})
    options.update({'neval_max': 1000000})
    options.update({'safety': None})

    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay= 0.01)
    lr_adjust = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.niters-400,args.niters-200], gamma=0.5, last_epoch=-1)
    mse = nn.MSELoss()

    LOSS = []
    L2_1 = []
    L2_2 = []
    Trans = []
    Sigma = []

    if args.save_dir is not None:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        sigma_now = 1
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            loss, loss1, sigma_now, L2_value1, L2_value2 = train_model(mse,func,args,data_train,train_time,integral_time,sigma_now,options,device,itr)

            loss.backward()
            optimizer.step()
            lr_adjust.step()

            LOSS.append(loss.item())
            Trans.append(loss1[-1].mean(0).item())
            Sigma.append(sigma_now)
            L2_1.append(L2_value1.tolist())
            L2_2.append(L2_value2.tolist())

            print('Iter: {}, loss: {:.4f}'.format(itr, loss.item()))

            if itr % 500 == 0:
                ckpt_path = os.path.join(args.save_dir, 'ckpt_itr{}.pth'.format(itr))
                torch.save({'func_state_dict': func.state_dict()}, ckpt_path)
                print('Iter {}, Stored ckpt at {}'.format(itr, ckpt_path))

    except KeyboardInterrupt:
        if args.save_dir is not None:
            ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    ckpt_path = os.path.join(args.save_dir, 'ckpt.pth')
    torch.save({
        'func_state_dict': func.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'LOSS':LOSS,
        'TRANS':Trans,
        'L2_1': L2_1,
        'L2_2': L2_2,
        'Sigma': Sigma
    }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))

## Generate the plot of regulatory matrix
# Plot avergae regulatory matrices of cells (z_t) at time (time_pt).
# Here we use first time points as an examples
time_pt = 0
z_t = data_train[time_pt]
plot_jac_v(func,z_t,time_pt,'Average_jac_d0.pdf',['UMAP1','UMAP1','UMAP1'],args,device)

## Generate the plot of gradient of growth
# Plot avergae gradients of g of cells (z_t) at time (time_pt).
# Here we use first time points as an examples
time_pt = 0
z_t = data_train[time_pt]
plot_grad_g(func,z_t,time_pt,'Average_grad_d0.pdf',['UMAP1','UMAP1','UMAP1'],args,device)

## Generate the plot of trajecotry
# Plot 3-dmensional plot of inferred trajectories of 20 cells
plot_3d(funk,data_train,train_time,integral_time,args,device)
