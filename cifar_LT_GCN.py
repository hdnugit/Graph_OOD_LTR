'''
This code is partialy based on from https://github.com/amazon-science/long-tailed-ood-detection/
'''

import sys, os, argparse, time, csv , random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as trn
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.nn import knn_graph
from torch_geometric.utils import to_undirected, degree
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #to get further deterministic Network layers

from datasets.ImbalanceCIFAR import IMBALANCECIFAR10, IMBALANCECIFAR100
from datasets.SCOODBenchmarkDataset import SCOODDataset
from datasets.tinyimages_300k import TinyImages
from models.resnet import ResNet18
from models.GCN_simple import GCN1
from utils.ltr_metrics import *



def creat_input_graph(isTrain, test_ood_loader=None):
    '''
    create KNN graph instance
    '''
    RNet.eval()
    x= torch.FloatTensor([]).to(device)
    y= torch.LongTensor([]).to(device)
    in_lbl= torch.LongTensor([]).to(device)
    in_lbl_tst= torch.LongTensor([]).to(device)
    sc_lbl= torch.LongTensor([]).to(device)
    val_lbl= torch.LongTensor([]).to(device)
    test_mask_ID= torch.BoolTensor([]).to(device)
    test_mask_OOD= torch.BoolTensor([]).to(device)
    train_mask_ID= torch.BoolTensor([]).to(device)
    train_mask_OOD= torch.BoolTensor([]).to(device)
    val_mask_ID= torch.BoolTensor([]).to(device)
    with torch.no_grad():
        if (isTrain):
            for images, targets in train_loader:
                images, targets = images.to(device), targets.to(device)
                _, features = RNet(images)
                x = torch.cat((x,features.data),0)
                y = torch.cat((y,targets),0)
                in_lbl = torch.cat((in_lbl,targets),0)
                train_mask_ID = torch.cat((train_mask_ID, torch.BoolTensor([True for i in range(images.shape[0])]).to(device)),0)
                train_mask_OOD = torch.cat((train_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_ID = torch.cat((test_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_OOD = torch.cat((test_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                val_mask_ID = torch.cat((val_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
            for images, _ in ood_loader:
                images = images.to(device)
                _, features = RNet(images)
                x = torch.cat((x,features.data),0)
                y = torch.cat((y,torch.LongTensor([-1 for i in range(images.shape[0])]).to(device)),0)
                train_mask_ID = torch.cat((train_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                train_mask_OOD = torch.cat((train_mask_OOD, torch.BoolTensor([True for i in range(images.shape[0])]).to(device)),0)
                test_mask_ID = torch.cat((test_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_OOD = torch.cat((test_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                val_mask_ID = torch.cat((val_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                _, features = RNet(images)
                x = torch.cat((x,features.data),0)
                y = torch.cat((y,targets),0)
                val_lbl = torch.cat((val_lbl,targets),0)
                train_mask_ID = torch.cat((train_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                train_mask_OOD = torch.cat((train_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_ID = torch.cat((test_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_OOD = torch.cat((test_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                val_mask_ID = torch.cat((val_mask_ID, torch.BoolTensor([True for i in range(images.shape[0])]).to(device)),0)
        else:
            for images, sc_labels in test_ood_loader:
                images, sc_labels = images.to(device), sc_labels.to(device)
                _, features = RNet(images)
                x = torch.cat((x,features.data),0)
                y = torch.cat((y,torch.LongTensor([-1 for i in range(images.shape[0])]).to(device)),0)
                sc_lbl = torch.cat((sc_lbl,sc_labels),0)
                train_mask_ID = torch.cat((train_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                train_mask_OOD = torch.cat((train_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_ID = torch.cat((test_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_OOD = torch.cat((test_mask_OOD, torch.BoolTensor([True for i in range(images.shape[0])]).to(device)),0)
                val_mask_ID = torch.cat((val_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                _, features = RNet(images)
                x = torch.cat((x,features.data),0)
                y = torch.cat((y,targets),0)
                in_lbl_tst= torch.cat((in_lbl_tst, targets), 0)
                train_mask_ID = torch.cat((train_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                train_mask_OOD = torch.cat((train_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                test_mask_ID = torch.cat((test_mask_ID, torch.BoolTensor([True for i in range(images.shape[0])]).to(device)),0)
                test_mask_OOD = torch.cat((test_mask_OOD, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)
                val_mask_ID = torch.cat((val_mask_ID, torch.BoolTensor([False for i in range(images.shape[0])]).to(device)),0)


    edge_index = knn_graph(x, k=args.k, loop=True, cosine=True, num_workers=args.num_workers)
    edge_index = to_undirected(edge_index, num_nodes=x.shape[0])   
    deg = degree(edge_index[0], x.shape[0], dtype=torch.long)
    print('degree',deg)
    node_degree_density = degree(deg, dtype=torch.long)
    print('node_degree_density',node_degree_density)
    
    raw_graph = Data(x=x, y=y, edge_index=edge_index, sc_lbl=sc_lbl, in_lbl=in_lbl, in_lbl_tst=in_lbl_tst, val_lbl=val_lbl, train_mask_ID=train_mask_ID,\
                     train_mask_OOD=train_mask_OOD, val_mask_ID=val_mask_ID, test_mask_ID=test_mask_ID,test_mask_OOD=test_mask_OOD,num_classes=num_classes, num_workers=args.num_workers)
    return raw_graph


def oe_loss_fn(logits):
    '''
    The original instable implementation. torch.logsumexp is not numerically stable.
    '''
    return -(logits.mean(1) - torch.logsumexp(logits, dim=1)).mean()
    
def train_GCN():
    training_losses, val_clean_losses = [], []
    f1s, overall_accs, many_accs, median_accs, low_accs = [], [], [], [], []
    best_overall_acc = 0
    best_metric = 0
    best_t = args.epochs - 1
    fp = open(os.path.join(save_dir, 'g_train_log.txt'), 'w')
    fp_val = open(os.path.join(save_dir, 'g_val_log.txt'), 'w')
    
    optimizer = torch.optim.Adam(gmodel.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        gmodel.train()
        current_lr = scheduler.get_last_lr()
        out , gp4 = gmodel(raw_graph_train)
        in_loss = F.cross_entropy(out[raw_graph_train.train_mask_ID],raw_graph_train.in_lbl)
        train_acc = accuracy_v2(out[raw_graph_train.train_mask_ID].argmax(dim=1),raw_graph_train.in_lbl)
        ood_loss = oe_loss_fn(out[raw_graph_train.train_mask_OOD]) 
        
        loss = in_loss + args.Lambda * ood_loss

        # backward:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # lr update:
        scheduler.step()
        
        train_str = 'epoch %d (train): acc %.4f loss %.4f (%.4f, %.4f) | lr %s' %(epoch, train_acc, loss.item(), in_loss.item(), ood_loss.item(), current_lr)
        print(train_str)
        fp.write(train_str + '\n')
        fp.flush()
        
        #validation
        gmodel.eval()
        with torch.no_grad():
            logits, _ = gmodel(raw_graph_train)
        preds = logits.argmax(dim=1)
        overall_acc = (preds[raw_graph_train.val_mask_ID] == raw_graph_train.val_lbl).float().mean().item()
        val_loss = F.cross_entropy(logits[raw_graph_train.val_mask_ID], raw_graph_train.val_lbl)
        many_acc, median_acc, low_acc, acc_per_cls = shot_acc(preds[raw_graph_train.val_mask_ID].detach().cpu().numpy(), raw_graph_train.val_lbl.detach().cpu().numpy(), img_num_per_cls, acc_per_cls=True)
        
        val_str = 'epoch %d (val): ACC %.4f (%.4f, %.4f, %.4f)' % (epoch, overall_acc, many_acc, median_acc, low_acc)
        print(val_str)
        fp_val.write(val_str + '\n')
        fp_val.flush()

        training_losses.append(loss.item())
        val_clean_losses.append(val_loss.item())
        overall_accs.append(overall_acc)
        many_accs.append(many_acc)
        median_accs.append(median_acc)
        low_accs.append(low_acc)

        plt.plot(training_losses, 'b', label='training_losses')
        plt.plot(val_clean_losses, 'g', label='val_clean_losses')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'losses_GCN_'+args.dataset+'.png'))
        plt.close()

        plt.plot(overall_accs, 'm', label='overall_accs')
        if args.imbalance_ratio < 1:
            plt.plot(many_accs, 'r', label='many_accs')
            plt.plot(median_accs, 'g', label='median_accs')
            plt.plot(low_accs, 'b', label='low_accs')
        plt.grid()
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'val_accs_GCN_'+args.dataset+'.png'))
        plt.close()

        current_metric = overall_acc
        if current_metric > best_metric:
            best_t = epoch
            best_metric = current_metric
            torch.save(gmodel.state_dict(), os.path.join(save_dir, 'best_GCN.pkl'))
            
    torch.save(gmodel.state_dict(), os.path.join(save_dir, 'latest_GCN.pkl'))
    return best_t, best_metric


def val_GCN(dout):

    if dout == 'cifar':
        if args.dataset == 'cifar10':
            dout = 'cifar100'
        elif args.dataset == 'cifar100':
            dout = 'cifar10'
    test_ood_set = SCOODDataset(os.path.join(args.data_root_path, 'SCOOD'), id_name=args.dataset, ood_name=dout, transform=test_transform) 
    test_ood_loader = DataLoader(test_ood_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                drop_last=False, pin_memory=True)
    print('Dout is %s with %d images' % (dout, len(test_ood_set)))

    raw_graph_test = creat_input_graph(isTrain=False, test_ood_loader=test_ood_loader)

    gmodel.load_state_dict(torch.load(os.path.join(save_dir, 'latest_GCN.pkl')))
    gmodel.eval()
    logits, p42 = gmodel(raw_graph_test)
    preds1 = logits.argmax(dim=1)
    preds = logits.data.max(1)[1]
    probs = F.softmax(logits, dim=1)
    msp = probs.max(dim=1).values
    scores = - msp  # The larger MSP, the smaller uncertainty

    acc = preds[raw_graph_test.test_mask_ID].eq(raw_graph_test.in_lbl_tst.data).float().mean() 
    test_acc = acc.item()
    
    in_scores = scores[raw_graph_test.test_mask_ID].detach().cpu().numpy()
    in_labels = raw_graph_test.in_lbl_tst.detach().cpu().numpy()
    in_preds = preds[raw_graph_test.test_mask_ID].detach().cpu().numpy()

    many_acc, median_acc, low_acc, acc_each_class = shot_acc(in_preds, in_labels, img_num_per_cls, acc_per_cls=True)
    head_acc = np.mean(acc_each_class[0:int(0.5*num_classes)])
    tail_acc = np.mean(acc_each_class[int(0.5*num_classes):int(num_classes)])

    clean_str = 'ACC: %.4f (%.4f, %.4f, %.4f, %.4f, %.4f)' % (test_acc, many_acc, median_acc, low_acc, head_acc, tail_acc)
    print(clean_str)

    # confidence distribution of correct samples:
    ood_scores = scores[raw_graph_test.test_mask_OOD].detach().cpu().numpy()
    sc_labels = raw_graph_test.sc_lbl.detach().cpu().numpy()

    # move some elements in ood_scores to in_scores:
    print('in_scores:', in_scores.shape)
    print('ood_scores:', ood_scores.shape)
    fake_ood_scores = ood_scores[sc_labels >= 0]
    real_ood_scores = ood_scores[sc_labels < 0]
    real_in_scores = np.concatenate([in_scores, fake_ood_scores], axis=0)
    print('fake_ood_scores:', fake_ood_scores.shape)
    print('real_in_scores:', real_in_scores.shape)
    print('real_ood_scores:', real_ood_scores.shape)

    auroc, aupr, fpr95 = get_measures(real_ood_scores, real_in_scores)

    # print:
    ood_detectoin_str = 'auroc: %.4f, aupr: %.4f, fpr95: %.4f' % (auroc, aupr, fpr95)
    print(ood_detectoin_str)

    with open(os.path.join(save_dir, 'OOD_'+args.dataset+dout+'.csv'), 'a+') as fl1:
        writer = csv.writer(fl1)  
        writer.writerow([auroc * 100, aupr * 100, fpr95 * 100]) 
    with open(os.path.join(save_dir, 'ACC_'+args.dataset+'.csv'), 'a+') as fl2:
        writer = csv.writer(fl2)
        writer.writerow([test_acc*100, many_acc*100, median_acc*100, low_acc*100, head_acc*100, tail_acc*100])


if __name__ == '__main__':
    for rm in [451, 452, 453, 454, 455, 456]:
        parser = argparse.ArgumentParser(description='Test a CIFAR Classifier')
        parser.add_argument('--gpu', default='0')
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--dataset', '--ds', default='cifar100', choices=['cifar10', 'cifar100'], help='which dataset to use')
        parser.add_argument('--imbalance_ratio', '--rho', default=0.01, type=float)
        parser.add_argument('--data_root_path', '--drp', default='/data/gpfs/projects/punim1942/my_datasets/', help='path for input datasets.')
        #
        parser.add_argument('--k', default=7, type=int, help='k for KNN graph')
        parser.add_argument('--Lambda', default=0.5, type=float, help='OE loss term tradeoff hyper-parameter')
        parser.add_argument('--test_batch_size', '--tb', type=int, default=1000)
        parser.add_argument('--epochs', default=200, type=int)
        parser.add_argument('--lr', default=0.001, type=float)

        parser.add_argument('--output_path', '--op', default='/data/scratch/projects/punim1942/Graph_OOD_LT/outputs/')
        parser.add_argument('--num_ood_samples', default=30000, type=float, help='Number of OOD samples to use.')
        parser.add_argument('--random_seed_data', default=rm, type=int, help='Generate 6 random runs.')
        args = parser.parse_args([])
        print(args)

        # Reproducing the same results
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        random.seed(args.random_seed_data)
        os.environ['PYTHONHASHSEED'] = str(args.random_seed_data)
        np.random.seed(args.random_seed_data)
        torch.manual_seed(args.random_seed_data)
        torch.cuda.manual_seed(args.random_seed_data)
        torch.cuda.manual_seed_all(args.random_seed_data)

        # intialize device:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        save_dir = os.path.join(args.output_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # data:
        mean = [0.5] * 3 
        std = [0.5] * 3 

        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.Resize((32,32)), trn.ToTensor(), trn.Normalize(mean, std)])

        if args.dataset == 'cifar10':
            num_classes = 10
            train_set = IMBALANCECIFAR10(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path, val=False)
            test_set = IMBALANCECIFAR10(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path, val=False)
            val_set = IMBALANCECIFAR10(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path, val=True)
            
        elif args.dataset == 'cifar100':
            num_classes = 100
            train_set = IMBALANCECIFAR100(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path, val=False)
            test_set = IMBALANCECIFAR100(train=False, transform=test_transform, imbalance_ratio=1, root=args.data_root_path, val=False)
            val_set = IMBALANCECIFAR100(train=True, transform=test_transform, imbalance_ratio=args.imbalance_ratio, root=args.data_root_path, val=True)
                       

        train_loader = DataLoader(train_set, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)
        
        ood_set = Subset(TinyImages(args.data_root_path, transform=test_transform), list(range(args.num_ood_samples)))
        ood_loader = DataLoader(ood_set, batch_size=args.test_batch_size, shuffle=True, num_workers=args.num_workers,
                                    drop_last=False, pin_memory=True)


        img_num_per_cls = np.array(train_set.img_num_per_cls)
        img_num_per_cls = torch.from_numpy(img_num_per_cls).to(device)
        print("train set img_num_per_cls ",img_num_per_cls)


        #load pre-trained model after (Gaussianization) trained using downsampled ImageNet
        RNet = ResNet18(num_classes=num_classes, return_features=True)
        ckpt = torch.load('./pre_trained_checkpoints/Gaussianization/cifar100fine_tune_bn_latest.pkl',  map_location=torch.device(device))
        RNet.load_state_dict(ckpt, strict=False)
        for p in RNet.parameters():
            p.requires_grad = False
        RNet = RNet.to(device)
        
        ### graph model
        gmodel = GCN1(num_node_features=512,num_classes=num_classes).to(device)
        print("no of params of GCN:",count_parameters(gmodel))
        
        ### creating train graph
        raw_graph_train = creat_input_graph(isTrain=True)

        ### training GCN model
        best_t, best_metric = train_GCN()
        print('best_metric: {}, best_t:{}.'.format(best_metric, best_t))
        
        ### testing on OOD nodes
        for dout in ['svhn', 'cifar', 'tin', 'lsun', 'texture', 'places365']:
            val_GCN(dout)
            
            
    for dout in ['svhn', 'cifar10', 'tin', 'lsun', 'texture', 'places365']:
        data = pd.read_csv(os.path.join(save_dir, 'OOD_' +args.dataset+dout + '.csv'), header=None)
            
        with open(os.path.join(save_dir, 'avg_OOD_'+args.dataset+'.csv'), 'a+') as fl1:
            writer = csv.writer(fl1)
            writer.writerow([dout, data.iloc[:,0].mean(), data.iloc[:,1].mean(), data.iloc[:,2].mean()])

    data = pd.read_csv(os.path.join(save_dir, 'avg_OOD_'+args.dataset+'.csv'), header=None)
    with open(os.path.join(save_dir, 'avg_OOD_' +args.dataset + '.csv'), 'a+') as fl3:
        writer = csv.writer(fl3)
        writer.writerow(['avg', data.iloc[:,1].mean(), data.iloc[:,2].mean(), data.iloc[:,3].mean()])

    data = pd.read_csv(os.path.join(save_dir, 'ACC_'+args.dataset+'.csv'), header=None)
    with open(os.path.join(save_dir, 'avg_ACC_'+args.dataset+'.csv'), 'a+') as fl4:
        writer = csv.writer(fl4)
        writer.writerow(['avg', data.iloc[:,0].mean(), data.iloc[:,-2].mean(), data.iloc[:,-1].mean()])






