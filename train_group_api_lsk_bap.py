import random

import torch
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import numpy as np
import torch.nn as nn
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

from models.model_group_api_lsk_bap import Net, RegularLoss
from dataset import BaseDataset


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "./"))


def get_loader(num_classes, bs, shuffle, drop_last, workers, dataset_type, dataset_name, img_size=448):
    dataset = BaseDataset(num_classes=num_classes, data_type=dataset_type, img_size=img_size, dataset_name=dataset_name)
    # train
    loader = DataLoader(dataset=dataset, num_workers=workers, batch_size=bs, shuffle=shuffle, drop_last=drop_last)
    return loader


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for img, lab in test_loader:
            img, lab = img.cuda(), lab.cuda()
            _, outputs = model(img, flag='val')
            # sum_outputs = torch.sum(torch.stack(outputs), dim=0)
            _, predicted = torch.max(outputs.data, 1)
            total += lab.size(0)
            # print(f"label.shape:{lab.shape}, logits.shape={logits.shape}")
            correct += (predicted == lab).sum().item()
            # print(f"total = {total}, correct = {correct}")
    acc = 100 * correct / total
    return acc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def calculate_pooling_center_loss(features, centers, label, alfa=0.95):
    # centers = model.centers
    # print('111111111',sum(sum(centers)))
    # mse_loss = torch.nn.MSELoss()
    features = features.reshape(features.shape[0], -1)
    # print(features.shape)
    centers_batch = centers[label]
    # print(centers_batch)
    # print(centers_batch.shape,centers.shape)
    centers_batch = torch.nn.functional.normalize(centers_batch, dim=-1)
    diff =  (1-alfa)*(features.detach() - centers_batch)
    distance = torch.pow(features - centers_batch,2)
    distance = torch.sum(distance, dim=-1)
    center_loss = torch.mean(distance)
    # loss2 = mse_loss(features,centers_batch)
    # print('================',center_loss.item(),loss2.item())
    return center_loss, diff

def init_crossx_params():
    # gamma1 = 0.5  # 0.35
    # gamma2 = 0.5  # 0.3
    # gamma3 = 1  # 0.35
    gamma1 = 1 # last 0.35
    gamma2 = 0.3 # last 0.3  优先降值
    return gamma1, gamma2

if __name__ == "__main__":
    set_seed(42)
    # model_name =
    print('-' * 30)
    dataset_config = {0: "Accessorites"}
    print(dataset_config)
    dataset_name = dataset_config[0]
    print('-' * 30)
    model_name = 'GroupC3S_API_change_4'
    print(model_name)
    print('-' * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    configs = {
        "num_classes": 13, "img_size": 448, "lr": 1e-3, "epoch": 300,
        "train": {"workers": 3, "bs": 16, "shuffle": True, "drop_last": False},
        "test": {"workers": 0, "bs": 16, "shuffle": False, "drop_last": False},
        "valid": {"workers": 0, "bs": 16, "shuffle": False, "drop_last": False}
    }

    train_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["train"]["bs"],
        shuffle=configs["train"]["shuffle"],
        drop_last=configs["train"]["drop_last"],
        workers=configs["train"]["workers"],
        dataset_type="train",
        img_size=configs["img_size"],
        dataset_name=dataset_name
    )

    test_loader = get_loader(
        num_classes=configs["num_classes"],
        bs=configs["test"]["bs"],
        shuffle=configs["test"]["shuffle"],
        drop_last=configs["test"]["drop_last"],
        workers=configs["test"]["workers"],
        dataset_type="test",
        img_size=configs["img_size"],
        dataset_name=dataset_name
    )

    # 3. Resnet101
    net = Net(num_classes=configs["num_classes"], num_groups=4).cuda()
    net = torch.nn.DataParallel(net)

    lr, momentum, decay_step, weight_decay = configs["lr"], 0.9, 10, 0.0001
    optimizer = torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, params=net.parameters())
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW优化器

    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.1) # 300epochs 160epoch下降0.1
    # lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=0.1, last_epoch=-1)
    # scheduler_fc = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, configs["epoch"] * len(train_loader))
    
    nparts = 4
    
    gamma1, gamma2 = init_crossx_params()
    # reg_loss_group1 = RegularLoss(gamma=gamma1, nparts=2)
    reg_loss_group2 = RegularLoss(gamma=gamma2, nparts=nparts)
    criterion = nn.CrossEntropyLoss()
    
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(device)
    now = datetime.now()
    
    center = torch.zeros(13, 2048 * 32).cuda()
    acc_best = {"acc": 0, "epoch": 0}
    writer = SummaryWriter(log_dir='./logs/encoder_sample2')
    
    for epoch in range(0, configs["epoch"], 1):
        running_loss = 0.0
        correct = 0
        total = 0
        # 验证
        if epoch % 1 == 0:
            accuracy = evaluate(model=net, test_loader=test_loader)
            print(" ---------------- Test ACC ----------------")
            print(f"epoch:{epoch} The accuracy is:{accuracy}")
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            if accuracy > acc_best["acc"]:
                acc_best["acc"] = accuracy
                acc_best["epoch"] = epoch
                model_path = os.path.join(
                    root_dir, "save", "model", model_name,
                    "epoch_" + str(epoch) + "_acc_" + str(round(accuracy, 3)) + ".pth")
                torch.save(net.state_dict(), model_path)
            print(" ---------------- ACC ----------------")
            print(f"The best is:{acc_best}, appear in epoch:{acc_best['epoch']}")

        net.train(True)
        current_lr = optimizer.param_groups[0]['lr']
        print("Epoch: ", epoch, "Current learning rate: ", current_lr)
        for batch_cnt, batch in enumerate(train_loader):
            image, label = batch[0].cuda(), torch.LongTensor(batch[1]).cuda()
            optimizer.zero_grad()
            raw_features, logit_all  = net(image) # len nparts*nparts
            
            labels1 = label
            labels2 = label
            
            batch_size = logit_all[0].shape[0]
            
            loss = criterion(logit_all, label)
            
            feature_center_loss_all = 0
            features = raw_features.reshape(raw_features.shape[0], -1)

            feature_center_loss, center_diff = calculate_pooling_center_loss(
                    features, center, label, alfa=0.95
                )
            feature_center_loss_all += feature_center_loss
            
            center[label] += center_diff
            
            
            all_loss = loss + feature_center_loss_all
            all_loss.backward()
            
            optimizer.step()
            # if epoch % decay_step == 0 and epoch != 0:
            
            running_loss += loss.item()
            _, predicted = torch.max(logit_all.data, 1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            if batch_cnt % 100 == 0:
                step = epoch * len(train_loader) + batch_cnt
                writer.add_scalar('Loss/train', running_loss / (batch_cnt + 1), step)
                writer.add_scalar('Accuracy/train', 100 * correct / total, step)
                
                
            if batch_cnt % 10 == 0:
                now = datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"epoch = {epoch}, iteration: {batch_cnt},all loss: {all_loss.item()} label loss: {loss.item()} feature_center_loss loss: {feature_center_loss_all.item()}")
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        print(f"Train Epoch {epoch}: Accuracy: {100. * train_acc:.2f}% Train Loss: {train_loss:.6f}")
    
        scheduler.step()
    writer.close()
