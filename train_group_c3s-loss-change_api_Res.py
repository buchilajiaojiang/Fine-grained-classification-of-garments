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

from models.model_Group_change_c3s_loss_apinet_Res import Net, RegularLoss
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
            outputs = model(img, flag='val')
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

def init_crossx_params():
    # gamma1 = 0.5  # 0.35
    # gamma2 = 0.5  # 0.3
    # gamma3 = 1  # 0.35
    gamma1 = 1 # last 0.35
    gamma2 = 0.3 # last 3
    return gamma1, gamma2

if __name__ == "__main__":
    set_seed(42)
    # model_name =
    print('-' * 30)
    dataset_config = {0: "Accessorites"}
    print(dataset_config)
    dataset_name = dataset_config[0]
    print('-' * 30)
    model_name = 'GroupC3S_API'
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
    net = Net(num_classes=configs["num_classes"]).cuda()
    net = torch.nn.DataParallel(net)

    lr, momentum, decay_step, weight_decay = configs["lr"], 0.9, 160, 0.0001
    optimizer = torch.optim.SGD(lr=lr, momentum=momentum, weight_decay=weight_decay, params=net.parameters())
    # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)  # AdamW优化器

    scheduler = lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=0.1) # 300epochs 160epoch下降0.1
    
    gamma1, gamma2 = init_crossx_params()
    reg_loss_group1 = RegularLoss(gamma=gamma1, nparts=2)
    reg_loss_group2 = RegularLoss(gamma=gamma2, nparts=2)
    criterion = nn.CrossEntropyLoss()
    
    rank_criterion = nn.MarginRankingLoss(margin=0.05)
    softmax_layer = nn.Softmax(dim=1).to(device)
    now = datetime.now()

    acc_best = {"acc": 0, "epoch": 0}
    for epoch in range(0, configs["epoch"], 1):
        if epoch % 1 == 0:
            accuracy = evaluate(model=net, test_loader=test_loader)
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
            # print(f"学习率：{optimizer.param_groups[0]['lr']}")
            group1, group2, logit1_self, logit1_other, logit2_self, logit2_other = net(image)
            
            labels1 = label
            labels2 = label
            
            batch_size = logit1_self.shape[0]
            
            self_logits = torch.zeros(2 * batch_size, configs["num_classes"]).to(device)
            other_logits = torch.zeros(2 * batch_size, configs["num_classes"]).to(device)
            self_logits[:batch_size] = logit1_self # 自身得分
            self_logits[batch_size:] = logit2_self # 相似图片的自身得分
            other_logits[:batch_size] = logit1_other # 自身与相似gate交叉的分数
            other_logits[batch_size:] = logit2_other # 相似图与自身图gate交叉的分数
            
            # compute loss
            logits = torch.cat([self_logits, other_logits], dim=0)
            targets = torch.cat([labels1, labels2, labels1, labels2], dim=0)
            loss = criterion(logits, targets) # 分类损失

            self_scores = softmax_layer(self_logits)[
                torch.arange(2 * batch_size).to(device).long(),
                torch.cat([labels1, labels2], dim=0),
            ]
            other_scores = softmax_layer(other_logits)[
                torch.arange(2 * batch_size).to(device).long(),
                torch.cat([labels1, labels2], dim=0),
            ]
            flag = torch.ones(
                [
                    2 * batch_size,
                ]
            ).to(device)
            rank_loss = rank_criterion(self_scores, other_scores, flag)

            # loss = softmax_loss + rank_loss
            
            ### C3S Loss
            ## CHANGE: 
            group1 = [logit1_self, logit1_other]
            group2 = [logit2_self, logit2_other]
            loss_group1 = reg_loss_group1(group1)
            loss_group2 = reg_loss_group2(group2)
            
            
            # loss = criterion(logits, label)
            
            all_loss = loss + loss_group1 + loss_group2 + rank_loss
            all_loss.backward()
            
            optimizer.step()
            # if epoch % decay_step == 0 and epoch != 0:
            if batch_cnt % 10 == 0:
                now = datetime.now()
                print(now.strftime("%Y-%m-%d %H:%M:%S"))
                print(f"epoch = {epoch}, iteration: {batch_cnt},all loss: {all_loss.item()} label loss: {loss.item()} group1 loss: {loss_group1.item()} group2 loss: {loss_group2.item()} rank loss: {rank_loss.item()}")
        scheduler.step()
