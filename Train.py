#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pickle
import random
import os
import argparse

import time
from datetime import datetime

from leafvein2 import Leafvein

from unet3 import *
from backboneModels import *
from utils import *


import xgboost as xgb
import optuna


from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


#from utils import progress_bar

## reproducility
seed=1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(seed)
cudnn.benchmark = False 


# In[2]:


test_acc=[]
train_acc=[]
train_total_losses=[]
test_total_losses=[]
train_iou=[]
test_iou=[]


# In[3]:


import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Training PyTorch Models For Ultra Fine Grained')

# Add arguments
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--max_epoch', default=150, type=int, help='resume from checkpoint')
parser.add_argument('--backbone_class', type=str, default='densenet161', choices=['densenet161',  'resnet50', 'resnet34', 'resnet18', 'vgg19', 'mobilenet_v2', 'inception_v3'], help='resume from checkpoint')
parser.add_argument('--dataset', type=str, default='soybean_2_1', choices=['soybean_1_1', 'soybean_2_1', 'btf', 'hainan_leaf'], help='resume from checkpoint')
parser.add_argument('--data_dir', type=str, default='/work/MGANet/data')
parser.add_argument('--seg_size', default=448, type=int, help='Segmentation Dimension')
parser.add_argument('--num_classes', default=200, type=int, help='Number of Classes')

parser.add_argument('--dataparallel', action='store_true', help='Enable Data Parallel')
parser.add_argument('--seg_included', action='store_true', help='Enable Segmentation Training')
parser.add_argument('--cls_included', action='store_true', help='Enable Classification Training')
parser.add_argument('--freeze_all', action='store_true', help='Freeze the Encoder Module')
parser.add_argument('--mmanet', action='store_true', help='Using the MMANet')
parser.add_argument('--xmnet', action='store_true', help='Using the XMNet')
parser.add_argument('--maskguided', action='store_true', help='Using the MGANet')
parser.add_argument('--unet', action='store_true', help='Using the Unet')
parser.add_argument('--amp', action='store_true', help='using automatic mixed precision training')

parser.add_argument('--model_path', type=str, help='Use Pretrained Model')
    



# Use parse_known_args()
args, unknown = parser.parse_known_args()

if args.model_path is not None:
    args.batch_size=8
else:
    args.batch_size=32
    


# In[4]:


current_working_directory = os.getcwd()
print("Current Working Directory:", current_working_directory)

name=get_folder_path(args)
print('folder_path',name)


# Get the current date and time
current_datetime = datetime.now()

# Format the hour:minutes date and time as day-month-year 
formatted_datetime = current_datetime.strftime("%H:%M-%d-%m-%Y")

    
folder_path =os.path.join(current_working_directory,'results',args.backbone_class,name,formatted_datetime)
    
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    
    
accuracy_file_path =os.path.join(folder_path,'model_accuracies.txt')
print(accuracy_file_path)

models_folder='./checkpoint'
if not os.path.exists(models_folder):
    os.makedirs(models_folder)

    
    


# In[5]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


# In[6]:


batchsize = args.batch_size


MMANet=args.mmanet
XMNet=args.xmnet
mask_guided=args.maskguided

seg_included=args.seg_included
cls_included=args.cls_included
freeze_all=args.freeze_all

num_classes=args.num_classes
    
start_epoch=0

model_name=args.backbone_class


if args.model_path is None:
    model_path= f'./checkpoint/{name}_{formatted_datetime}.pth'
    
print(model_path)


# In[7]:


print('\nBatch_size',args.batch_size)
print('MMANet',MMANet)
print('XMNet',XMNet)
print('mask_guided',mask_guided)
print('seg_included',seg_included)
print('cls_included',cls_included)
print('freeze_all',freeze_all)  


# In[ ]:


start=time.time()
train = Leafvein(args,crop=[448,448],hflip=True,vflip=False,erase=False,mode='train')
test = Leafvein(args,crop=[448,448],mode='test')
end=time.time()
print(end-start)


# In[ ]:


trainloader = DataLoader(train, batch_size=batchsize, shuffle=True)
testloader = DataLoader(test, batch_size=batchsize, shuffle=False)


# In[ ]:


model=XMNET(backbone_name=model_name,num_classes=num_classes,MMANet=MMANet,XMNet=XMNet,mask_guided=mask_guided,seg_included=seg_included,freeze_all=freeze_all,Unet=args.unet)

if args.model_path is not None:
    model_dict = torch.load(model_path)
    state_dict = model_dict['net'] 
    model.load_state_dict(state_dict, strict=False)

net=model.to('cuda')

if args.seg_included:
    model_path= f'./checkpoint/{name}-Segmentation-{formatted_datetime}.pth'


# In[ ]:


total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters in the model: {total_params/1e+6}")


with open(accuracy_file_path, 'a') as f:
    f.write(f'\n ***************************Start******************************** \n')
    for arg in vars(args):
        f.write(f"{arg}: {getattr(args, arg)}  \n")
    f.write(f"Total parameters in the model: {total_params/1e+6}")


# In[ ]:


if device == 'cuda' and args.dataparallel:
    net = torch.nn.DataParallel(net)

    
input_size = [args.seg_size, args.seg_size]
after_firstConv_size = [size // 2 for size in input_size]


# In[ ]:


class_loss_fn = nn.CrossEntropyLoss()
seg_loss_fn   = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()


#optimizer = optim.Adam(net.parameters(), lr=0.0001)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epoch)


# In[ ]:


scaler = torch.cuda.amp.GradScaler()

def train_epoch_Seg(epoch):
    print('\nEpoch: %d' % epoch)
    with open(accuracy_file_path, 'a') as f:
        f.write(f'\n Epoch:{epoch}\n')

    
    
    net.train()
    if freeze_all and not(cls_included):
    
        for name, module in net.features.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.Dropout)):
                module.eval()
                
                
    train_loss = 0
    train_ce_loss=0
    averageIoU=0
    correct = 0
    total = 0

      
        

    for batch_idx, (inputs, targets, masks) in enumerate(trainloader):
        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

        masks=masks.unsqueeze(1)
        masks = F.interpolate(masks, size=after_firstConv_size, mode='bilinear', align_corners=False)
            
        
        optimizer.zero_grad()
     
        ce_loss_,se_loss_,mse_loss_=0,0,0
        
        if args.amp:
            with torch.autocast(device_type='cuda', dtype=torch.float16):

                if mask_guided:
                    outputs = net(inputs, masks) 
                    masks=outputs['mask']
                    fg_att=outputs['fg_att']
                    mse_loss_ = mse(fg_att,masks)

                else:
                    outputs = net(inputs)

                if seg_included:
                    Final_seg=outputs['Final_seg']
                    se_loss_ = seg_loss_fn(Final_seg,masks)

                    preds = torch.sigmoid(Final_seg)
                    preds = (preds > 0.5).float()

                    iou = iou_binary(preds, masks)
                    averageIoU+=iou


                out=outputs['out']
                ce_loss_ = class_loss_fn(out, targets)

                loss =  seg_included*se_loss_ + cls_included*ce_loss_+ mask_guided*0.1*mse_loss_

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        else:
            if mask_guided:
                outputs = net(inputs, masks) 
                masks=outputs['mask']
                fg_att=outputs['fg_att']
                mse_loss_ = mse(fg_att,masks)

            else:
                outputs = net(inputs)

            if seg_included:
                Final_seg=outputs['Final_seg']
                se_loss_ = seg_loss_fn(Final_seg,masks)

                preds = torch.sigmoid(Final_seg)
                preds = (preds > 0.5).float()

                iou = iou_binary(preds, masks)
                averageIoU+=iou


            out=outputs['out']
            ce_loss_ = class_loss_fn(out, targets)

            loss =  seg_included*se_loss_ + cls_included*ce_loss_+ mask_guided*0.1*mse_loss_

            
            
            loss.backward()
            optimizer.step()
        
        

        train_loss += loss.item()
        train_ce_loss+=ce_loss_.item()

        
        _, predicted = out.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        

        
        Final_seg,masks,inputs,targets,outputs=None,None,None,None,None
    
    train_ce_loss/=(batch_idx+1)
    train_loss/=(batch_idx+1)
        
    averageIoU=averageIoU*100/(batch_idx+1)
    accuracy=100.*correct/total
    
    print(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {train_ce_loss:.4f}| Total Loss: {train_loss:.4f}| IoU :{averageIoU:.4f}')
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Acc: {accuracy:.3f}% ({correct}/{total})| CE: {train_ce_loss:.4f}| Total Loss: {train_loss:.4f}| IoU :{averageIoU:.4f}\n')

    
    
    train_total_losses.append(train_loss)
    train_iou.append(averageIoU)
    return averageIoU,accuracy,train_ce_loss


# In[ ]:


def test_epoch_Seg(epoch):
    net.eval()
    global best_iou, best_acc
    test_loss = 0
    test_ce_loss = 0
    
    averageIoU=0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets,masks) in enumerate(testloader):
            inputs, targets,masks= inputs.to(device), targets.to(device),masks.to(device)
            
            
            masks=masks.unsqueeze(1)
            masks = F.interpolate(masks, size=after_firstConv_size, mode='bilinear', align_corners=False)
        

            ce_loss_,se_loss_,mse_loss_=0,0,0

            if mask_guided:
                outputs=net(inputs,masks)
                masks=outputs['mask']
                fg_att=outputs['fg_att']
                mse_loss_ = mse(fg_att,masks)
            else:
                outputs=net(inputs)
                

            if seg_included:
                Final_seg=outputs['Final_seg']
                se_loss_ = seg_loss_fn(Final_seg,masks)

                preds = torch.sigmoid(Final_seg)
                preds = (preds > 0.5).float()

                iou = iou_binary(preds, masks)
                averageIoU+=iou

            out=outputs['out']
            ce_loss_ = class_loss_fn(out, targets)

            loss =  seg_included*se_loss_ + cls_included*ce_loss_+ mask_guided*0.1*mse_loss_



            test_ce_loss+=ce_loss_.item()
            test_loss += loss.item()
            
            
            _, predicted = out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            
            
            
           
            segs,masks,inputs,targets,outputs=None,None,None,None,None

    averageIoU=averageIoU*100/(batch_idx+1)
    accuracy=100.*correct/total
    

    if averageIoU>best_iou:
        best_iou=averageIoU
        
    if accuracy>best_acc:
        best_acc=accuracy
    
    test_ce_loss/=(batch_idx+1)
    test_loss/=(batch_idx+1)
    
    test_total_losses.append(test_loss)
    test_iou.append(averageIoU)
      
        
    return averageIoU,accuracy,test_ce_loss


# In[ ]:


            
def check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end):    
    global min_loss, Best_test_acc_BOT, best_test_iou
    if train_ce_loss<min_loss:
        print('Saving..')
        min_loss=train_ce_loss
        Best_test_acc_BOT=test_acc
        best_test_iou=curr_test_iou
        if isinstance(net, torch.nn.DataParallel):
            net_nwrap = net.module
        else:
            net_nwrap=net
        state = {'net': net_nwrap.state_dict(), 'test_iou': curr_test_iou, 'Test_acc': test_acc, 'epoch': epoch,}
        torch.save(state, model_path)
    print(f'Time Elapsed:{end-start}\n')    
    with open(accuracy_file_path, 'a') as f:
        f.write(f'Time Elapsed:{end-start}\n')
        
    

    


# In[ ]:


best_iou=0
best_acc=0

train_best_iou=0
best_test_iou=0

min_loss=1e10
test_acc=0
Best_test_acc_BOT=0

for epoch in range(start_epoch, args.max_epoch):
    start = time.time()
    iou,train_acc,train_ce_loss= train_epoch_Seg(epoch)
    curr_test_iou,test_acc,test_ce_loss=test_epoch_Seg(epoch)
    scheduler.step()
    end = time.time()
    
    if seg_included and not (cls_included):
        check_seg_performance() 
    elif not(seg_included) and cls_included:
        check_class_performance(train_ce_loss, test_acc, curr_test_iou,net, epoch, model_path, accuracy_file_path, start, end)
        
        


# In[11]:


'all_results_file.txt'[:-4]


# In[ ]:


all_results_file =os.path.join(current_working_directory,'results',args.backbone_class,'all_results_file.txt')

from filelock import Timeout, FileLock
lock = FileLock(all_results_file[:-4]+'.lock', timeout=120)  # Timeout after two minutes
try:
    with lock:
        with open(all_results_file, 'a') as file:
            file.write(f'\n*************************************************************\n')
            file.write(f"**{name} 's Testing Accuracies**\n")
            file.write(f'***Total parameters in the model: {total_params/1e+6}***\n')            
            file.write(f'*****{formatted_datetime} Time*****\n')
            file.write(f'*************************************************************\n\n')
except Timeout:
    print("Could not acquire the lock within 120 seconds.")
    

trainloader = DataLoader(train, batch_size=16, shuffle=True)
testloader = DataLoader(test, batch_size=16, shuffle=False)
    
def objective(trial):
    global epochs,best_accuracy,lowest_loss,overall_best
    
    # Define the hyperparameter search space
    
    param = {
        'verbosity': 0,
        'objective': 'multi:softprob',
        'num_class': 200,
        'booster': 'gblinear',  # Set booster to gblinear
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.2, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1.0),
        'eta': trial.suggest_float('eta', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
    }

    train_param = {
        'early_stopping_rounds': 30,
        'verbose_eval': False
    }
    # Train the model
    X_train,y_train=training_inference()
    y_train=y_train.squeeze()
    model = xgb.XGBClassifier(**param, **train_param, random_state=1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Predict probabilities and calculate log loss on the training set
    train_predictions = model.predict_proba(X_train)
    loss = log_loss(y_train, train_predictions)

    train_predictions = model.predict(X_train)
    accuracy = accuracy_score(y_train, train_predictions)
    train_acc.append(accuracy)
    
    
    # Predict on the test set and calculate accuracy
    test_predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, test_predictions)
    test_acc.append(accuracy)
    
    if loss<lowest_loss:
        lowest_loss=loss
        best_accuracy=accuracy
        
    best_acc_bot.append(best_accuracy)
        
    if overall_best<accuracy:
        overall_best=accuracy
    print(f"Trial {trial.number}, Test Accuracy: {accuracy}, Best Acc:{best_accuracy}, Overall Best:{overall_best}")

    return loss


def training_inference():
    all_features_array_train=np.array([]).reshape(0, 2208)
    target_class_train=np.array([], dtype=int).reshape(0,1)

    for batch_idx, (inputs, targets, masks) in enumerate(trainloader):

        inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
        outputs = net(inputs)
        features=outputs['features']
        
        tensor_np = features.detach().cpu().numpy()
        all_features_array_train=np.vstack([all_features_array_train, tensor_np])

        tensor_np = targets.detach().cpu().numpy().reshape(-1,1)
        target_class_train=np.vstack([target_class_train,tensor_np])
    return all_features_array_train,target_class_train

all_features_array_test=np.array([]).reshape(0, 2208)
target_class_test=np.array([], dtype=int).reshape(0,1)



for batch_idx, (inputs, targets, masks) in enumerate(testloader):
    
    inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)
    outputs = model(inputs)
    features=outputs['features']
    
    
    tensor_np = features.detach().cpu().numpy()
    all_features_array_test=np.vstack([all_features_array_test, tensor_np])
    
    tensor_np = targets.detach().cpu().numpy().reshape(-1,1)
    target_class_test=np.vstack([target_class_test,tensor_np])
    
print(all_features_array_test.shape)
print(target_class_test.shape)



X_test, y_test = all_features_array_test, target_class_test.squeeze()

best_accuracy,overall_best=0,0
lowest_loss=10000000

train_acc=[]
test_acc=[]
best_acc_bot=[]


best_accuracy,overall_best=0,0
lowest_loss=10000000


# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=150)

# Print the best parameters
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
