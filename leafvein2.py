import os
import PIL
from torch.utils.data import Dataset
import pickle
import torchvision.transforms as transforms
import torch
import random
import numpy as np
import torchvision.transforms.functional as F

class Leafvein(Dataset):
    
    def __init__(self,
                 args,
                 crop=None,
                 hflip=None,
                 vflip=None,
                 rotate=None,
                 erase=None,
                 mode='train'):
        """
        @ image_dir:   path to directory with images
        @ label_df:    df with image id (str) and label (0/1) - only for labeled test-set
        @ transforms:  image transformation; by default no transformation
        @ sample_n:    if not None, only use that many observations
        """
        self.data_dir = args.data_dir
        #self.transform = transform
        self.mode=mode
        self.dataset=args.dataset
        with open(os.path.join(self.data_dir,self.dataset,'labels.pkl'),'rb') as df:
            self.label=pickle.load(df)
        self.img_files=os.listdir(os.path.join(self.data_dir, self.dataset, self.mode))
        

        self.data_items = {}

        # Populate the dictionary
        for img_file in self.img_files:
            img_id = int(img_file.split('.')[0])
            img_path = os.path.join(self.data_dir, self.dataset, self.mode, img_file)
            
            img = PIL.Image.open(img_path).convert('RGB')
            mask = PIL.Image.open(os.path.join(self.data_dir,self.dataset,'l2_mask',str(img_id)+'.png')) # for soybean and hainan leaf dataset

        
            img_label = self.label[img_id] - 1 
            self.data_items[img_id] = {'img': img.copy(), 'label': img_label,'mask':mask.copy()}
            
            img.close()
            mask.close()


        self.crop=crop
        self.hflip=hflip
        self.vflip=vflip
        self.erase=erase
        self.rotate=rotate
        self.mode=mode
        if mode=='train':
            self.transforms=transforms.Compose([
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            # the mean and std for leafvein dataset
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms=transforms.Compose([
            #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            transforms.ToTensor(),
            # the mean and std for leafvein dataset
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
            ])
        
        print(f'Initialized datatset with {len(self.img_files)} images.\n')
    
#    @function_timer
    def _load_images(self):
        print('loading images in memory...')
        id2image = {}
        
        for file_name in self.img_files:
            img = PIL.Image.open(os.path.join(self.data_dir,self.mode, file_name))
            img = self.transforms(img)
            id_ = file_name.split('.')[0]
            id2image[id_] = img
        
        return id2image
    
    def __getitem__(self, idx):
        file_name = self.img_files[idx]
        id_=int(file_name.split('.')[0])
        #img = PIL.Image.open(os.path.join(self.data_dir, self.dataset, self.mode,file_name)).convert('RGB')
        
        #img_id = list(self.data_items.keys())[idx]
        img_info=self.data_items[id_]

        img=img_info['img']
        label=img_info['label']
        mask=img_info['mask']


        # if self.dataset=='btf':
        #     mask = PIL.Image.open(os.path.join(self.data_dir,self.dataset,'l2_mask',file_name)) # for btf dataset
        #     label=self.label[file_name]-1 # for btf dataset  
        # else:
        #     mask = PIL.Image.open(os.path.join(self.data_dir,self.dataset,'l2_mask',id_+'.png')) # for soybean and hainan leaf dataset
        #     label=self.label[int(id_)]-1 # for soybean and hainan leaf dataset
        

        if self.hflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                
        if self.vflip:
            if random.random() < 0.5:
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        img = self.transforms(img)
        mask = torch.FloatTensor(np.array(mask).astype(np.float64))
        
        
        if self.mode=='train':    
            
            # perform random crop
            h, w = img.shape[1], img.shape[2]
            pad_tb = max(0, self.crop[0] - h)
            pad_lr = max(0, self.crop[1] - w)
            img = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(img)
            h, w = img.shape[1], img.shape[2]
            i = random.randint(0, h - self.crop[0])
            j = random.randint(0, w - self.crop[1])
            img = img[:, i:i + self.crop[0], j:j + self.crop[1]]

            mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 0)(mask)
            mask = mask[i:i + self.crop[0], j:j + self.crop[1]]
                
            
            
            noterasedimg=torch.clone(img)
            noterasedmask=torch.clone(mask)
            
            
            if self.erase:
                if random.random() < 0.5:
                    i, j, h, w, v=transforms.RandomErasing.get_params(img,scale=(0.02, 0.33), ratio=(0.3, 3.3))
                    img = F.erase(img,i,j,h,w,v)
                    mask=mask.unsqueeze(0)
                    mask=mask.repeat(3,1,1)
                    mask = F.erase(mask,i,j,h,w,v)
                    mask=mask[0,:,:]
                    mask=mask.squeeze(0)
                    
                    
        else:
            
            h, w = img.shape[1], img.shape[2]
            pad_tb = max(0, self.crop[0] - h)
            pad_lr = max(0, self.crop[1] - w)
            img = torch.nn.ZeroPad2d((0, pad_lr, 0, pad_tb))(img)
            h, w = img.shape[1], img.shape[2]
            
            i = max(0, int((h - self.crop[0])/2))
            j = max(0, int((w - self.crop[1])/2))
            
            img = img[:, i:i + self.crop[0], j:j + self.crop[1]]

            mask = torch.nn.ConstantPad2d((0, pad_lr, 0, pad_tb), 0)(mask)
            mask = mask[i:i + self.crop[0], j:j + self.crop[1]]
            
            
            noterasedimg=torch.clone(img)
            noterasedmask=torch.clone(mask)
            
                    
                    
                    
                    
        return img,label,mask#,noterasedimg, noterasedmask
    
    def __len__(self):
        return len(self.img_files)
