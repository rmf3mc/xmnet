import torch.nn as nn
import torchvision.models as models

from torchvision.models import DenseNet161_Weights


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import TripletMarginLoss
import torchvision
import torchvision.transforms as transforms





class XMNET(nn.Module):
    def __init__ (self, backbone_name, num_classes,MMANet=True,XMNet=False,mask_guided=True,seg_included=None,freeze_all=False,Unet=True):


        super(XMNET, self).__init__()
        self.MMANet=MMANet
        self.XMNet=XMNet
        self.backbone_name=backbone_name
        self.mask_guided=mask_guided
        
        self.Unet=Unet
        self.seg_included= seg_included 
        
        
        print('self.seg_included:',self.seg_included)
        
        
        if backbone_name=='densenet161':          
            self.features=Densenet161Encoder(backbone_name, num_classes)         
            self.classifier=self.features.classifier
            self.attention=nn.Conv2d(2,1,kernel_size=1, bias=False)
            
            if freeze_all:
                self._freeze_layers(self.features.features)
                
                for param in self.features.parameters():
                    param.requires_grad = False
                
                for param in self.attention.parameters():
                    param.requires_grad = False
                    
                for param in self.classifier.parameters():
                    param.requires_grad = False
                
                
                
            else:
                self._freeze_layers(self.features.features,upto=9)
                
            
            
            
            if self.seg_included:
            
                  
                print('**********************************')
                
                
                self.Encoders=self.set_encoder_layers(self.features.features)               
                        
                no_outputs_ch=[ self.get_no_output(self.Encoders[str(i)]) if i<6 and i >1 else  self.find_latest_batchnorm(self.Encoders[str(i-1)])  for i in range(2,7)]   
                print(no_outputs_ch)
                
                
                
                shape=no_outputs_ch[-1]
    
                self.center=nn.Conv2d(shape, shape, kernel_size=3, padding=1).to('cuda')

                self.decoder_layers=nn.ModuleDict()
                
                if self.Unet:
                    for i in range(1,6):
                        self.decoder_layers[str(i)]=UNetDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=1)
                else:
                    for i in range(1,6):
                        self.decoder_layers[str(i)]=UNet3PlusDecoderLayerModule(lvl=i,no_channels=no_outputs_ch,no_classes=1)
     
     
            
    def _freeze_layers(self, model, upto=False):
        cnt, th = 0, 0
        print('freeze layers:')
        if upto:
            th = upto
            print(f'Freeze layers up to {th}s layer')
        else:
            th=float('inf')
        for name, child in model.named_children():
            cnt += 1
            if cnt < th:
                for name2, params in child.named_parameters():
                    params.requires_grad = False
                    print(name, name2, cnt,params.requires_grad)            
        
            
        

    def getAttFeats(self,att_map,features,type=None):
        features=0.5*features+0.5*(att_map*features)
        return features
    
        
        
        
    def forward(self,x,mask=None):

        if self.backbone_name[:6]=='resnet':
            features=self.features(x)
        

        elif self.backbone_name=='densenet161':
            features=self.features(x)
            
        
        #foreground attention
        outputs={}
        
        if self.MMANet:
            fg_att=torch.mean(features,dim=1).unsqueeze(1)   
            fg_att=torch.sigmoid(fg_att)  
            features=self.getAttFeats(fg_att,features)
        
        elif self.XMNet:
            fg_att=self.attention(torch.cat((torch.mean(features,dim=1).unsqueeze(1),torch.max(features,dim=1)[0].unsqueeze(1)),dim=1))
            fg_att=torch.sigmoid(fg_att)
            features=self.getAttFeats(fg_att,features)
            
            
            
        if self.mask_guided:    

            
            h,w = fg_att.shape[2],fg_att.shape[3]
            mask=F.adaptive_avg_pool2d(mask, (h, w))
            fg_att = fg_att.view(fg_att.shape[0],-1)
            mask = mask.view(mask.shape[0],-1)

            mask += 1e-12
            max_elmts=torch.max(mask,dim=1)[0].unsqueeze(1)
            mask = mask/max_elmts
            
            outputs['mask']=mask
            outputs['fg_att']=fg_att
            
        


        
        if self.backbone_name=='densenet161':
            features = F.relu(features, inplace=True)
        

        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        
        outputs['features'] = torch.flatten(out, 1)
        
        out = self.classifier(out)
            
        outputs['out']=out
        

            
        if self.seg_included:
            Encoder_outputs = self.get_encoder_ops(x)
            Encoder_5=Encoder_outputs[4]
            Conv_Encoder_5=self.center(Encoder_5)
            Final_seg=get_segmentation(self.decoder_layers,Encoder_outputs,Conv_Encoder_5)
            outputs['Final_seg']=Final_seg
            
            
        return outputs


    def get_encoder_ops(self,x):
        Encoder_outputs=[]

        for i in range(1,5):
            x = self.Encoders[str(i)](x)
            Encoder_outputs.append(x)
            
        x = self.Encoders[str(5)](x)

        mean_att=torch.mean(x,dim=1).unsqueeze(1)
        mean_att=torch.sigmoid(mean_att)  
        features=self.getAttFeats(mean_att,x)
        Encoder_outputs.append(features)

        return Encoder_outputs
                

    def set_encoder_layers(self,model):
        layers=nn.ModuleDict()
        for i in range(1,6):
            if i==1:
                layers[str(i)]=model[:3]

            elif i>1 and i<5:
                layers[str(i)]=model[(i-1)*2+1:(i-1)*2+3]

            else:
                layers[str(i)]=model[(i-1)*2+1:]
        return layers




class Densenet161Encoder(nn.Module):
    def __init__(self, backbone_name, num_classes):
        super(Densenet161Encoder, self).__init__()
        self.features=getattr(models, backbone_name)(weights=DenseNet161_Weights.DEFAULT).features
        #self.features=getattr(models, backbone_name)(pretrained=True).features
        
        self.classifier=nn.Linear(self.features[-1].num_features, num_classes)
                
    def forward(self, x):
        features = self.features(x)
        return features
    


    
    
