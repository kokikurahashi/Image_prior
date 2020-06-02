import torch
import torchvision
import pandas as pd
import torchvision.datasets as dset
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from mpl_toolkits.mplot3d import axes3d
from torchvision.datasets import MNIST
import os
import math
from Utilities.Mydatasets import Mydatasets
from models.Generator import Generator
from models.Discriminator import Discriminator
import theano
import pylab
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.nn.utils.spectral_norm import spectral_norm
from theano.tensor.shared_randomstreams import RandomStreams
beta1 = 0.5
num_epochs = 100 #エポック数
batch_size = 1 #バッチサイズ
learning_rate = 1e-4 #学習率
train =True#学習を行うかどうかのフラグ
pretrained =False#事前に学習したモデルがあるならそれを使う
save_img =True#ネットワークによる生成画像を保存するかどうかのフラグ
loss_list_g = []
loss_list_d = []
def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), x.shape[1], x.shape[2],x.shape[3])
    return x

#データセットを調整する関数
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ))])
   
#訓練用データセット
#ここのパスは自分のGoogleDriveのパスに合うように変えてください

dataset = dset.ImageFolder(root='./drive/My Drive/Image_prior_train/',
                              transform=transforms.Compose([
                              transforms.RandomResizedCrop(64, scale=(1.0, 1.0), ratio=(1., 1.)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                          ])) 

#データセットをdataoaderで読み込み
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def preserve_result_img(img,dir,filename,epoch):
  value = int(math.sqrt(batch_size))
  pic = to_img(img.cpu().data)
  pic = torchvision.utils.make_grid(pic,nrow = value)
  save_image(pic, dir+'{}'.format(int(epoch))+filename+'.png')

def model_init(net,input,output,model_path,device):
  model = net(input,output).to(device)
  if pretrained:
      param = torch.load(model_path)
      model.load_state_dict(param)
  return model

def reset_model_grad(G,D):
  G.zero_grad() 
  D.zero_grad()

def main():
    #もしGPUがあるならGPUを使用してないならCPUを使用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    G = model_init(Generator,3,3,'./drive/My Drive/result_Image_prior/image_prior_G.pth',device)
    D = model_init(Discriminator,3,64,'./drive/My Drive/result_Image_prior/image_prior_D.pth',device)
    
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    optimizerG = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
    optimizerD= torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, 0.999), weight_decay=1e-5) 
    
  
    for epoch in range(num_epochs):
        print(epoch)
        i=0
        for data in dataloader:
            real_image , _ = data
            real_image = real_image.to(device)   # 本物画像
            sample_size = real_image.size(0)  # 画像枚数
            real_target = torch.full((sample_size,1,1), random.uniform(1, 1), device=device)   # 本物ラベル
            fake_target = torch.full((sample_size,1,1), random.uniform(0, 0), device=device)   # 偽物ラベル
            
# ------------------------------------------------------------------------------------------
            reset_model_grad(G,D)
            
            fake_image =G(real_image) #生成画像
            
            output = D(fake_image) #生成画像に対するDiscriminatorの結果
            
            adversarial_loss_fake = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            loss_G = adversarial_loss_fake
            
            loss_G.backward(retain_graph = True) # 誤差逆伝播
            loss_list_g.append(loss_G)
            optimizerG.step()  # Generatorのパラメータ更新

            
#勾配情報の初期化
            reset_model_grad(G,D)

            fake_image =G(real_image) #生成画像
            
            output = D(fake_image) #生成画像に対するDiscriminatorの結果

            adversarial_nogi_loss_fake = criterion2(output,fake_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            output = D(real_image) #生成画像に対するDiscriminatorの結果

            adversarial_nogi_loss_real = criterion2(output,real_target) #Discriminatorの出力結果と正解ラベルとのBCELoss

            loss_D = adversarial_nogi_loss_real+adversarial_nogi_loss_fake
            loss_D.backward(retain_graph = True) # 誤差逆伝播
            loss_list_g.append(loss_D)
            optimizerD.step()  # Discriminatorのパラメータ更新
          

            fake_image = G(real_image) #生成画像
            
            i=i+1
            print(i, len(dataloader),loss_G,loss_D)     
    
    path = "loss_G.txt"                 
    with open(path, mode='w') as f:
      for loss in loss_list_g:
        f.write("{}".format(loss))

    path = "loss_D.txt"                 
    with open(path, mode='w') as f:
      for loss in loss_list_d:
        f.write("{}".format(loss))


    if train == True:
            #モデルを保存
            torch.save(G.state_dict(), './drive/My Drive/result_Image_prior/image_prior_G.pth')
            torch.save(D.state_dict(), './drive/My Drive/result_Image_prior/image_prior_D.pth')
                #ここのパスは自分のGoogleDriveのパスに合うように変えてください
    if save_img == True:
                preserve_result_img(fake_image,'./drive/My Drive/result_Image_prior/','fake_image',epoch)
                preserve_result_img(real_image,'./drive/My Drive/result_Image_prior/','real_image',epoch)
if __name__ == '__main__':
    main() 