import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader,ConcatDataset
import torchvision.transforms as T
from PIL import Image
import piq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TIRed(Dataset):
  def __init__(self,A,B,mask=None,image_size=224,normalize=True,return_paths=False):
    self.A=A
    self.B=B
    self.mask=mask
    self.image_size=image_size
    self.return_paths=return_paths
    files=sorted(os.listdir(A))
    self.data=[]
    for f in files:
      mp=None
      if mask is not None and os.path.isdir(mask):
        mp=os.path.join(mask,f)
      self.data.append((os.path.join(A,f),os.path.join(B,f),mp))
    if normalize:
      self.transform=T.Compose([T.Resize((image_size,image_size)),T.ToTensor(),T.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
    else:
      self.transform=T.Compose([T.Resize((image_size,image_size)),T.ToTensor()])
    self.maskTransform=T.Compose([T.Resize((image_size,image_size),interpolation=Image.NEAREST),T.ToTensor()])

  def __len__(self):
    return len(self.data)

  def __getitem__(self,idx):
    AA,BB,MM=self.data[idx]
    if MM is not None:
      m=self.maskTransform(Image.open(MM).convert("L"))
    else:
      m=torch.zeros(1,self.image_size,self.image_size)
    x=self.transform(Image.open(AA).convert("RGB"))
    y=self.transform(Image.open(BB).convert("RGB"))
    if self.return_paths:
      return x,y,m,AA,BB
    return x,y,m

class Conv(nn.Module):
  def __init__(self,in_channels,out_channels,ksize=3,stride=1,padding=None,dilation=1,bn=True,act="relu"):
    super().__init__()
    if padding is None:
      padding=(ksize//2)*dilation
    self.conv=nn.Conv2d(in_channels,out_channels,ksize,stride=stride,padding=padding,dilation=dilation,bias=not bn)
    if bn:
      self.bn=nn.BatchNorm2d(out_channels)
    else:
      self.bn=None
    if act=="relu":
      self.activation=nn.ReLU(inplace=True)
    elif act=="leaky_relu":
      self.activation=nn.LeakyReLU(0.2,inplace=True)
    else:
      self.activation=None

  def forward(self,x):
    x=self.conv(x)
    if self.bn is not None:
      x=self.bn(x)
    if self.activation is not None:
      x=self.activation(x)
    return x

class ResNet(nn.Module):
  def __init__(self,channels):
    super().__init__()
    self.Fx=nn.Sequential(Conv(channels,channels,ksize=3,act="relu"),Conv(channels,channels,ksize=3,act=None))
    self.activation=nn.ReLU()

  def forward(self,x):
    y=self.Fx(x)
    return self.activation(y+x)

class Encoder(nn.Module):
  def __init__(self,in_channels,out_channels,num_convs,rates=(1,2,3),downsample=True):
    super().__init__()
    self.downsample=downsample
    self.conv1=Conv(in_channels,out_channels,ksize=3,act="leaky_relu")
    self.atrous1=Conv(out_channels,out_channels,ksize=3,dilation=rates[1],act="leaky_relu")
    if num_convs==5:
      self.atrous2=Conv(out_channels,out_channels,ksize=3,dilation=rates[2],act="leaky_relu")
    else:
      self.atrous2=None
    if num_convs>=4:
      self.conv1x1=Conv(out_channels*2,out_channels,ksize=1,act="leaky_relu")
    else:
      self.conv1x1=None
    if downsample and num_convs>=4:
      self.down_conv=Conv(out_channels,out_channels,ksize=3,stride=2,act="leaky_relu")
    else:
      self.down_conv=None

  def forward(self,x):
    y1=self.conv1(x)
    y=self.atrous1(y1)
    if self.atrous2 is not None:
      y=self.atrous2(y)
    if self.conv1x1 is not None:
      y=self.conv1x1(torch.cat([y1,y],dim=1))
    skip=y
    if self.down_conv is not None:
      y=self.down_conv(y)
    return y,skip

class Decoder(nn.Module):
  def __init__(self,in_channels,skip_channels,out_channels):
    super().__init__()
    self.upsample=nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)
    self.conv=nn.Sequential(Conv(in_channels+skip_channels,out_channels,ksize=3,act="relu"),Conv(out_channels,out_channels,ksize=3,act="relu"))

  def forward(self,x,skip):
    x=self.upsample(x)
    x=torch.cat([x,skip],dim=1)
    x=self.conv(x)
    return x

class Generator(nn.Module):
  def __init__(self,in_channels=3,out_channels=3,channels=64):
    super().__init__()
    c1=channels
    c2=channels*2
    c3=channels*4
    c4=channels*8
    c5=channels*8
    self.enc1=Encoder(in_channels,c1,num_convs=5,rates=(1,2,3),downsample=True)
    self.enc2=Encoder(c1,c2,num_convs=5,rates=(1,2,3),downsample=True)
    self.enc3=Encoder(c2,c3,num_convs=4,rates=(1,2,3),downsample=True)
    self.enc4=Encoder(c3,c4,num_convs=4,rates=(1,2,3),downsample=True)
    self.enc5=Encoder(c4,c5,num_convs=2,rates=(1,2,3),downsample=False)
    self.bottleneck=nn.Sequential(ResNet(c5),ResNet(c5),ResNet(c5),ResNet(c5),ResNet(c5),ResNet(c5))
    self.dec1=Decoder(in_channels=c5,skip_channels=c4,out_channels=c4)
    self.dec2=Decoder(in_channels=c4,skip_channels=c3,out_channels=c3)
    self.dec3=Decoder(in_channels=c3,skip_channels=c2,out_channels=c2)
    self.dec4=Decoder(in_channels=c2,skip_channels=c1,out_channels=c1)
    self.final_conv=nn.Conv2d(c1,out_channels,kernel_size=3,padding=1)

  def forward(self,x):
    y1,s1=self.enc1(x)
    y2,s2=self.enc2(y1)
    y3,s3=self.enc3(y2)
    y4,s4=self.enc4(y3)
    y5,s5=self.enc5(y4)
    y=self.bottleneck(y5)
    y=self.dec1(y,s4)
    y=self.dec2(y,s3)
    y=self.dec3(y,s2)
    y=self.dec4(y,s1)
    y=self.final_conv(y)
    return y

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark=True
imagenetMean=torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(device)
imagenetStd=torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(device)
to_pil=T.ToPILImage()

def denorm_imagenet(x):
  return x*imagenetStd+imagenetMean

@torch.no_grad()
def evaluate(loader,G,device):
  G.eval()
  sums={"psnr_in":0.0,"psnr_out":0.0,"ssim_in":0.0,"ssim_out":0.0,"fsim_in":0.0,"fsim_out":0.0,"vif_in":0.0,"vif_out":0.0}
  n_batches=0
  for b in loader:
    x=b[0]
    y=b[1]
    x=x.to(device,non_blocking=True)
    y=y.to(device,non_blocking=True)
    y_pred=G(x)
    x_dn=torch.clamp(denorm_imagenet(x),0.0,1.0)
    y_dn=torch.clamp(denorm_imagenet(y),0.0,1.0)
    y_pred_dn=torch.clamp(denorm_imagenet(y_pred),0.0,1.0)
    sums["psnr_in"]+=piq.psnr(x_dn,y_dn,data_range=1.0).item()
    sums["psnr_out"]+=piq.psnr(y_pred_dn,y_dn,data_range=1.0).item()
    sums["ssim_in"]+=piq.ssim(x_dn,y_dn,data_range=1.0).item()
    sums["ssim_out"]+=piq.ssim(y_pred_dn,y_dn,data_range=1.0).item()
    sums["fsim_in"]+=piq.fsim(x_dn,y_dn,data_range=1.0).item()
    sums["fsim_out"]+=piq.fsim(y_pred_dn,y_dn,data_range=1.0).item()
    sums["vif_in"]+=piq.vif_p(x_dn,y_dn,data_range=1.0).item()
    sums["vif_out"]+=piq.vif_p(y_pred_dn,y_dn,data_range=1.0).item()
    n_batches+=1
  if n_batches==0:
    return {k:float("nan") for k in sums}
  return {k:v/n_batches for k,v in sums.items()}

def metrics_line(name,metrics):
  return f"{name:10s} | PSNR in/out: {metrics['psnr_in']:.3f} / {metrics['psnr_out']:.3f}  | SSIM in/out: {metrics['ssim_in']:.4f} / {metrics['ssim_out']:.4f}  | FSIM in/out: {metrics['fsim_in']:.4f} / {metrics['fsim_out']:.4f}  | VIF  in/out: {metrics['vif_in']:.4f} / {metrics['vif_out']:.4f}"

@torch.no_grad()
def save_outputs(G,loaders_dict,device,out_root,per_dataset=8):
  G.eval()
  os.makedirs(out_root,exist_ok=True)
  for name,loader in loaders_dict.items():
    save_dir=os.path.join(out_root,name)
    os.makedirs(save_dir,exist_ok=True)
    saved=0
    for b in loader:
      if saved>=per_dataset:
        break
      x=b[0]
      y=b[1]
      A_paths=b[3] if len(b)>3 else None
      B_paths=b[4] if len(b)>4 else None
      x=x.to(device,non_blocking=True)
      y=y.to(device,non_blocking=True)
      y_pred=G(x)
      bs=x.size(0)
      for i in range(bs):
        if saved>=per_dataset:
          break
        if (A_paths is not None) and (B_paths is not None):
          A_raw=Image.open(A_paths[i]).convert("RGB")
          B_raw=Image.open(B_paths[i]).convert("RGB")
          Aw,Ah=A_raw.size
          Bw,Bh=B_raw.size
          x_i=np.asarray(A_raw).astype(np.float32)/255.0
          y_i=np.asarray(B_raw).astype(np.float32)/255.0
        else:
          x_i=torch.clamp(denorm_imagenet(x[i:i+1]),0.0,1.0)[0].detach().cpu().permute(1,2,0).numpy()
          y_i=torch.clamp(denorm_imagenet(y[i:i+1]),0.0,1.0)[0].detach().cpu().permute(1,2,0).numpy()
          Ah,Aw=x_i.shape[0],x_i.shape[1]
          Bh,Bw=y_i.shape[0],y_i.shape[1]
        out_t=torch.clamp(denorm_imagenet(y_pred[i:i+1]),0.0,1.0)[0].detach().cpu()
        out_pil=to_pil(out_t).resize((Bw,Bh),resample=Image.BICUBIC)
        y_pred_i=np.asarray(out_pil).astype(np.float32)/255.0
        fig,axes=plt.subplots(1,3,figsize=(9,3))
        axes[0].imshow(x_i)
        axes[0].set_title(f"Input ({Aw}x{Ah})")
        axes[0].axis("off")
        axes[1].imshow(y_i)
        axes[1].set_title(f"GT Retargeted ({Bw}x{Bh})")
        axes[1].axis("off")
        axes[2].imshow(y_pred_i)
        axes[2].set_title(f"MRGAN Output ({Bw}x{Bh})")
        axes[2].axis("off")
        fname=os.path.join(save_dir,f"{name}_{saved:05d}.png")
        fig.tight_layout()
        fig.savefig(fname,dpi=150,bbox_inches="tight")
        plt.close(fig)
        saved+=1

def load_G(G,ckpt_path,device):
  obj=torch.load(ckpt_path,map_location=device)
  if isinstance(obj,dict) and ("G_state" in obj):
    G.load_state_dict(obj["G_state"])
    return obj
  if isinstance(obj,dict) and ("state_dict" in obj):
    G.load_state_dict(obj["state_dict"])
    return obj
  G.load_state_dict(obj)
  return None

Path="/fs/scratch/PAS3162/TIReD"
batch_size=8
num_workers=4
per_dataset=8
out_root_base="/fs/scratch/PAS3162/Saraeb.1/mrgan_examples"

test_loaders={}
ava_test=TIRed(os.path.join(Path,"AVA","test","test_A"),os.path.join(Path,"AVA","test","test_B"),None,image_size=224,return_paths=True)
test_loaders["AVA"]=DataLoader(ava_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
coco_test=TIRed(os.path.join(Path,"COCO","test","test_A"),os.path.join(Path,"COCO","test","test_B"),None,image_size=224,return_paths=True)
test_loaders["COCO"]=DataLoader(coco_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
hku_test=TIRed(os.path.join(Path,"HKU-IS","test","test_A"),os.path.join(Path,"HKU-IS","test","test_B"),None,image_size=224,return_paths=True)
test_loaders["HKU-IS"]=DataLoader(hku_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)
wat_test=TIRed(os.path.join(Path,"Waterloo Exploration","test","test_A"),os.path.join(Path,"Waterloo Exploration","test","test_B"),None,image_size=224,return_paths=True)
test_loaders["Waterloo"]=DataLoader(wat_test,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

all_test_dataset=ConcatDataset([dl.dataset for dl in test_loaders.values()])
all_test_loader=DataLoader(all_test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

ckpts=[
  ("no_Lm_tv_best3","/fs/scratch/PAS3162/Saraeb.1/mrgan_no_Lm_tv_best3.pth"),
  ("tired_best","/fs/scratch/PAS3162/Saraeb.1/mrgan_tired_best.pth"),
]

print("device=",device)
print("data=",Path)
print("out=",out_root_base)

for tag,ckpt_path in ckpts:
  out_root=os.path.join(out_root_base,tag)
  os.makedirs(out_root,exist_ok=True)
  log_path=os.path.join(out_root,"run.log")
  logf=open(log_path,"w",encoding="utf-8")

  def p(s):
    print(s)
    logf.write(s+"\n")
    logf.flush()

  p("")
  p("model="+tag)
  p("ckpt="+ckpt_path)
  p("outdir="+out_root)

  G=Generator().to(device)
  meta=load_G(G,ckpt_path,device)
  G.eval()
  if isinstance(meta,dict) and ("epoch" in meta):
    p("epoch="+str(meta["epoch"]))

  p("")
  p("per-dataset:")
  for name,loader in test_loaders.items():
    m=evaluate(loader,G,device)
    p(metrics_line(name,m))

  m_all=evaluate(all_test_loader,G,device)
  p("")
  p("overall:")
  p(metrics_line("TIReD_all",m_all))

  p("")
  p("saving_examples="+str(per_dataset))
  save_outputs(G,test_loaders,device,out_root=out_root,per_dataset=per_dataset)
  p("done")

  logf.close()
