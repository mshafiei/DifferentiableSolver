import torch
import torch.optim.adam as adam
from cvgutils.nn.torchUtils.unet_model import UNet
import cvgutils.Image as cvgim
import cvgutils.Viz as viz
import tqdm
n_channels = 3
n_classes = 3
# img = torch.ones((1,3,100,100))
max_iters = 10000 
device = 'cuda:0'
logger = viz.logger('./logger/Unet_test','tb','Unet_test','Pytorch_unet')
model = UNet(n_channels, n_classes).to(device=device)
optim = adam.Adam(model.parameters(),eps=1e4)
fn = '/home/mohammad/Projects/optimizer/baselines/dataset/flash_no_flash/merged/Objects_002_ambient.png'
im = cvgim.imread(fn).transpose(2,0,1)[None,:,:448,:448]
im = torch.Tensor(im).to(device=device)
for i in tqdm.trange(max_iters):
    optim.zero_grad()
    pred = model(im)
    l = ((pred - im) ** 2).sum()
    l.backward()
    print(l)
    optim.step()
    with torch.no_grad():
        if(i % 10 == 0):
            imshow = torch.cat((pred,im),axis=-1)
            imshow = torch.clip(imshow,0,1)
            logger.addImage(imshow[0].permute(1,2,0),'image')
        logger.addScalar(l,'loss')
        logger.takeStep()

# train pytorch unet
# train jax unet