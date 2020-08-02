import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from model import Generator, Discriminator
from scipy.stats import sem, t
from scipy import mean

manualSeed = 777
random.seed(manualSeed)
torch.manual_seed(manualSeed)

dataroot = 'data/celeba'
modelroot = 'trained_model/'
resultroot = 'result/'

workers = 8
image_size = 64
batch_size = 128
nc = 3
nz = 100
ngf = 64
ndf = 64
ngpu = 2

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

netG = Generator(nz, ngf, nc).to(device)
if (device.type=='cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.load_state_dict(torch.load(modelroot + 'netG.pth'))

netD = Discriminator(ndf, nc).to(device)
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.load_state_dict(torch.load(modelroot + 'netD.pth'))

D_xs = []
D_G_zs = []

for i, data in enumerate(dataloader, 0):
    with torch.no_grad():
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        output = netD(real_cpu).view(-1)
        D_x = output.mean().item() # realを正しくrealと識別した割合
        D_xs.append(D_x)

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake = netG(noise)
        output = netD(fake.detach()).view(-1)
        D_G_z = output.mean().item() # fakeを誤ってrealと識別した割合
        D_G_zs.append(D_G_z)

acc_D = np.array(D_xs)
acc_G = np.array(D_G_zs)

m_D = mean(accD)
std_D = sem(accD)
m_G = mean(accG)
std_G = sem(accG)

s = "D(x) = %.4f +- %.4f\n D(G(z)) = %.4f +- %.4f" % (m_D, std_D, m_G, std_G)
print(s)

with open(resultroot + 'accuracy.txt', mode='w') as f:
    f.write(s)
