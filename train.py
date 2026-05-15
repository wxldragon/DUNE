import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
from utils_train import same_seeds, standard_loss, cifar10_training_data, add_random_gaussian_noise 
import time
import warnings
import argparse
import numpy as np
import csv
from classifiers.resnet import resnet18, resnet50
from classifiers.vgg import VGG16, VGG19
from classifiers.densenet import DenseNet121 
from io import BytesIO
from scipy.ndimage import map_coordinates
from madrys import MadrysLoss
import random
from torchvision.utils import save_image


class RandomPixelShift:
    def __init__(self, shift_range=0.1):
        self.shift_range = shift_range

    def __call__(self, img):
        if not torch.is_tensor(img):
            raise TypeError("Input should be a torch.Tensor in [C,H,W] format")
        shifts = [random.uniform(-self.shift_range, self.shift_range) for _ in range(3)]
        shift_tensor = torch.tensor(shifts, device=img.device).view(3, 1, 1)
        img_shifted = img + shift_tensor
        img_shifted = torch.clamp(img_shifted, 0.0, 1.0)
        return img_shifted

class RandomGaussianNoise:
    def __init__(self, max_std=8/255): 
        self.max_std = max_std

    def __call__(self, img): 
        if not torch.is_tensor(img):
            raise TypeError("Input must be a torch.Tensor in [C,H,W] format")

        std = random.uniform(0, self.max_std)
        noise = torch.randn_like(img) * std
        img_noisy = img + noise
        img_noisy = torch.clamp(img_noisy, 0.0, 1.0)
        return img_noisy





def per_image_channel_normalization(img):
    """
    Per-image channel normalization.
    Input: PIL Image
    Output: PIL Image
    """
    x = np.asarray(img).astype(np.float32)  # HWC, RGB

    for c in range(3):
        mean = x[:, :, c].mean()
        std = x[:, :, c].std()
        if std < 1e-6:
            std = 1.0
        x[:, :, c] = (x[:, :, c] - mean) / std
    x_min = x.min()
    x_max = x.max()
    if x_max - x_min < 1e-6:
        x = np.zeros_like(x)
    else:
        x = (x - x_min) / (x_max - x_min) * 255.0

    x = np.clip(x, 0, 255).astype(np.uint8)
    return Image.fromarray(x)


def histogram_equalization(img):
    """
    Histogram equalization on each RGB channel separately.
    Input: PIL Image
    Output: PIL Image
    """
    x = np.asarray(img).astype(np.uint8)
    out = np.empty_like(x)

    for c in range(3):
        out[:, :, c] = cv2.equalizeHist(x[:, :, c])

    return Image.fromarray(out)


def lab_whitening(img):
    """
    LAB whitening: standardize each LAB channel per image, then map back to [0,255].
    Input: PIL Image
    Output: PIL Image
    """
    x = np.asarray(img).astype(np.uint8)
    lab = cv2.cvtColor(x, cv2.COLOR_RGB2LAB).astype(np.float32)

    for c in range(3):
        mean = lab[:, :, c].mean()
        std = lab[:, :, c].std()
        if std < 1e-6:
            std = 1.0
        lab[:, :, c] = (lab[:, :, c] - mean) / std
 
    for c in range(3):
        ch = lab[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max - ch_min < 1e-6:
            lab[:, :, c] = 0
        else:
            lab[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255.0

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)


def yuv_whitening(img):
    """
    YUV whitening: standardize each YUV channel per image, then map back to [0,255].
    Input: PIL Image
    Output: PIL Image
    """
    x = np.asarray(img).astype(np.uint8)
    yuv = cv2.cvtColor(x, cv2.COLOR_RGB2YUV).astype(np.float32)

    for c in range(3):
        mean = yuv[:, :, c].mean()
        std = yuv[:, :, c].std()
        if std < 1e-6:
            std = 1.0
        yuv[:, :, c] = (yuv[:, :, c] - mean) / std
 
    for c in range(3):
        ch = yuv[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()
        if ch_max - ch_min < 1e-6:
            yuv[:, :, c] = 0
        else:
            yuv[:, :, c] = (ch - ch_min) / (ch_max - ch_min) * 255.0

    yuv = np.clip(yuv, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(rgb)





def JPEGcompression(image, rate=10):
    outputIoStream = BytesIO()
    image.save(outputIoStream, "JPEG", quality=rate, optimize=True)
    outputIoStream.seek(0)
    return Image.open(outputIoStream)

def COIN_trans(image, severity=4):
    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    alpha = [0.8, 1.2, 1.6, 2, 2.4, 2.8, 3.2][severity - 1]

    dx = (np.random.uniform(-alpha, alpha, size=shape[:2])).astype(np.float32)
    dy = (np.random.uniform(-alpha, alpha, size=shape[:2])).astype(np.float32)
 
    if len(image.shape) < 3 or image.shape[2] < 3:
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    else:
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]
        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]),np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx,(-1, 1)), np.reshape(z, (-1, 1))
    trans_img = np.clip(map_coordinates(image, indices, order=1, mode='wrap').reshape(shape), 0, 1) * 255
    return trans_img

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Training on UEs')
parser.add_argument('--lr', default=0.1, type=float, help='learning-rate')
parser.add_argument('--epochs', default=80, type=int, help='number of epoch') 
parser.add_argument('--arch', default='resnet18', type=str, help='types of training architecture')    
parser.add_argument('--batch_size', default=128, type=int)   
parser.add_argument('--seed', default=0, type=int)    
parser.add_argument('--defense', default='wo', type=str) 
parser.add_argument('--ue', default='DUNE', type=str, help='UE pickle file name without .pkl')  
parser.add_argument('--shift', default=8, type=int, help='shift_range')
parser.add_argument('--std', default=8, type=int, help='std')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args() 

 
same_seeds(args.seed)
gpu_id = int(args.gpu)   # 例如 args.gpu = "0" 或 "1"
device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
num_classes = 10


transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])
if args.defense == 'jpeg':
    print("Using JPEG compression")
    transform.transforms.append(transforms.Lambda(JPEGcompression))
elif args.defense == 'gray':
    print("Using Grayscale Transformation")
    transform.transforms.append(transforms.Grayscale(3))
elif args.defense == 'coin':
    transform.transforms.append(transforms.Lambda(lambda x: np.uint8(COIN_trans(x)))) 
elif args.defense == 'jpeggray':
    print("Using JPEG compression + Grayscale Transformation")
    transform.transforms.append(transforms.Lambda(JPEGcompression))
    transform.transforms.append(transforms.Grayscale(3))
elif args.defense == 'picn':
    print("Using per-image channel normalization")
    transform.transforms.append(transforms.Lambda(per_image_channel_normalization))

elif args.defense == 'histeq':
    print("Using histogram equalization")
    transform.transforms.append(transforms.Lambda(histogram_equalization))

elif args.defense == 'lab':
    print("Using LAB whitening")
    transform.transforms.append(transforms.Lambda(lab_whitening))

elif args.defense == 'yuv':
    print("Using YUV whitening")
    transform.transforms.append(transforms.Lambda(yuv_whitening))

transform.transforms.append(transforms.ToTensor())

 
if args.defense == 'colorGaussian':
    transform.transforms.append(RandomPixelShift(shift_range=args.shift/255))
    transform.transforms.append(RandomGaussianNoise(max_std=args.std/255))
elif args.defense == 'colorat':
    transform.transforms.append(RandomPixelShift(shift_range=args.shift/255))

ue_dir = args.ue + ".pkl"
training_data_path = os.path.join("./UEs/cifar10", ue_dir)     #poisoned training data
if not os.path.isfile(training_data_path):
    raise FileNotFoundError(
        f"UE file not found: {training_data_path}. "
        "Pass --ue with a file name under UEs/cifar10 without the .pkl suffix."
    )
train_dataset = cifar10_training_data(training_data_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=4, batch_size=args.batch_size, shuffle=True) 


# save_dir = "./debug_training_images"
# os.makedirs(save_dir, exist_ok=True)

# images, labels = next(iter(train_loader))

# mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
# std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)
# images_vis = torch.clamp(images * std + mean, 0, 1)

# save_image(images_vis[:16], os.path.join(save_dir, "grid.png"), nrow=4)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)


from torchvision import models
from torchvision.models import mobilenet_v2
from torchvision.models.vision_transformer import VisionTransformer
 
if args.arch == "resnet18":
    net = resnet18(in_dims=3, out_dims=num_classes)
elif args.arch == "resnet50":
    net = resnet50(in_dims=3, out_dims=num_classes)
elif args.arch == "vgg16":
    net = VGG16()
elif args.arch == "vgg19":
    net = VGG19()
elif args.arch == "densenet121":
    net = DenseNet121() 
elif args.arch == 'mobilenetv2':
    net = mobilenet_v2(num_classes=num_classes)
elif args.arch == "efficientnet_b0":
    net = models.efficientnet_b0(num_classes=num_classes)
elif args.arch == 'vit-tiny/4':
    net = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=6,
    num_heads=3,
    hidden_dim=192,
    mlp_dim=768,
    dropout=0.1,
    attention_dropout=0.1,
    num_classes=num_classes,
) 
elif args.arch == 'vit-small/4':
    net = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=8,
    num_heads=6,
    hidden_dim=384,
    mlp_dim=1536,
    dropout=0.1,
    attention_dropout=0.1,
    num_classes=num_classes,
)     
elif args.arch == 'vit-base/4':
    net = VisionTransformer(
    image_size=32,
    patch_size=4,
    num_layers=12,
    num_heads=12,
    hidden_dim=768,
    mlp_dim=3072,
    dropout=0.1,
    attention_dropout=0.1,
    num_classes=num_classes,
)   


net = net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)


print(f"[Arch:{args.arch}]")
for epoch in range(args.epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    net.train()
    for i, (inputs, labels) in enumerate(train_loader, 0):       
        inputs = torch.clamp(inputs, 0, 1)
        labels = labels.long()
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        if args.defense == 'at' or args.defense == 'colorat':
            outputs, loss = MadrysLoss(epsilon=args.std / 255, distance="L_inf")(
                net, inputs, labels, optimizer
            )            
        else:
            outputs = net(inputs)
            loss, _ = standard_loss(args, net, inputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        loss.backward()
        optimizer.step()

    print('[Epoch：%d/%d] loss: %.3f Train Acc: %.3f' % (epoch + 1, args.epochs, running_loss / len(train_loader), 100. * correct / total)) 
    running_loss = 0.0

    if (epoch + 1) % 5 == 0: 
        # start_time = time.time() 
        
        net.eval()
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader, 0):
            if torch.cuda.is_available():
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print('Test Acc: %.2f' % (100. * correct / total)) 


        # end_time = time.time()
        # print(f"The total running time is: {(end_time - start_time) / 3600:.4f} hours")

with open(os.path.join(f'results.csv'), 'a') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow([args.arch, args.ue, args.seed, args.shift, args.std, args.defense, 100 * correct / total])  
print('Finished Training')
