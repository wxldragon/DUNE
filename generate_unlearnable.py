from torch.utils.data import DataLoader, Subset
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import sys
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
from utils import *
import pdb
from classifiers.resnet import resnet18, resnet50
from torchvision.utils import save_image
import torchvision.models as models
import torchvision.transforms

import time

start_time = time.time()


# 创建DataLoader
def create_dataloader_for_class(class_idx, batch_size):
    class_indices = [i for i, (_, label) in enumerate(cifar10_train) if label == class_idx]
    subset = Subset(cifar10_train, class_indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

# 定义触发器应用函数
def apply_trigger(imgs, trigger):
    trigger = trigger.to(imgs.device)
    poi_imgs = imgs + trigger
    poi_imgs = torch.clamp(poi_imgs, 0, 1)
    return poi_imgs

# 定义自然性限制的惩罚函数
def calculate_penalty(poisoned_image, clean_image, lambda_values, lpips_model):
    # pdb.set_trace() 
    poisoned_image_np = poisoned_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    clean_image_np = clean_image.squeeze().cpu().numpy().transpose(1, 2, 0)
    psnr_val = psnr(clean_image_np, poisoned_image_np, data_range=1)
    ssim_val = ssim(clean_image_np, poisoned_image_np, channel_axis=-1, data_range=1)
    lpips_val = lpips_model(poisoned_image, clean_image).item()
    e1 = max(0, lambda_values[0] - psnr_val)
    e2 = max(0, lambda_values[1] - ssim_val)
    e3 = max(0, lpips_val - lambda_values[2])
    penalty = e1 + e2 + e3


    # penalty = 0 # w/o the visual loss

    return penalty

 

 
def color_shift_optimization(dataloader, net, ckpt, num_class, class_idx, lambda1, lambda2, lambda3,  permutation_offset, dim=3, num_particles=10, max_iter=10):
    position = np.random.rand(num_particles, dim) * 0.2 - 0.1  # 初始化粒子的位置
    velocity = np.random.rand(num_particles, dim) * 0.02 - 0.01  # 初始化粒子的速度
    personal_best_position = np.copy(position)
    personal_best_value = np.array([float('inf')] * num_particles)
    global_best_value = float('inf')
    global_best_position = np.zeros(dim)
    for i in range(max_iter):
        for j in range(num_particles):
 
            """Loss Function to Optimize Color Shift"""           
            adv_loss, vis_loss = 0, 0
            for imgs, _ in dataloader:
                size = imgs.size(0)
                imgs = imgs.cuda()
                trigger_tensor = torch.tensor(position[j], dtype=torch.float32).cuda().view(1, 3, 1, 1)
                poi_imgs = apply_trigger(imgs, trigger_tensor)
                target_label = (class_idx + permutation_offset) % num_class
                target_labels = torch.tensor([target_label] * size, dtype=torch.long).cuda()

                for ckpt_i in ckpt:
                    net.load_state_dict(torch.load(ckpt_i)['net'])
                    net.eval()
                    adv_loss += torch.nn.functional.cross_entropy(net(poi_imgs), target_labels).item() * size
                adv_loss = adv_loss / len(ckpt)

                for poisoned_img, clean_img in zip(poi_imgs, imgs):
                    penalty = calculate_penalty(poisoned_img.unsqueeze(0), clean_img.unsqueeze(0), [lambda1, lambda2, lambda3], lpips_model)
                    vis_loss += penalty
                break       #只用第一个batch的images求the best color shift
            total_loss = adv_loss + vis_loss
            current_value = total_loss / size

            if current_value < personal_best_value[j]:
                personal_best_value[j] = current_value
                personal_best_position[j] = position[j]
            if current_value < global_best_value:
                global_best_value = current_value
                global_best_position = position[j]
        velocity = 0.5 * velocity + 0.3 * (personal_best_position - position) + 0.3 * (global_best_position - position)
        position += velocity
    return global_best_position

 
# def color_shift_optimization(  total_loss, dim=3, num_particles=10, max_iter=10):
#     position = np.random.rand(num_particles, dim) * 0.2 - 0.1  # 初始化粒子的位置
#     velocity = np.random.rand(num_particles, dim) * 0.02 - 0.01  # 初始化粒子的速度
#     personal_best_position = np.copy(position)
#     personal_best_value = np.array([float('inf')] * num_particles)
#     global_best_value = float('inf')
#     global_best_position = np.zeros(dim)
#     for i in range(max_iter):
#         for j in range(num_particles):
 
#             """Loss Function to Optimize Color Shift"""           
#              current_value = total_loss / size

#             if current_value < personal_best_value[j]:
#                 personal_best_value[j] = current_value
#                 personal_best_position[j] = position[j]
#             if current_value < global_best_value:
#                 global_best_value = current_value
#                 global_best_position = position[j]
#         velocity = 0.5 * velocity + 0.3 * (personal_best_position - position) + 0.3 * (global_best_position - position)
#         position += velocity
#     return global_best_position


def valid_test(inputs, targets, net, inputs_c, num_class, permutation_offset):
    net.eval()
    targets_attack = torch.ones(targets.shape, dtype=targets.dtype) * ((targets[0].item() + permutation_offset) % num_class)
    targets_attack = targets_attack.cuda()
    inputs_c = inputs if inputs_c is None else inputs_c.cuda()
    inputs_c.requires_grad = True
    with torch.enable_grad():
        outputs = net(inputs_c)
        loss = criterion(outputs, targets_attack)
        grad = torch.autograd.grad(loss, inputs_c)[0]
    return inputs_c.detach().cpu(), grad.detach().cpu()





def save_to_pkl(data_list, ue_output_dir, ue_name):
    file_path = os.path.join(ue_output_dir, f"{ue_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data_list, f)

 
 


 
def get_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--model', type=str, default='ResNet18')
    parser.add_argument('--lr', type=float, default=0.1)     
    
    parser.add_argument('--dataset', type=str, default='cifar10') 
    
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eps', type=int, default=8)
    
    parser.add_argument('--colorshift', action='store_true', default=False, help="only using colorshift") 
    parser.add_argument('--adv', action='store_true', default=False, help="only using adv") 
    parser.add_argument('--ours', action='store_true', default=False, help="using our scheme")     
    parser.add_argument('--lambda1', type=float, default=20) 
    parser.add_argument('--lambda2', type=float, default=0.80) 
    parser.add_argument('--lambda3', type=float, default=0.03) 
    parser.add_argument('--permutation_offset', type=int, default=3)

    parser.add_argument('--gpuid', type=str, default='0')
    parser.add_argument('--imagenet100_path', type=str, default='./data/imagenet100')

    parser.add_argument('--num_model', type=int, default=5)     # 1    3    5    7
    parser.add_argument('--alpha', type=float, default=0.5)     #0.5  1.0  1.5  2.0
    parser.add_argument('--batch', type=int, default=1000)      #500  1000 1500 2000
    parser.add_argument('--num_step', type=int, default=30)     #20   25  30  35
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return args
 
if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    ckpt_path = os.path.join("./ckpt", args.dataset, args.model)
    if not os.path.isdir(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_path}. "
            "Please place pretrained checkpoints under ckpt/<dataset>/<model>."
        )

    if args.dataset == 'cifar10':
        num_of_classes = 10
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor(),]))
        net = get_model(args.model, num_of_classes=num_of_classes, dataset=args.dataset).cuda()

        #choose ckpts    
        ckpts = [ckpt_path + "/" + x for x in os.listdir(ckpt_path) if 's2-e' in x]
        ckpts.sort(key=lambda x: int(x.split('-')[2][1:]))  #26 ckpts' paths
        if len(ckpts) < args.num_model:
            raise ValueError(
                f"Need at least {args.num_model} checkpoints in {ckpt_path}, "
                f"but found {len(ckpts)}."
            )
        interval = int(len(ckpts) / args.num_model)
        used_ckpts = []
        dev = 0
        while len(used_ckpts) != args.num_model:
            used_ckpts = ckpts[interval - dev::interval]
            dev += 1
        print(used_ckpts)
        # exit(-1)


    elif args.dataset == 'cifar100':
        num_of_classes = 100
    elif args.dataset == 'imagenet100':
        num_of_classes = 100
        net = models.resnet18(pretrained=False, num_classes=100).cuda()
        used_ckpts = [os.path.join(ckpt_path, x) for x in os.listdir(ckpt_path)]
        print(f"Totally {len(used_ckpts)} ImageNet ckpts are used.")

        imagenet100_path = args.imagenet100_path
        if not os.path.isdir(imagenet100_path):
            raise FileNotFoundError(
                f"ImageNet100 directory not found: {imagenet100_path}. "
                "Pass --imagenet100_path to point to the dataset."
            )
        classes = sorted(os.listdir(imagenet100_path), key=lambda x: int(x))
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        imagenet_train = datasets.ImageFolder(
            root=imagenet100_path,
            transform=transforms.ToTensor(),
        )
        imagenet_train.class_to_idx = class_to_idx
        imagenet_train.samples = [(p, class_to_idx[os.path.basename(os.path.dirname(p))]) 
                                 for p, _ in imagenet_train.samples]

        print("train_class_id:", imagenet_train.class_to_idx)

    else:
        raise NotImplementedError

    lpips_model = lpips.LPIPS(net='alex').cuda()

    criterion = nn.CrossEntropyLoss()
    eps, alpha = args.eps / 255, args.alpha / 255
    
 
    ue_output_dir = os.path.join('./UEs', args.dataset)
    if not os.path.exists(ue_output_dir):
        os.makedirs(ue_output_dir)
    dataset_list = []


    cnt = 0
    for class_idx in range(num_of_classes):
        if args.dataset == 'cifar10':
            dataloader = create_dataloader_for_class(class_idx, batch_size=args.batch)
        elif args.dataset == 'imagenet100':
            indices = [i for i, (_, y) in enumerate(imagenet_train.samples) if y == class_idx]
            subset = Subset(imagenet_train, indices)
            dataloader = DataLoader(subset, batch_size=args.batch, shuffle=True)            

        if args.colorshift or args.ours:   #module A or module A+B
            best_trigger = color_shift_optimization(dataloader, net, used_ckpts, num_of_classes,  class_idx, args.lambda1, args.lambda2, args.lambda3,  args.permutation_offset)
            trigger = torch.tensor(best_trigger, dtype=torch.float32).view(1, 3, 1, 1).cuda()
            print(f"Best trigger for class {class_idx}: {best_trigger}")
            
        for batch_idx, (imgs, _) in enumerate(dataloader):
            imgs = imgs.cuda()
            targets = torch.tensor([class_idx] * imgs.size(0), dtype=torch.long).cuda()
            if args.colorshift:     
                unl_samples = apply_trigger(imgs, trigger)
                save_img_tag = "colorshift"        
            elif args.adv or args.ours:
                unl_samples = None
                for i in range(args.num_step):
                    grads = []
                    for ckpt in used_ckpts:
                        net.load_state_dict(torch.load(ckpt)['net'])

                        # total_params = sum(p.numel() for p in net.parameters())
                        # print(total_params)
                        # exit(-1)

                        unl_samples, grad = valid_test(imgs, targets, net, unl_samples, num_of_classes, args.permutation_offset)
                        grads.append(grad)
                    if not i:
                        inputs_ori = unl_samples.detach().cpu().clone()
                    grad_avg = sum(grads) / len(grads)
                    perturbed_imgs = torch.min(torch.max(unl_samples - grad_avg.sign() * alpha, inputs_ori - eps), inputs_ori + eps)
 
                    unl_samples = torch.clamp(perturbed_imgs, 0.0, 1.0)                
                if args.adv:
                    save_img_tag = "ensadv"
                else:
                    unl_samples = apply_trigger(unl_samples, trigger)
                    save_img_tag = "ours"

            if args.dataset == 'imagenet100':
                ue_output_dir = os.path.join('./UEs/imagenet100', str(class_idx))
                if not os.path.exists(ue_output_dir):
                    os.makedirs(ue_output_dir)   
                for _, img_tensor in enumerate(unl_samples):
                    img = torchvision.transforms.ToPILImage()(img_tensor)    
                    img_idx = len(os.listdir(ue_output_dir))
                    img.save(os.path.join(ue_output_dir, f"{img_idx}.png"))
                    # if cnt < 100:
                    #     clean_img = torchvision.transforms.ToPILImage()(imgs[cnt])
                    #     clean_img.save(os.path.join("./1-vis-imagenet", f"{img_idx}_clean.png"))
                    #     img.save(os.path.join("./1-vis-imagenet", f"{img_idx}.png"))
                    #     cnt += 1
            else:
                # if batch_idx == 0:
                #     if class_idx == 0 or class_idx == 1 or class_idx == 3:
                #         save_image(imgs[0], "./vis-img/" + "clean_class" + str(class_idx) + "_"+ str(len(os.listdir("./vis-img"))) +".png")
                #         save_image(unl_samples[0], "./vis-img/" + save_img_tag + "_class" + str(class_idx) + "_"+ str(len(os.listdir("./vis-img"))) +".png")
                samples = unl_samples.cpu().numpy()
                dataset_list += [(samples[k], class_idx) for k in range(len(samples))]       
                print(f"Processed all images for batch {batch_idx}")       


    if args.dataset == 'cifar10':
        ue_name =  save_img_tag + "_" + str(args.num_model) +  "_" + str(args.alpha) +  "_" + str(args.batch) +  "_" + str(args.num_step)  
        save_to_pkl(dataset_list, ue_output_dir, ue_name)
        print("Successfully Dumped in Pickle File!")
    # else:
    end_time = time.time()
    print(f"The total running time is: {(end_time - start_time) / 3600:.4f} hours")
 
