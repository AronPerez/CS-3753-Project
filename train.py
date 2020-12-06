# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 06:14:01 2020

@author: AmongStar
"""
import os, gc
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import l5kit
import random
import time
import psutil
from tqdm import tqdm
from typing import Dict
import torchvision
from IPython.display import display
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101

#level5 toolkit
from l5kit.dataset import AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit 
from l5kit.rasterization import build_rasterizer

# deep learning
from torch import nn, optim
from torch.utils.data import DataLoader

# %%
l5kit.__version__
# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)
# %%
# --- Lyft configs ---
cfg = {
    'format_version': 5,
    'data_path': ".\\Dataset",
    'model_params': {
        'model_architecture': 'resnet50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "model_resnet50_output",
        'lr': 1e-3,
        'weight_path': ".\\Dataset\\trainDt\\model_multi_update_lyft_public.pth",
        'train': True,
        'predict': True,
        'num_modes': 3,
        'lr_reduce': 0.667,         # not used when using lr scheduler
        'lr_reduce_steps': 300_000, # not used when using lr scheduler
        'lr_scheduler': True,
        'max_lr': 0.001,
        'lr_scheduler_expect_rounds': 5
    },

    'raster_params': {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 0
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 0
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 0
        
    },
    'train_params': {
        'steps': 10000,
        'update_steps': 100,
        'checkpoint_steps': 50,
        'max_num_steps': 20000,
        'checkpoint_every_n_steps': 5000
    }
}
# %%
# Memory measurement
def memory(verbose=True):
    mem = psutil.virtual_memory()
    gb = 1024*1024*1024
    if verbose:
        print('Physical memory:',
              '%.2f GB (used),'%((mem.total - mem.available) / gb),
              '%.2f GB (available)'%((mem.available) / gb), '/',
              '%.2f GB'%(mem.total / gb))
    return (mem.total - mem.available) / gb

def gc_memory(verbose=True):
    m = gc.collect()
    if verbose:
        print('GC:', m, end=' | ')
        memory()

memory();
    
# %%
os.getcwd()
# %%
# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)
# %% [markdown]
# # ===== INIT TRAIN DATASET=====

# %%
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
print("==================================TRAIN DATA==================================")
print(train_dataset)

# %% [markdown]
# # ======INIT TEST DATASET====== 
# 

# %%
test_cfg = cfg["test_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask) #Artifically applies agents
test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
                             num_workers=test_cfg["num_workers"])
print("==================================TEST DATA==================================")
print(test_dataset)

# %% [markdown]
# # ====== INT VAL DATASET ====== 

# %%
val_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_dataset = AgentDataset(cfg, val_zarr, rasterizer)
val_dataloader = DataLoader(val_dataset,shuffle=val_cfg["shuffle"],batch_size=val_cfg["batch_size"],num_workers=val_cfg["num_workers"])
print("==================================VAL DATA==================================")
print(val_dataset)

# %% [markdown]
# # Visuals

# %%
#def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    #data = dataset[index]
    #im = data["image"].transpose(1, 2, 0)
    #im = dataset.rasterizer.to_rgb(im)
    #target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    #draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])
    
    #figsize = plt.subplots(figsize = (8,6))
   # plt.title(title, fontsize=20)
   # plt.imshow(im[::-1])
    #plt.show()


# %%
#plt.figure(figsize = (8,6))
#visualize_trajectory(train_dataset, index=90)

# %% [markdown]
# # Loss function

# %%
from torch import Tensor


# %%
def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), "confidences should sum to 1"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)




# %% [markdown]
def set_train_for_resnet(model, is_train):
    for child in model.children():
        if isinstance(child, torchvision.models.resnet.ResNet):
            for param in child.parameters():
                param.requires_grad = is_train

def check_resnet_train(model):
    is_train = []
    for child in model.children():
        if isinstance(child, torchvision.models.resnet.ResNet):
            for param in child.parameters():
                is_train.append(param.requires_grad)
    return is_train

# %%
class LyftMultiModel(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Conv2d(
            num_in_channels,
            self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=False,
        )

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features, out_features=4096),
        )

        self.num_preds = num_targets * num_modes
        self.num_modes = num_modes

        self.logit = nn.Linear(4096, out_features=self.num_preds + num_modes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.head(x)
        x = self.logit(x)
#         # Compute weighted mean velocity and rotation matrix
#         # speed (batch_size)
#         # rotation_matrix (batch_size) x (2D coords target) x (2D coords inner)
#         speed, rotation_matrix = get_weighted_avg_velocities_and_rotation(history_positions, history_avail)
#         # extrapolate using the speed in x direction
#         extrapolation_positions_th = extrapolate_position_x(speed)
        
#         # Extrapolate historical positions using weighted mean velocity
#         extrapolation_positions_th = extrapolate_positions(history_positions, history_avail)
#         # pred (batch_size)x(modes)x(time)x(2D coords)
#         # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
#         # pred = pred + extrapolation_positions_th[:, None, :, :]
#         pred0, pred12 = torch.split(pred, [1, 2], dim=1)
#         pred0 = pred0 + extrapolation_positions_th[:, None, :, :]  # only add to the first mode
#         pred = torch.cat((pred0, pred12), dim=1)
#         # rotate from inner space to target space
#         pred = torch.sum(pred[:, :, :, None, :] * rotation_matrix[:, None, None, :, :], dim=-1)
        # assert confidences.shape == (bs, self.num_modes)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences

# %%
def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences

# Now let us initialize the model and 
# load the pretrained weights. Note that since the pretrained model was trained on GPU, you also need to enable GPU when running this notebook.

# %% [markdown]
# # Init model

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LyftMultiModel(cfg)

#load weight if there is a pretrained model
#weight_path = cfg["model_params"]["weight_path"]
#if weight_path: #https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
#    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=cfg["model_params"]["lr"])
print(f'device {device}')


# %%
print(model)

# %%
w = model.logit.weight
b = model.logit.bias

b.shape

b_pred, b_confidences = torch.split(b, model.num_preds, dim=0)
b_pred, b_confidences

b_pred_v = b_pred.view(3, model.future_len, 2)
b_pred_v

b_pred_np = b_pred_v.detach().cpu().numpy().copy()

# bias on the last layer
for i in range(3):
    plt.plot(*b_pred_np[i].T, marker='.', label=f'mode = {i}', alpha=0.8)
plt.grid(); plt.legend(); plt.show()

# bias overall
for i in range(3):
    plt.plot(*b_pred_np[i].T, marker='.', label=f'mode = {i}', alpha=0.8, )
# plt.plot(*target_positions_mean.cpu().numpy().copy().T, linestyle='--', alpha=0.5, label='target mean')
plt.grid(); plt.legend(); plt.show()

# confidence bias
torch.softmax(b_confidences.detach(), dim=0).cpu().numpy()


# %% [markdown]
# # Training loop
# %%
train_dataset_total_batches = int(np.ceil(len(train_dataset) / cfg['train_data_loader']['batch_size']))
print('Number of batches in train:', train_dataset_total_batches)
print('We will only train:', cfg["train_params"]["steps"], 'batches (%.4f%%)'%(cfg["train_params"]["steps"] * 100 / train_dataset_total_batches))
# %%
# New in v11
# from collections import deque
class ReplayMemory(object):
    ''' storage class for sample reuse '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, item):
        """Saves a transition."""
        if len(self.memory) >= self.capacity:
            del self.memory[0]
        self.memory.append(item)

    def sample(self, last_item=False):
        if last_item:
            item = self.memory[-1]
        else:
            item = self.memory[np.random.randint(len(self.memory))]
        return item

    def __len__(self):
        return len(self.memory)
# %%
lr_reduce = cfg['model_params']['lr_reduce']
lr_reduce_steps = cfg['model_params']['lr_reduce_steps']
lr_reduce, lr_reduce_steps
# %%
def reduce_learning_rate(optimizer, reduce_factor, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * reduce_factor
        param_group['lr'] = new_lr
        if verbose:
            print('Reduce learning rate of group {} from {:.4e} to {:.4e}.'.format(i, old_lr, new_lr))
            
            
            
            
# %%time
if cfg["model_params"]["train"]:
    tr_it = iter(train_dataloader)
    n_steps = cfg["train_params"]["steps"]
    progress_bar = tqdm(range(1, 1 + n_steps), mininterval=5.)
    losses = []
    iterations = []
    metrics = []
    memorys = []
    times = []
    model_name = cfg["model_params"]["model_name"]
    update_steps = cfg['train_params']['update_steps']
    checkpoint_steps = cfg['train_params']['checkpoint_steps']
    t_start = time.time()
    torch.set_grad_enabled(True)
        
    for i in progress_bar:
        try:
            data = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            data = next(tr_it)
        model.train()   # somehow we need this is ever batch or it perform very bad (not sure why)
        loss, _, _ = forward(data, model, device)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_v = loss.item()
        losses.append(loss_v)
        
        if i % update_steps == 0:
            mean_losses = np.mean(losses)
            timespent = (time.time() - t_start) / 60
            print('i: %5d'%i,
                  'loss: %10.5f'%loss_v, 'loss(avg): %10.5f'%mean_losses, 
                  '%.2fmins'%timespent, end=' | ')
            mem = memory()
            if i % checkpoint_steps == 0:
                torch.save(model.state_dict(), f'{model_name}_{i}.pth')
                torch.save(optimizer.state_dict(), f'{model_name}_optimizer_{i}.pth')
            iterations.append(i)
            metrics.append(mean_losses)
            memorys.append(mem)
            times.append(timespent)

    torch.save(model.state_dict(), f'{model_name}_final.pth')
    torch.save(optimizer.state_dict(), f'{model_name}_optimizer_final.pth')
    results = pd.DataFrame({
        'iterations': iterations, 
        'metrics (avg)': metrics,
        'elapsed_time (mins)': times,
        'memory (GB)': memorys,
    })
    results.to_csv(f'train_metrics_{model_name}_{n_steps}.csv', index=False)
    print(f'Total training time is {(time.time() - t_start) / 60} mins')
    memory()
    display(results)
# %%
if cfg["model_params"]["train"]:
    plt.figure(figsize=(12, 4))
    plt.plot(results['iterations'], results['metrics (avg)'])
    plt.xlabel('steps'); plt.ylabel('metrics (avg)')
    plt.grid(); plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(results['iterations'], results['memory (GB)'])
    plt.xlabel('steps'); plt.ylabel('memory (GB)')
    plt.grid(); plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(results['iterations'], results['elapsed_time (mins)'])
    plt.xlabel('steps'); plt.ylabel('elapsed_time (mins)')
    plt.grid(); plt.show()
# %% Prediction 
print('Number of batches for predictoin:', int(np.ceil(len(test_dataset) / cfg['test_data_loader']['batch_size'])))

