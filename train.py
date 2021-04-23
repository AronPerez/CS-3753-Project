from torchsummary import summary
import copy
from torch import Tensor
import os, gc
import zarr
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import l5kit
import psutil
import os
import random
import time
import pprint
from tqdm import tqdm
import tqdm.notebook as tq
from typing import Dict
from collections import Counter
from prettytable import PrettyTable
import warnings
warnings.filterwarnings("ignore")
from matplotlib import animation, rc
from IPython.display import HTML, display

rc('animation', html='jshtml')
import matplotlib.patches as mpatches


#level5 toolkit
from l5kit.data import labels
from l5kit.dataset import EgoDataset, AgentDataset
from l5kit.data import ChunkedDataset, LocalDataManager

# level5 toolkit 
from l5kit.configs import load_config_data
from l5kit.geometry import transform_points
from l5kit.rasterization import build_rasterizer
from l5kit.visualization import draw_trajectory, draw_reference_trajectory, TARGET_POINTS_COLOR
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation import *
from l5kit.kinematic import AckermanPerturbation
from l5kit.random import GaussianRandomGenerator

# visualization
from matplotlib import animation
from colorama import Fore, Back, Style

# deep learning
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet18, resnet50, resnet34


print("We've completed loading the modules")

"""
Set the seed
"""
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)

"""
Set out config paths
"""

DIR_INPUT = "/home-new/tle728/dataset/"
cfg = load_config_data(
    "/home-new/tle728/dataset/config.yaml")
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)

"""
LOAD TRAIN DATA into DATALOADER
"""
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)

#perturb_prob = cfg["train_data_loader"]["perturb_probability"]
#perturbation = AckermanPerturbation(random_offset_generator=GaussianRandomGenerator(mean=np.array([0.0, 0.0]), std=np.array([1.0, np.pi / 6])),
                                    #perturb_prob=train_cfg["perturb_probability"],)

train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()

train_dataset_2 = AgentDataset(cfg, train_zarr, rasterizer)
train_dataloader_2 = DataLoader(train_dataset_2, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"],
                                num_workers=train_cfg["num_workers"])

print("==================================TRAIN DATA==================================")
print(train_dataset_2)

"""
Loss Function
"""


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
    assert len(
        pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len,
                        num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (
        batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones(
        (batch_size,))), "confidences should sum to 1"
    assert avails.shape == (
        batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(
    ), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    # reduce coords and use availability
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)

    # when confidence is 0 log goes to -inf, but we're fine with it
    with np.errstate(divide="ignore"):
        # error (batch_size, num_modes)
        error = torch.log(confidences) - 0.5 * \
            torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    # error are negative at this point, so max() gives the minimum one
    max_value, _ = error.max(dim=1, keepdim=True)
    error = -torch.log(torch.sum(torch.exp(error - max_value),
                       dim=-1, keepdim=True)) - max_value  # reduce modes
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

"""
Model
"""


class Model(nn.Module):

    def __init__(self, cfg: Dict, num_modes=3):
        super().__init__()

        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        self.backbone = backbone

        num_history_channels = (
            cfg["model_params"]["history_num_frames"] + 1) * 2

        num_in_channels = 3 + num_history_channels

        self.backbone.conv1 = nn.Sequential(
            nn.Conv2d(
                num_in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=False,
            ),
            nn.ReLU(),  # Added ReLU
            # Pooling layer
            nn.MaxPool2d(kernel_size=self.backbone.conv1.kernel_size),
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
            nn.Dropout(0.2),
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

        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        bs, _ = x.shape
        pred, confidences = torch.split(x, self.num_preds, dim=1)
        pred = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        return pred, confidences


def forward(data, model, device, criterion=pytorch_neg_multi_log_likelihood_batch):
    inputs = data["image"].to(device)
    target_availabilities = data["target_availabilities"].to(device)
    targets = data["target_positions"].to(device)
    # Forward pass
    preds, confidences = model(inputs)
    loss = criterion(targets, preds, confidences, target_availabilities)
    return loss, preds, confidences


"""
Setup usage of cuda
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model(cfg)

#load weight if there is a pretrained model
weight_path = cfg["model_params"]["weight_path"]
if weight_path:  # https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
    torch.save(model.state_dict(), weight_path)
    model.load_state_dict(copy.deepcopy(torch.load(weight_path, device)))

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f'device {device}')
from torchsummary import summary
# Added this to output our layers of the "pretrained"
summary(model, (25, 512, 1024))


"""
Train time
"""

tr_it = iter(train_dataloader_2)

# progress bar for command line
progress = tqdm(range(cfg["train_params"]["max_num_steps"]))

# ## progress bar for google colab
# progress = tq.tqdm(range(cfg["train_params"]["max_num_steps"]))

torch.set_grad_enabled(False)
#############################################################
# using this to prevent error not sure what this list or array should be
# Metrics otherwise
losses_train = []
iterations = []
metrics = []
times = []
start = time.time()
model_name = cfg["model_params"]["model_name"]
#############################################################
for i in progress:
    try:
        data = next(tr_it)
    except StopIteration:
        tr_it = iter(train_dataloader_2)
        data = next(tr_it)

    model.train()
    torch.set_grad_enabled(True)
    loss, *_ = forward(data, model, device)
    """
    Error: AssertionError: confidences should sum to 1
    The error appeared because the model puts in somecases NaN values out, which resulted to AssertionError: confidences should sum to 1.
    This issue is also known under the term Exploding Gradient
    https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles/discussion/187773
    """
    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), 1)

    optimizer.step()

    losses_train.append(loss.item())
    ## We'll gather metrics every even step
    if i % cfg['train_params']['checkpoint_every_n_steps'] == 0:
        torch.save(model.state_dict(), f'{model_name}_{i}.pth')
        iterations.append(i)
        metrics.append(np.mean(losses_train))
        times.append((time.time()-start)/60)


"""
Plot and save model
"""

plt.plot(np.arange(len(losses_train)), losses_train, label="train loss")
plt.legend()
plt.savefig("pre_results.png")
plt.show()
#############################################################
to_save = torch.jit.script(model.cpu())
path_to_save = "/home-new/tle728/dataset/models/planning_model.pt"
to_save.save(path_to_save)
print(f"MODEL STORED at {path_to_save}")
#############################################################
# Allow is to have better idea of the overall standing of the training
progress.set_description(
    f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
results = pd.DataFrame({'iterations': iterations,
                       'metrics (avg)': metrics, 'elapsed_time (mins)': times})
results.to_csv(f"train_metrics_{model_name}_{num_iter}.csv", index=False)
print(f"Total training time is {(time.time()-start)/60} mins")
print(results.head())
