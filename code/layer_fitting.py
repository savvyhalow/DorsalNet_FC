import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as tf
import sys
from dorsalnet import DorsalNet, FC, interpolate_frames
sys.path.append('../code')
from VWAM.utils import SingleImageFolder, iterate_children, hook_model
from tqdm import tqdm
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import wandb
wandb.login()
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model name", default='DorsalNet', nargs =1, type=str)
parser.add_argument("--experiment", help="experiment name", default='NaturalMovies', nargs =1, type=str)
parser.add_argument("--subject", help="subject id", default='S00', nargs =1, type=str)
parser.add_argument("--opt", help="optimizer", default='adam', nargs =1, type=str)
parser.add_argument("--epochs", help="number of epochs", default=50, nargs =1, type=int)
parser.add_argument("--lr", help="learning rate", default=1e-1, nargs =1, type=float)
parser.add_argument("--gpu", help="cuda device", default='0', nargs =1, type=int)
parser.add_argument("--dtype", help="data type", default='bfloat16', nargs =1)
args = parser.parse_args()

MODEL_NAME = args.model[0] if isinstance(args.model, list) else args.model
OPTIMIZER = args.opt[0] if isinstance(args.opt, list) else args.opt
N_EPOCHS = args.epochs[0] if isinstance(args.epochs, list) else args.epochs
LR_INIT = args.lr[0] if isinstance(args.lr, list) else args.lr
EXPERIMENT = args.experiment[0] if isinstance(args.experiment, list) else args.experiment
SUBJECT_ID = args.subject[0] if isinstance(args.subject, list) else args.subject
DEVICE = f'cuda:{args.gpu[0] if isinstance(args.gpu, list) else args.gpu}'
DTYPE = args.dtype[0] if isinstance(args.dtype, list) else args.dtype

run = wandb.init(
    # Set the project where this run will be logged
    project="DorsalNet_FC_Pilot",
    # Track hyperparameters and run metadata
    config={
        "model_name": MODEL_NAME,
        "experiment": EXPERIMENT,
        "subject_id": SUBJECT_ID,
        "optimizer": OPTIMIZER,
        "epochs": N_EPOCHS,
        "learning_rate": LR_INIT,
})

DTYPE = torch.bfloat16

if MODEL_NAME.lower() == 'dorsalnet':
    DEPTH = 1
    INPUT_SIZE = (1, 3, 32, 112, 112)
    preprocess = tf.Compose([
        tf.Resize(112),
        tf.ToTensor(),
    ])
    image_augmentations = tf.Compose([
        tf.RandomCrop(112, padding=4),
        tf.RandomRotation(10),
        tf.RandomCrop(112, padding=3),
    ])
    model = DorsalNet(False, 32).eval().to(DEVICE).to(DTYPE)
    model.load_state_dict(torch.load('/home/matthew/Data/DorsalNet_FC/base_models/DorsalNet/pretrained.pth'))
else:
    DEPTH = 2
    INPUT_SIZE = (1, 3, 224, 224)
    preprocess = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])
    image_augmentations = tf.Compose([
        tf.RandomCrop(224, padding=4),
        tf.RandomRotation(10),
        tf.RandomCrop(224, padding=3),
    ])
    if MODEL_NAME.lower() == 'alexnet':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True).eval().to(DEVICE).to(DTYPE)
    elif MODEL_NAME.lower() == 'inception_v3':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True).eval().to(DEVICE).to(DTYPE)
    elif MODEL_NAME.lower() == 'vgg16':
        model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True).eval().to(DEVICE).to(DTYPE)

MAX_FS = 5000

def choose_downsampling(activations, max_fs):
    num_channels = activations.shape[1]
    if activations.ndim == 4:
        max_output_dim = int((max_fs / num_channels)**(1/2))
        return torch.nn.AdaptiveMaxPool2d(max_output_dim)
    elif activations.ndim == 5:
        max_output_dim = int((max_fs / num_channels)**(1/3))
        return torch.nn.AdaptiveMaxPool3d(max_output_dim)

layers_dict = iterate_children(model, depth=DEPTH)
layers_dict = {layer_name: ds_function for layer_name, ds_function in layers_dict.items() if 'dropout' not in layer_name and 'concat' not in layer_name}
model = hook_model(model, layers_dict)
model(torch.randn(INPUT_SIZE).to(DEVICE).to(DTYPE))

layer_downsampling_fns = {}
for layer_name, layer_activations in model.activations.items():
    layer_activations = layer_activations
    # print('**************')
    # print(layer_name)
    # print('old_shape:', layer_activations.shape)
    # print('old # activations:', layer_activations.flatten().shape)
    layer_downsampling_fn = choose_downsampling(layer_activations, MAX_FS)
    layer_downsampling_fns[layer_name] = layer_downsampling_fn
    if layer_downsampling_fn is not None:
        layer_activations = layer_downsampling_fns[layer_name](layer_activations)
    # print('new_shape:', layer_activations.shape)
    # print('new # activations:', layer_activations.flatten().shape)
### Initialize FC Layer
trn_brain = np.load(f'/home/matthew/Data/DorsalNet_FC/fMRI_data/{SUBJECT_ID}/NaturalMovies/trn.npy')
trn_brain = torch.tensor(np.nan_to_num(trn_brain), device=DEVICE)
n_voxels = trn_brain.shape[1]

val_brain = np.load(f'/home/matthew/Data/DorsalNet_FC/fMRI_data/{SUBJECT_ID}/NaturalMovies/val_rpts.npy')
val_brain = torch.tensor(np.nan_to_num(val_brain).mean(0), device=DEVICE)

fc = FC(n_voxels).to(DEVICE).to(DTYPE)
# print(fc)

def column_corr(A, B, dof=0):
    """Efficiently compute correlations between columns of two matrices
    
    Does NOT compute full correlation matrix btw `A` and `B`; returns a 
    vector of correlation coefficients. FKA ccMatrix."""
    zs = lambda x: (x-np.nanmean(x, axis=0))/np.nanstd(x, axis=0, ddof=dof)
    rTmp = np.nansum(zs(A)*zs(B), axis=0)
    n = A.shape[0]
    # make sure not to count nans
    nNaN = np.sum(np.logical_or(np.isnan(zs(A)), np.isnan(zs(B))), 0)
    n = n - nNaN
    r = rTmp/n
    return r


batch_sizes = {
    'NaturalMovies': 30,
    'vedb_ver01': 50,
}

preprocess = tf.Compose([
    tf.Resize(112),
    tf.ToTensor(),
])

image_augmentations = tf.Compose([
    tf.RandomCrop(112, padding=4),
    tf.RandomRotation(10),
    tf.RandomCrop(112, padding=3),
])

trn_dl = DataLoader(
    SingleImageFolder(f'/home/matthew/Data/DorsalNet_FC/stimuli/{EXPERIMENT}/images/trn', transform=preprocess),
    batch_size=batch_sizes[EXPERIMENT], 
    shuffle=False)

val_dl = DataLoader(
    SingleImageFolder(f'/home/matthew/Data/DorsalNet_FC/stimuli/{EXPERIMENT}/images/val', transform=preprocess),
    batch_size=batch_sizes[EXPERIMENT], 
    shuffle=False)

torch.cuda.empty_cache()
if OPTIMIZER.lower() == 'sgd':
    optimizer = torch.optim.SGD(fc.parameters(), lr=LR_INIT)
elif OPTIMIZER.lower() == 'adam':
    optimizer = torch.optim.Adam(fc.parameters(), lr=LR_INIT)

def train():
    pbar = tqdm(enumerate(trn_dl), total=len(trn_brain), desc=f"Epoch {epoch} Training")
    trn_epoch_losses = []
    for i, batch in pbar:
        optimizer.zero_grad()
        if MODEL_NAME.lower() == 'DorsalNet':
            batch = interpolate_frames(batch, INPUT_SIZE[2]).unsqueeze(0)
        model.forward(image_augmentations(batch).to(DTYPE).to(DEVICE))
        all_activations = []
        for layer_name, layer_activations in model.activations.items():
            layer_downsampling_fn = layer_downsampling_fns[layer_name]
            if layer_downsampling_fn is not None:
                layer_activations = layer_downsampling_fn(layer_activations)
            all_activations.append(layer_activations.mean(0).flatten())
            model.activations[layer_name] = 0
        fc_out = fc(torch.cat(all_activations).unsqueeze(0))
        batch_brain = (trn_brain[min(i+2, len(trn_brain)-1)] + trn_brain[min(i+3, len(trn_brain)-1)]) / 2
        loss = torch.square(fc_out[0]/1000 - batch_brain).sum().sqrt()
        loss.backward()
        optimizer.step()
        trn_epoch_losses.append(loss.item())
        pbar.set_postfix_str(f"Mean Epoch Loss: {torch.mean(torch.tensor(trn_epoch_losses)).item():.2f}")
    return trn_epoch_losses

def validate():
    with torch.no_grad():
        pbar = tqdm(enumerate(val_dl), total=len(val_brain), desc=f"Epoch {epoch} Validation")
        val_outputs = []
        val_epoch_losses = []
        for i, batch in pbar:
            if MODEL_NAME.lower() == 'DorsalNet':
                batch = interpolate_frames(batch, INPUT_SIZE[2]).unsqueeze(0)
            model.forward(batch.unsqueeze(DTYPE).to(DEVICE))
            all_activations = []
            for layer_name, layer_activations in model.activations.items():
                layer_downsampling_fn = layer_downsampling_fns[layer_name]
                if layer_downsampling_fn is not None:
                    layer_activations = layer_downsampling_fn(layer_activations)
                all_activations.append(layer_activations.mean(0).flatten())
                model.activations[layer_name] = 0
            fc_out = fc(torch.cat(all_activations).unsqueeze(0))
            batch_brain = (val_brain[min(i+2, len(val_brain)-1)] + val_brain[min(i+3, len(val_brain)-1)]) / 2
            loss = torch.square(fc_out[0]/1000 - batch_brain).sum().sqrt()
            val_outputs.append(fc_out.cpu().float().numpy())
            val_epoch_losses.append(loss.item())
            pbar.set_postfix_str(f"Mean Epoch Loss: {torch.mean(torch.tensor(val_epoch_losses)).item():.2f}")
        ccs = column_corr(np.concatenate(val_outputs), val_brain.cpu().numpy())
        print(f"Mean Prediction Accuracy: {ccs.mean():.2f}")
    return val_epoch_losses, ccs
        
def log(epoch, trn_epoch_losses, val_epoch_losses, ccs):
    wandb.log({
        "epoch": epoch,
        "trn_loss": torch.mean(torch.tensor(trn_epoch_losses)).item(),
        "val_loss": torch.mean(torch.tensor(val_epoch_losses)).item(),
        "val_acc": ccs.mean(),
    })

save_dir = f'/home/matthew/Data/DorsalNet_FC/fits/{EXPERIMENT}/{SUBJECT_ID}'
os.makedirs(save_dir, exist_ok=True)
        
def log(epoch, trn_epoch_losses, val_epoch_losses, ccs):
    wandb.log({
        "epoch": epoch,
        "trn_loss": torch.mean(torch.tensor(trn_epoch_losses)).item(),
        "val_loss": torch.mean(torch.tensor(val_epoch_losses)).item(),
        "val_acc": ccs.mean(),
    })

save_dir = f'/home/matthew/Data/DorsalNet_FC/fits/{EXPERIMENT}/{SUBJECT_ID}'
os.makedirs(save_dir, exist_ok=True)
# epoch = -1
# validate()
for epoch in range(N_EPOCHS):
    trn_epoch_losses = train()
    val_epoch_losses, ccs = validate()
    log(epoch, trn_epoch_losses, val_epoch_losses, ccs)
torch.save(model.state_dict(), f"{save_dir}/{MODEL_NAME}_{OPTIMIZER}_{N_EPOCHS}_{LR_INIT}.pt")