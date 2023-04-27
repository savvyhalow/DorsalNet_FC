### Do in normal terminal:
# mamba create -n DorsalNet_FC python=3.10 -y
# conda activate DorsalNet_FC

mamba install -c anaconda libopenblas -y
mamba install -c "nvidia/label/cuda-11.7.0" cuda-toolkit -y
mamba install pytorch torchvision sympy ipykernel ipywidgets tqdm matplotlib -y
mamba install -c conda-forge wandb -y