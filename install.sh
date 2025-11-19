# sudo apt install git git-lfs
conda create -n parkour python=3.8
conda activate parkour
sudo cp ~/miniconda3/envs/parkour/lib/libpython3.8.so.1.0 /usr/lib/
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xf isaac-gym-preview-4
rm -r isaac-gym-preview-4
cd isaacgym/python && pip install -e .
cd ../../rsl_rl && pip install -e .
cd ../legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask