# Twice the Eyes, Half the Falls: Multi-View Depth-Based Learning for Robust Quadruped Locomotion

## Requirements
- Ubuntu (Tested under Ubuntu 22.04 and 24.04)
- build-essential
- git
- git lfs
- [Isaac Gym binaries](https://developer.nvidia.com/isaac-gym)
- [Anaconda](https://anaconda.org/)
    - Python 3.8
    - pip
        - numpy
        - pydelatin
        - wandb
        - tqdm
        - opencv
        - ipdb
        - pyfqmr
        - flask
    - torch
    - torchvision

## Installation
The installation instructions assume the multiview-parkour repository will be installed in the home folder, while Isaac Gym is installed in the repository folder (multiview-parkour/).
First download and install [Anaconda](https://anaconda.org/).
Then, download the [repository](https://anonymous.4open.science/r/multiview-parkour-6FB8).
Finally, you can install the rest of the dependancies with one of the following two methods:

- With the installation script:
    ```bash
    cd multiview-parkour
    source install.sh
    ```

- Manually (Please download and uncompress the [Isaac Gym binaries](https://developer.nvidia.com/isaac-gym) in ~/multiview-parkour/):
    ```bash
    sudo apt install build-essential git git-lfs 
    conda create -n parkour python=3.8 
    conda activate parkour
    sudo cp ~/miniconda3/envs/parkour/lib/libpython3.8.so.1.0 /usr/lib/
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    cd multiview-parkour
    # Download and uncompress the Isaac Gym binaries in multiview-parkour/ from https://developer.nvidia.com/isaac-gym
    cd isaacgym/python && pip install -e .
    cd ../../rsl_rl && pip install -e .
    cd ../legged_gym && pip install -e .
    pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
    ```

## Usage
### Training
Move to the right folder:
`cd legged_gym/legged_gym/scripts`
1. Train base policy:  
```bash
python train.py --task go2 --exptid teacher-XXX-YY-go2 --max_iterations 15000 --headless --device cuda:0 --no_wandb
```
Train 10-15k iterations (8-10 hours on 5070 Super) (at least 15k recommended).

2. Train distillation policy:
The parameters used in the paper were:
    1. Baseline policy:
        ```bash
        python train.py --task go2 --exptid student-XXX-YY-go2 --resume --resumeid teacher-XXX-YY-go2 --max_iterations 10000 --no_wandb --headless --device cuda:0 --num_envs 96  --delay --use_camera
        ```

    2. Remote-only vision policy:
        ```bash
        python train.py --task go2 --exptid student-XXX-YY-go2 --resume --resumeid teacher-XXX-YY-go2 --max_iterations 10000 --no_wandb --headless --device cuda:0 --num_envs 96  --delay --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 
        ```

    3. Combined vision policy trained without RD:
        ```bash
        python train.py --task go2 --exptid student-XXX-YY-go2 --resume --resumeid teacher-XXX-YY-go2 --max_iterations 10000 --no_wandb --headless --device cuda:0 --num_envs 96 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 
        ```

    4. Combined vision policy trained with RD:
        ```bash
        python train.py --task go2 --exptid student-XXX-YY-go2 --resume --resumeid teacher-XXX-YY-go2 --max_iterations 10000 --no_wandb --headless --device cuda:0 --num_envs 96 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 --depth_encoder_aware_of_disconnection --disconnect_third_person_camera --disconnection_rate_third_person_camera 0.1 --disconnection_mean_time_third_person_camera 2.0 --disconnection_std_dev_time_third_person_camera 0.1
        ```

    5. Combined vision policy trained with random displacements:
        ```bash
        python train.py --task go2 --exptid student-XXX-YY-go2 --resume --resumeid teacher-XXX-YY-go2 --max_iterations 10000 --no_wandb --headless --device cuda:0 --num_envs 96 --delay --use_camera --use_third_person_camera --third_person_displacement_radius [DR] --depth_encoder_aware_of_disconnection --disconnect_third_person_camera --disconnection_rate_third_person_camera 0.1 --disconnection_mean_time_third_person_camera 2.0 --disconnection_std_dev_time_third_person_camera 0.1
        ```
        where DR = {0.1, 0.2, 0.3, 0.4, 0.5}

Train 5-10k iterations (5-10 hours on 5070 Super) (at least 5k recommended).

### Playing the policy
If using the provided models, please make sure they are in the following folder: legged_gym/logs/parkour_new/
For example, the provided base teacher policy should be:
multiview-parkour/legged_gym/logs/parkour_new/teacher-go2/model_15000.pt

Move to the right folder:
`cd legged_gym/legged_gym/scripts`
1. Play base policy:
```bash
python play.py --task go2 --exptid teacher-XXX-YY
```
To play the provided model, please use the following:
    python play.py --task go2 --exptid teacher-go2
Delay is added after 8k iters. If you want to play after 8k, add `--delay`

2. Play distillation policy:
    python play.py --task go2  --exptid student-XXX-YY --delay --use_camera
To play the provided models, please use the following commands:
    1. Baseline policy:
        ```bash
        python play.py --task go2 --exptid student-onboard_vision-go2 --delay --use_camera
        ```
    
    2. Remote-only vision policy:
        ```bash
        python play.py --task go2 --exptid student-remote_vision-go2 --delay --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 
        ```

    3. Combined vision policy trained without RD:
        1. Without RD during evaluation:
            ```bash
            python play.py --task go2 --exptid student-combined_vision-RD_unaware-go2 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 
            ```

        2. With RD during evaluation:
            ```bash
            python play.py --task go2 --exptid student-combined_vision-RD_unaware-go2 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 --disconnect_third_person_camera --disconnection_rate_third_person_camera 0.1 --disconnection_mean_time_third_person_camera 2.0 --disconnection_std_dev_time_third_person_camera 0.1
            ```

    4. Combined vision policy trained with RD:
        1. Without RD during evaluation:
            ```bash
            python play.py --task go2 --exptid student-combined_vision-RD_aware-go2 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 --depth_encoder_aware_of_disconnection
            ```
        2. With RD during evaluation:
            ```bash
            python play.py --task go2 --exptid student-combined_vision-RD_aware-go2 --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 --depth_encoder_aware_of_disconnection --disconnect_third_person_camera --disconnection_rate_third_person_camera 0.1 --disconnection_mean_time_third_person_camera 2.0 --disconnection_std_dev_time_third_person_camera 0.1
            ```

    5. Combined vision policy trained with random displacements:
        ```bash
        python train.py --task go2 --exptid student-combined_vision-RD_aware-dr[DR]-go2 --delay --use_camera --use_third_person_camera--third_person_displacement_radius [DR] --depth_encoder_aware_of_disconnection
        ```
        where DR = {0.1, 0.2, 0.3, 0.4, 0.5}

### Evaluate the policy
To evaluate the policies, use the same commands as playing, but with the "evaluate.py", with the "--headless" argument, such as:
For example, to evaluate the Combined vision policy trained with RD with RD during evaluation:
```bash
python evaluate.py --task go2 --exptid student-combined_vision-RD_aware-go2 --headless --delay --use_camera --use_third_person_camera --third_person_fixed --third_person_displacement_radius 0.0 --depth_encoder_aware_of_disconnection --disconnect_third_person_camera --disconnection_rate_third_person_camera 0.1 --disconnection_mean_time_third_person_camera 2.0 --disconnection_std_dev_time_third_person_camera 0.1
```

### Viewer Usage
Can be used in both IsaacGym and web viewer.
- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### Arguments
- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --max_iterations 10000: the specific iteration you want to stop at.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --headless: suppresses the gui of IsaacGym.
- --num_envs: number of environments to run in parallel.
- --use_camera: use egocentric camera or scandots.
- --use_third_person_camera: use exocentric camera.
- --third_person_fixed: fixes the exocentric camera to its initial position (relative to the robot). Even if this argument is set, the camera will be set at a random position according to "third_person_displacement_radius".
- --third_person_displacement_radius: radius used for the spherical displacement model to randomly move the exocentric camera around a fixed position.
- --disconnect_third_person_camera: activates random dropout of the exocentric camera.
- --depth_encoder_aware_of_disconnection: allows the depth encoder to know about the status of the exocentric camera.
- --complete_disconnect_third_person_camera: disconnects the exocentric camera and never reconnects it, emulating the baseline conditions (only egocentric perception data available).
- --disconnection_rate_third_person_camera: the rate at which the exocentric camera disconnects or reconnects.
- --disconnection_mean_time_third_person_camera: mean value for the normal distribution used for remote dropout duration.  
- --disconnection_std_dev_time_third_person_camera: standard deviation for the normal distribution used for remote dropout duration.
- --web: used for playing on headless machines. It will forward a port with vscode and you can visualize seemlessly in vscode with your idle gpu or cpu. [Live Preview](https://marketplace.visualstudio.com/items?itemName=ms-vscode.live-server) vscode extension required, otherwise you can view it in any browser.

### Acknowledgement
https://github.com/chengxuxin/extreme-parkour
https://github.com/leggedrobotics/legged_gym  
https://github.com/Toni-SM/skrl
