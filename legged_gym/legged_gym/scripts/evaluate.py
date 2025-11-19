# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import code

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import cv2
from collections import deque
import statistics
import faulthandler
from copy import deepcopy
import matplotlib.pyplot as plt
from time import time, sleep
from legged_gym.utils import webviewer
from tqdm import tqdm

PRINT_FOR_LATEX = False

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
                    
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
        # code.interact(local=locals())
    # else:
    #     model = "model{}_jit.pt".format(checkpoint) 

    # load_path = root + model
    return model, checkpoint

def run_depth_encoder(env, third_person_camera_disconnection_time, ppo_runner, depth_encoder, infos, obs):
    if ppo_runner.if_third_person_depth_disconnection:
        if ppo_runner.if_third_person_depth_completely_disconnected:
            infos["third_person_depth_disconnected"] = torch.ones_like(infos["third_person_depth_disconnected"], dtype=torch.bool, device=ppo_runner.device)
        else:
            el = (torch.rand(env.num_envs, device=ppo_runner.device) <= env.cfg.domain_rand.disconnection_rate_third_person_camera) & (torch.normal(third_person_camera_disconnection_time["mean"], third_person_camera_disconnection_time["std_dev"]).to(ppo_runner.device) <= infos["third_person_depth_disconnection_time"])
            infos["third_person_depth_disconnected"][el] = ~infos["third_person_depth_disconnected"][el]
            infos["third_person_depth_disconnection_time"][el] = 0.0
        if infos["third_person_depth"] != None:
            infos["third_person_depth"][infos["third_person_depth_disconnected"]] = torch.zeros_like(infos["third_person_depth"][0,:], device=ppo_runner.device)
        elif ppo_runner.if_third_person_depth:
            infos["depth"][infos["third_person_depth_disconnected"]] = torch.zeros_like(infos["depth"][0,:]).to(device=ppo_runner.device)

    if infos["third_person_depth"] != None:
        if ppo_runner.if_third_person_depth_disconnection:
            infos["third_person_depth"][infos["third_person_depth_disconnected"]] = torch.zeros_like(infos["third_person_depth"][0,:]).to(device=ppo_runner.device)
        if ppo_runner.if_use_disconnection_aware_depth_encoder:
            depth_latent_and_yaw = depth_encoder(infos["depth"].clone(), infos["third_person_depth"].clone(), infos["third_person_depth_disconnected"].float().unsqueeze(1).clone(), obs)  # clone is crucial to avoid in-place operation
        else:
            depth_latent_and_yaw = depth_encoder(infos["depth"].clone(), infos["third_person_depth"].clone(), obs)  # clone is crucial to avoid in-place operation
    else:
        depth_latent_and_yaw = depth_encoder(infos["depth"], obs)
    
    return depth_latent_and_yaw

def play(args):
    if args.web:
        web_viewer = webviewer.WebViewer()
    faulthandler.enable()
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = 256
    env_cfg.env.episode_length_s = 20
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {"smooth slope": 0., 
                                    "rough slope up": 0.0,
                                    "rough slope down": 0.0,
                                    "rough stairs up": 0., 
                                    "rough stairs down": 0., 
                                    "discrete": 0., 
                                    "stepping stones": 0.0,
                                    "gaps": 0., 
                                    "smooth flat": 0,
                                    "pit": 0.0,
                                    "wall": 0.0,
                                    "platform": 0.,
                                    "large stairs up": 0.,
                                    "large stairs down": 0.,
                                    "parkour": 0.25,
                                    "parkour_hurdle": 0.25,
                                    "parkour_flat": 0.,
                                    "parkour_step": 0.25,
                                    "parkour_gap": 0.25, 
                                    "demo": 0}
    
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    
    # env_cfg.depth.angle = [0, 1]
    # env_cfg.depth.angle = [[0, 0, 0], 
    #                        [0, 1, 0]] # X, Y, Z; positive pitch down
    # env_cfg.depth.third_person_displacement_radius = 0.5
    # env_cfg.depth.third_person_angle = [[0, 0, 0], 
    #                                     [0, 1, 0]] # X, Y, Z; positive pitch down
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = True
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    depth_latent_buffer = []
    # prepare environment
    env: LeggedRobot
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if args.web:
        web_viewer.setup(env)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    if env.cfg.depth.use_camera or env.cfg.depth.use_third_person_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
    
    total_steps = 1000
    rewbuffer = deque(maxlen=total_steps)
    lenbuffer = deque(maxlen=total_steps)
    num_waypoints_buffer = deque(maxlen=total_steps)
    time_to_fall_buffer = deque(maxlen=total_steps)
    edge_violation_buffer = deque(maxlen=total_steps)

    killed_by_disconnection_buffer = deque(maxlen=total_steps)
    disconnectionTime_buffer = deque(maxlen=total_steps)

    cur_reward_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_episode_length = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_edge_violation = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    cur_time_from_start = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    # infos["depth"] = env.depth_buffer.clone().to(ppo_runner.device)[:, -1] if ppo_runner.if_depth else None
    infos["depth"] = env.third_person_depth_buffer.clone().to(ppo_runner.device)[:, -1] if (ppo_runner.if_third_person_depth and not ppo_runner.if_depth) else (env.depth_buffer.clone().to(ppo_runner.device)[:, -1]  if ppo_runner.if_depth else None)
    infos["third_person_depth"] = env.third_person_depth_buffer.clone().to(ppo_runner.device)[:, -1] if (ppo_runner.if_third_person_depth and ppo_runner.if_depth) else None
    infos["third_person_depth_disconnected"] = torch.zeros(env.num_envs, dtype=torch.bool, device=ppo_runner.device)
    infos["third_person_depth_disconnection_time"] = torch.zeros(env.num_envs, dtype=torch.float, device=ppo_runner.device)
    
    third_person_camera_disconnection_time = {}
    third_person_camera_disconnection_time["mean"] = torch.ones(env.num_envs, dtype=torch.float, device=ppo_runner.device) * env.cfg.domain_rand.disconnection_mean_time_third_person_camera
    third_person_camera_disconnection_time["std_dev"] = torch.ones(env.num_envs, dtype=torch.float, device=ppo_runner.device) * env.cfg.domain_rand.disconnection_std_dev_time_third_person_camera
    
    for i in tqdm(range(1500)):

        if env.cfg.depth.use_camera or env.cfg.depth.use_third_person_camera:
            if infos["depth"] is not None:
                obs_student = obs[:, :env.cfg.env.n_proprio]
                obs_student[:, 5:7] = 0
                with torch.no_grad():
                    # depth_latent_and_yaw = depth_encoder(infos["depth"], obs_student)
                    depth_latent_and_yaw = run_depth_encoder(env, third_person_camera_disconnection_time, ppo_runner, depth_encoder, infos, obs_student)
                depth_latent = depth_latent_and_yaw[:, :-2]
                yaw = depth_latent_and_yaw[:, -2:]
            obs[:, 5:7] = 1.5*yaw
                
        else:
            depth_latent = None

        if hasattr(ppo_runner.alg, "depth_actor"):
            with torch.no_grad():
                actions = ppo_runner.alg.depth_actor(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
            
        cur_goal_idx = env.cur_goal_idx.clone()
        obs, _, rews, dones, extras = env.step(actions.detach())
        infos.update(extras) # Updating to avoid removing values not updated by step, such as third_person_depth_disconnected
        

        if args.web:
            web_viewer.render(fetch_results=True,
                        step_graphics=True,
                        render_all_camera_sensors=True,
                        wait_for_page_load=True)
        
        id = env.lookat_id
        # Log stuff
        edge_violation_buffer.extend(env.feet_at_edge.sum(dim=1).float().cpu().numpy().tolist())
        # cur_edge_violation += env.feet_at_edge.sum(dim=1).float()
        cur_reward_sum += rews
        cur_episode_length += 1
        cur_time_from_start += 1

        new_ids = (dones > 0).nonzero(as_tuple=False)
        killed_ids = ((dones > 0) & (~infos["time_outs"])).nonzero(as_tuple=False)
        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
        num_waypoints_buffer.extend(cur_goal_idx[new_ids][:, 0].cpu().numpy().tolist())
        time_to_fall_buffer.extend(cur_time_from_start[killed_ids][:, 0].cpu().numpy().tolist())

        if ppo_runner.if_third_person_depth_disconnection:
            killed_by_disconnection_buffer.extend(infos["third_person_depth_disconnected"][killed_ids][:, 0].cpu().numpy().tolist())
            infos["third_person_depth_disconnected"][dones] = False

            disconnectionTime_buffer.extend(infos["third_person_depth_disconnection_time"][new_ids][:, 0].cpu().numpy().tolist())
            infos["third_person_depth_disconnection_time"] += env.dt
            infos["third_person_depth_disconnection_time"][dones] = 0.0

        cur_reward_sum[new_ids] = 0
        cur_episode_length[new_ids] = 0
        cur_edge_violation[new_ids] = 0
        cur_time_from_start[killed_ids] = 0
    
    #compute buffer mean and std
    rew_mean = statistics.mean(rewbuffer)
    rew_std = statistics.stdev(rewbuffer)

    len_mean = statistics.mean(lenbuffer)
    len_std = statistics.stdev(lenbuffer)

    num_waypoints_mean = np.mean(np.array(num_waypoints_buffer).astype(float)/7.0)
    num_waypoints_std = np.std(np.array(num_waypoints_buffer).astype(float)/7.0)

    # time_to_fall_mean = statistics.mean(time_to_fall_buffer)
    # time_to_fall_std = statistics.stdev(time_to_fall_buffer)

    edge_violation_mean = np.mean(edge_violation_buffer)
    edge_violation_std = np.std(edge_violation_buffer)

    failure_rate = float(len(time_to_fall_buffer))/float(len(num_waypoints_buffer))*100.0
    success_rate = 100.0 - failure_rate

    if len(killed_by_disconnection_buffer) > 0:
        crash_by_disconnection = float(killed_by_disconnection_buffer.count(True))/float(len(killed_by_disconnection_buffer))
    else:
        crash_by_disconnection = 0.0

    time_disconnected_mean = np.mean(disconnectionTime_buffer)
    time_disconnected_std = np.std(disconnectionTime_buffer)

    print("Mean reward: {:.2f}\u00B1{:.2f}".format(rew_mean, rew_std))
    print("Mean episode length: {:.2f}\u00B1{:.2f}".format(len_mean, len_std))
    print("Absolute success rate: {:.2f}".format(success_rate))
    print("Mean number of waypoints: {:.2f}\u00B1{:.2f}".format(num_waypoints_mean, num_waypoints_std))
    # print("Mean time to fall: {:.2f}\u00B1{:.2f}".format(time_to_fall_mean, time_to_fall_std))
    print("Mean edge violation: {:.2f}\u00B1{:.2f}".format(edge_violation_mean, edge_violation_std))
    print("Failure rate: {:.2f}".format(failure_rate))
    if ppo_runner.if_third_person_depth_disconnection:
        print("Failures caused by disconnections: {:.2f}".format(crash_by_disconnection))
        print("Mean time disconnected: {:.2f}\u00B1{:.2f}".format(time_disconnected_mean, time_disconnected_std))
    else:
        print("Crashes caused by disconnections: N/A")
        print("Mean time disconnected: N/A")


    if PRINT_FOR_LATEX:
        print("LaTeX-formatted results:")
        print("{:.2f}\u00B1{:.2f} & % Mean reward".format(rew_mean, rew_std))
        print("% {:.2f}\u00B1{:.2f} & % Mean episode length".format(len_mean, len_std))
        print("{:.2f} \\% & % Absolute success rate".format(success_rate))
        print("{:.2f}\u00B1{:.2f} & % Mean number of waypoints".format(num_waypoints_mean, num_waypoints_std))
        # print("{:.2f}\u00B1{:.2f} & % Mean time to fall".format(time_to_fall_mean, time_to_fall_std))
        print("{:.2f}\u00B1{:.2f} \\\\ % Mean edge violation".format(edge_violation_mean, edge_violation_std))
        print("% {:.2f} \\% & % Failure rate".format(failure_rate))
        if ppo_runner.if_third_person_depth_disconnection:
            print("% {:.2f} & % Failures caused by disconnections".format(crash_by_disconnection))
            print("{:.2f}\u00B1{:.2f} \\\\ % Mean time disconnected".format(time_disconnected_mean, time_disconnected_std))
        else:
            print("% N/A &% Failures caused by disconnections")
            print("% N/A \\\\ % Mean time disconnected")

if __name__ == '__main__':
    EXPORT_POLICY = False
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
