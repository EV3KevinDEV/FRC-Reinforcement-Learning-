# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from .robot import ROBOT_CFG


@configclass
class FrcRlEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 15.0  # Increased for coral collection tasks
    # - spaces definition
    action_space = 5  # 5 actions: vx, vy, omega, elevator/intake level, gripper
    observation_space = 15  # robot motion (3) + goal geometry (3) + mechanism readiness (3) + time context (1) + latency compensation (5)
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation, device="cpu")

    # robot(s)
    robot_cfg: ArticulationCfg = ROBOT_CFG.replace(prim_path="/World/envs/env_.*/MasterRCSH")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4, env_spacing=25.0, replicate_physics=False)

    # FRC RL specific parameters
    max_corals = 10  # Maximum corals per environment
    
    # reward scales - optimized for phase-based system
    rew_scale_alive = 0.05  # Reduced base alive bonus to prevent overwhelm
    rew_scale_tipping = -10.0  # Stronger tipping penalty for safety
    rew_scale_coral_grip = 20.0  # Strong pickup reward for successful events
    rew_scale_coral_carrying = 0.3  # Carrying bonus (applied in carry phases)
    rew_scale_coral_drop = -8.0  # Strong drop penalty to discourage dropping
    rew_scale_weld_success = 50.0  # High success reward for completing task
    rew_scale_distance_improvement = 5.0  # Moderate improvement reward
    rew_scale_smoothness = -0.02  # Slightly stronger smoothness penalty
    rew_scale_time_penalty = -0.005  # Reduced time pressure (phase-variable)
    
    # action scale
    action_scale = 1.0