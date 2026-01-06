# swerve_twist_action.py
# Isaac Lab ActionTerm for robot-centric swerve control.
# Inputs: (vx, vy, wz) per env. Outputs: steer joint position targets, drive joint velocity targets.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Optional

import torch

# Isaac Lab base classes
from isaaclab.envs.mdp.actions import ActionTerm, ActionTermCfg


# ---------------------------- Config ----------------------------

@dataclass
class SwerveTwistActionCfg(ActionTermCfg):
    """
    Robot-centric swerve action term config.

    Attributes:
      asset_name:        name of the robot asset in env.scene
      steer_dof_indices: length M; steer joint indices (per articulation) in desired FL,FR,RL,RR order
      drive_dof_indices: length M; drive joint indices (per articulation) in same order
      module_xy:         length M; per-module (x,y) position in meters, robot frame (+x fwd, +y left)
      wheel_radius:      wheel radius in meters (linear -> angular conversion)
      max_wheel_speed:   wheel linear speed limit (m/s). Used for desaturation.
      max_wheel_accel:   optional wheel linear accel limit (m/s^2) for slew limiting
      scale:             multiplier on incoming (vx,vy,wz) from policy
      zero_eps:          small threshold to treat command as zero to avoid jitter
    """
    class_type: type = None  # set by caller or registry
    asset_name: str = "robot"
    steer_dof_indices: Sequence[int] = ()
    drive_dof_indices: Sequence[int] = ()
    module_xy: Sequence[Tuple[float, float]] = ()
    wheel_radius: float = 0.050  # m
    max_wheel_speed: float = 4.5  # m/s
    max_wheel_accel: Optional[float] = 8.0  # m/s^2 (None disables)
    scale: float = 1.0
    zero_eps: float = 1e-6


# ---------------------------- Action Term ----------------------------

class SwerveTwistAction(ActionTerm):
    """
    Robot-centric swerve twist ActionTerm.
    - process_actions(): store scaled (vx,vy,wz)
    - apply_actions(): compute per-module steer/drive targets, write to articulation
    """

    cfg: SwerveTwistActionCfg

    # ---------- Action space dim ----------
    @property
    def action_dim(self) -> int:
        # (vx, vy, wz)
        return 3

    # ---------- Init ----------
    def __init__(self, cfg: SwerveTwistActionCfg, env):
        super().__init__(cfg, env)
        self.robot = self._env.scene[cfg.asset_name]

        # geometry
        self.M = len(cfg.steer_dof_indices)
        assert self.M == len(cfg.drive_dof_indices), "steer/drive idx length mismatch"
        assert self.M == len(cfg.module_xy), "module_xy length must equal number of modules"

        # buffers
        self._vxvywz = torch.zeros((env.num_envs, 3), device=env.device)
        self._last_lin_speed = torch.zeros((env.num_envs, self.M), device=env.device)

        # module positions as (1, M) for broadcast
        xy = torch.tensor(cfg.module_xy, dtype=torch.float32, device=env.device)  # (M,2)
        self._x = xy[:, 0].unsqueeze(0)  # (1,M)
        self._y = xy[:, 1].unsqueeze(0)  # (1,M)

        # selected joint indices (PyTorch tensors for API)
        self._steer_ids = torch.tensor(cfg.steer_dof_indices, dtype=torch.int64, device=env.device)
        self._drive_ids = torch.tensor(cfg.drive_dof_indices, dtype=torch.int64, device=env.device)

        # small constants
        self._pi = math.pi
        self._half_pi = math.pi / 2.0

    # ---------- Actions from policy ----------
    def process_actions(self, actions: torch.Tensor):
        """
        actions: (N, 3) -> (vx, vy, wz) robot-centric per env.
        """
        # Ensure we have valid actions
        if actions is None or actions.numel() == 0:
            self._vxvywz = torch.zeros((self._env.num_envs, 3), device=self._env.device, dtype=torch.float32)
            return
            
        # scale per config
        self._vxvywz = actions * self.cfg.scale

    # ---------- Apply to sim ----------
    def apply_actions(self):
        """
        Compute per-module commands and write to articulation targets.
        - steer: joint position target (rad)
        - drive: joint velocity target (rad/s at wheel shaft)
        """
        dt = float(self._env.step_dt)
        dev = self._env.device
        N = self._env.num_envs
        M = self.M

        # current steer joint positions (N, total_joints) -> (N,M) index select
        q_all = self.robot.data.joint_pos  # (N, J)
        q = q_all.index_select(dim=1, index=self._steer_ids)  # (N, M), radians

        # unpack commands (N,1) for broadcasting vs (1,M)
        vx = self._vxvywz[:, 0:1]  # (N,1)
        vy = self._vxvywz[:, 1:2]  # (N,1)
        wz = self._vxvywz[:, 2:3]  # (N,1)

        # per-module rigid-body twist (robot frame)
        # v_i = [ vx - wz*y_i , vy + wz*x_i ]  -> (N,M)
        vx_i = vx - wz * self._y  # (N,M)
        vy_i = vy + wz * self._x  # (N,M)

        # raw setpoints
        angle_raw = torch.atan2(vy_i, vx_i)     # (N,M)
        speed_raw = torch.hypot(vx_i, vy_i)     # (N,M)  linear m/s

        # desaturation per-env to respect max wheel speed
        # peak = max over modules (abs)
        peak = torch.amax(speed_raw.abs(), dim=1, keepdim=True)  # (N,1)
        max_w = max(self.cfg.max_wheel_speed, 1e-12)
        scale = torch.clamp(max_w / torch.clamp(peak, min=1e-12), max=1.0)  # (N,1)
        speed = speed_raw * scale  # (N,M)
        angle = angle_raw          # (N,M)

        # zero-command hold (avoid jitter)
        zero_cmd = (self._vxvywz.abs().sum(dim=1, keepdim=True) < self.cfg.zero_eps)  # (N,1)
        if zero_cmd.any():
            # where zero, hold angle = q and set speed = 0
            angle = torch.where(zero_cmd, q, angle)
            speed = torch.where(zero_cmd, torch.zeros_like(speed), speed)

        # angle optimization: if shortest delta > 90°, add pi and flip speed
        delta = wrap_pi_t(angle - q)  # (N,M)
        flip_mask = (delta.abs() > self._half_pi)  # (N,M)
        angle_opt = wrap_pi_t(angle + flip_mask * self._pi)
        speed_opt = torch.where(flip_mask, -speed, speed)

        # optional slew-rate limit on linear speed
        if self.cfg.max_wheel_accel is not None and self.cfg.max_wheel_accel > 0.0 and dt > 0.0:
            speed_cmd = slew_limit_t(speed_opt, self._last_lin_speed, self.cfg.max_wheel_accel, dt)
        else:
            speed_cmd = speed_opt

        # cache last speed for next step
        self._last_lin_speed = speed_cmd

        # final commands:
        # steer position target (rad)
        steer_pos = angle_opt  # (N,M)
        # drive velocity target: convert linear m/s -> wheel rad/s
        wheel_omega = speed_cmd / max(self.cfg.wheel_radius, 1e-12)  # (N,M)

        # write to articulation
        # Note: Isaac Lab Articulation expects (N, num_selected) targets when joint_ids provided.
        self.robot.set_joint_position_target(steer_pos, joint_ids=self._steer_ids)
        self.robot.set_joint_velocity_target(wheel_omega, joint_ids=self._drive_ids)

    # ---------- Required properties ----------
    @property
    def action_dim(self) -> int:
        """Return the dimension of the action space."""
        return 3  # (vx, vy, wz)
    
    @property
    def processed_actions(self) -> torch.Tensor:
        """Return the processed actions (vx, vy, wz) after scaling."""
        return self._vxvywz
    
    @property 
    def raw_actions(self) -> torch.Tensor:
        """Return the raw actions before processing."""
        return self._vxvywz / self.cfg.scale


# ---------------------------- Torch helpers ----------------------------

def wrap_pi_t(a: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle to [-pi, pi). Works elementwise for tensors.
    """
    # Equivalent to ((a + pi) mod 2pi) - pi, but stable on GPU
    two_pi = 2.0 * math.pi
    a_wrapped = (a + math.pi) % two_pi - math.pi
    # Handle the edge case where result == -pi (map to +pi for consistency with wrap)
    a_wrapped = torch.where(a_wrapped == -math.pi, torch.full_like(a_wrapped, math.pi), a_wrapped)
    return a_wrapped


def slew_limit_t(target: torch.Tensor,
                 prev: torch.Tensor,
                 max_rate: float,
                 dt: float) -> torch.Tensor:
    """
    Vectorized slew limiter: clamps (target - prev) to +/- (max_rate * dt).
    """
    if max_rate <= 0 or dt <= 0:
        return target
    delta = target - prev
    lim = max_rate * dt
    delta_clamped = torch.clamp(delta, min=-lim, max=+lim)
    return prev + delta_clamped
