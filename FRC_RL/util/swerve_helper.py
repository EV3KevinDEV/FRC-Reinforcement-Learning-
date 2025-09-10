from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Iterable, List, Tuple


# ---------------------- math helpers ----------------------

def wrap_pi(a: float) -> float:
    """Wrap angle to [-pi, pi)."""
    while a >= math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def optimize_angle(target: float, current: float, speed: float) -> Tuple[float, float]:
    """
    Minimize steering movement:
    If the shortest rotation exceeds 90°, flip the wheel (add pi) and negate speed.
    """
    delta = wrap_pi(target - current)
    if abs(delta) > math.pi / 2:
        return wrap_pi(target + math.pi), -speed
    return target, speed


def slew_limit(target: float, prev: float, max_rate: float, dt: float) -> float:
    """
    Limit first derivative of a value (e.g., wheel linear speed).
    max_rate in units/sec (e.g., m/s^2 for speed).
    """
    if dt <= 0 or max_rate <= 0:
        return target
    delta = target - prev
    limit = max_rate * dt
    if delta >  limit: return prev + limit
    if delta < -limit: return prev - limit
    return target


def desaturate_speeds(speeds: Iterable[float], max_speed: float) -> Tuple[float, List[float]]:
    """
    Scale speeds uniformly so that max(|speed|) <= max_speed. Returns (scale, scaled_speeds).
    If all zero or max_speed <= 0, returns scale=1 and original speeds.
    """
    spd_list = list(speeds)
    if max_speed <= 0:
        return 1.0, spd_list
    peak = max((abs(s) for s in spd_list), default=0.0)
    if peak < 1e-12:
        return 1.0, spd_list
    scale = min(1.0, max_speed / peak)
    return scale, [s * scale for s in spd_list]


def field_to_robot(vx_f: float, vy_f: float, heading_rad: float) -> Tuple[float, float]:
    """
    Convert field-centric planar velocity to robot-centric using robot yaw heading_rad.
    Pass result (vx_r, vy_r) into swerve solver if you want field-centric control.
    """
    ch, sh = math.cos(heading_rad), math.sin(heading_rad)
    # rotate by -heading
    vx_r =  vx_f * ch + vy_f * sh
    vy_r = -vx_f * sh + vy_f * ch
    return vx_r, vy_r


@dataclass
class ModuleState:
    """
    Per-module state/commands.
    x, y: module position (m) from robot center (+x fwd, +y left)
    angle_now: measured steering angle (rad), continuous or wrapped consistently
    speed_cmd: last commanded linear speed (m/s) (used if you enable slew limiting)
    """
    x: float
    y: float
    angle_now: float = 0.0
    angle_cmd: float = 0.0
    speed_cmd: float = 0.0
    angle_offset: float = 0.0  # optional absolute-encoder zero calibration (rad)

# ---------------------- class API ----------------------

class SwerveKinematics:
    """
    Robot-centric swerve solver.
    Use:
        kin = SwerveKinematics(modules, max_wheel_speed=4.5, max_wheel_accel=8.0)
        kin.solve(vx, vy, omega, dt)
        -> modules[i].angle_cmd / speed_cmd updated in-place
    """
    def __init__(self,
                 modules: List[ModuleState],
                 max_wheel_speed: float,
                 max_wheel_accel: float | None = None):
        self.modules = modules
        self.max_wheel_speed = max_wheel_speed
        self.max_wheel_accel = max_wheel_accel

    def solve(self, vx: float, vy: float, omega: float, dt: float = 0.0) -> None:
        """
        Compute per-module (angle_cmd, speed_cmd) for robot-centric (vx, vy, omega).
        vx, vy in m/s; omega in rad/s (CCW).
        """
        # 1) raw per-module twist
        tmp_angles: List[float] = []
        tmp_speeds: List[float] = []
        for m in self.modules:
            vx_i = vx - omega * m.y
            vy_i = vy + omega * m.x
            tmp_angles.append(math.atan2(vy_i, vx_i))
            tmp_speeds.append(math.hypot(vx_i, vy_i))

        # 2) desaturate uniformly
        _, tmp_speeds = desaturate_speeds(tmp_speeds, self.max_wheel_speed)

        # 3) zero-command hold (prevents jitter)
        zero_cmd = (abs(vx) + abs(vy) + abs(omega)) < 1e-6

        # 4) angle optimization + optional slew
        for idx, m in enumerate(self.modules):
            if zero_cmd:
                m.angle_cmd = m.angle_now   # hold
                m.speed_cmd = 0.0
                continue

            tgt_angle, tgt_speed = optimize_angle(tmp_angles[idx], m.angle_now, tmp_speeds[idx])

            if self.max_wheel_accel and dt > 0.0:
                tgt_speed = slew_limit(tgt_speed, m.speed_cmd, self.max_wheel_accel, dt)

            m.angle_cmd = tgt_angle
            m.speed_cmd = tgt_speed

# ---------------------- functional API ----------------------

def swerve_solve(vx: float,
                 vy: float,
                 omega: float,
                 modules: List[ModuleState],
                 max_wheel_speed: float,
                 dt: float = 0.0,
                 max_wheel_accel: float | None = None) -> None:
    """
    Functional solver (no class). Updates modules in-place.
    """
    # raw per-module velocities
    tmp_angles: List[float] = []
    tmp_speeds: List[float] = []
    for m in modules:
        vx_i = vx - omega * m.y
        vy_i = vy + omega * m.x
        tmp_angles.append(math.atan2(vy_i, vx_i))
        tmp_speeds.append(math.hypot(vx_i, vy_i))

    # desaturate
    _, tmp_speeds = desaturate_speeds(tmp_speeds, max_wheel_speed)

    zero_cmd = (abs(vx) + abs(vy) + abs(omega)) < 1e-6

    for i, m in enumerate(modules):
        if zero_cmd:
            m.angle_cmd = m.angle_now
            m.speed_cmd = 0.0
            continue
        angle, spd = optimize_angle(tmp_angles[i], m.angle_now, tmp_speeds[i])
        if max_wheel_accel and dt > 0.0:
            spd = slew_limit(spd, m.speed_cmd, max_wheel_accel, dt)
        m.angle_cmd = angle
        m.speed_cmd = spd
