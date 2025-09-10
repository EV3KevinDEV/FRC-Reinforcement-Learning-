# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# MODIFICATION: Replaced Isaac Lab SurfaceGripper with custom implementation using:
# - Fixed joints (PhysX) for physical attachment
# - Position constraints as fallback
# - Proximity detection based on intake position
# - Custom gripper state management (-1: open, 0: closing, 1: closed)

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.assets import SurfaceGripper, SurfaceGripperCfg
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.sensors import ContactSensor
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.sensors import ContactSensorCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
import isaaclab.utils.math as math_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# USD imports for prim manipulation
try:
    from pxr import Usd, UsdGeom
    USD_AVAILABLE = True
except ImportError:
    USD_AVAILABLE = False
    print("⚠️  USD libraries not available - prim operations will be limited")

# debugging orientation markers
def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 1.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)

import os
import sys
import re

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from util.swerve_helper import SwerveKinematics, ModuleState

from .frc_rl_env_cfg import FrcRlEnvCfg

# Regex patterns for reef welding
SOCKET_NAME_RE = re.compile(r"^blue_[A-L]_L[1-4]$", re.IGNORECASE) # Regex for socket names: blue_A_L4 format
SOCKET_FOLDER_RE = re.compile(r"^(BlueReefSensor)$", re.IGNORECASE) # Regex for socket parent folders
SOCKET_PATH_PATTERN = r"/(BlueReefSensor)/" # Pattern to match paths containing reef sensor folders

    # Configure Swerve Drive (for your implementation)
modules = [
    ModuleState( 0.29, 0.29),  # front-left
    ModuleState( 0.29, -0.29),  # front-right
    ModuleState(-0.29, 0.29),  # rear-left
    ModuleState(-0.29, -0.29),  # rear-right
]
max_wheel_speed = 4.0  # m/s
kin = SwerveKinematics(modules, max_wheel_speed, max_wheel_accel=3.0)
wheel_radius = 0.05  # meters

# Define joint groups (corrected indices)
swerve_turret_indices = [2, 3, 4, 5]  # dof_lturret_1, dof_lturret_2, dof_rturret_1, dof_rturret_2
wheel_indices = [7, 8, 9, 10]         # dof_lwheel_1, dof_lwheel_2, dof_rwheel_1, dof_rwheel_2
elevator_indices = [0, 1]             # dof_elevator_1, dof_elevator_2
intake_indices = [6]                  # dof_intake_1


# Zero Position (Level 0)
elevator_1_zero_position = 0.1
elevator_2_zero_position = 0
intake_zero_position = 0.1

# Level 1
elevator_1_level_1_position = 0.5349
elevator_2_level_1_position = 0.498983
intake_level_1_position = 1.83259571459  # 25 degrees in radians

# Level 2
elevator_1_level_2_position = 0.761153418
elevator_2_level_2_position = 0.712046746
intake_level_2_position = 0.6109  # 35 degrees in radians

# Level 3
elevator_1_level_3_position = 0.7874 
elevator_2_level_3_position = 0.7366
intake_level_3_position = 1.83259571459 # 25 degrees in radians



class FrcRlEnv(DirectRLEnv):
    cfg: FrcRlEnvCfg


    def __init__(self, cfg: FrcRlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # These will be initialized after the scene is setup
        self.joint_names = self.robot.joint_names  # Use joint_names instead of get_joint_names()
        
        # Define DOF indices for different robot parts
        self.dof_idx = torch.arange(self.robot.num_joints, device=self.device)
        

        
        # Debug: Print all joint names and their indices
        self._debug_print_joint_info()
        
        # Coral spawning timer and tracking
        self.coral_positions = []
        self.coral_count = 0
        
        # Cache for sensor positions to avoid repeated USD queries
        self._sensor_positions_cache = {}
        
        # Joint tracking for coral welding cleanup
        self.coral_joints = {}
        
        # FRC 2025 Reefscape scoring system
        self.frc_scoring = {
            'L1': 3,  # Level 1 sensors = 3 points
            'L2': 4,  # Level 2 sensors = 4 points  
            'L3': 5,  # Level 3 sensors = 5 points
            'L4': 6   # Level 4 sensors = 6 points
        }
        
        # Track total score achieved
        self.total_score = torch.zeros(self.num_envs, device=self.device)
        self.level_counts = {
            'L1': torch.zeros(self.num_envs, device=self.device),
            'L2': torch.zeros(self.num_envs, device=self.device), 
            'L3': torch.zeros(self.num_envs, device=self.device),
            'L4': torch.zeros(self.num_envs, device=self.device)
        }
        
        # Debug flag for observation size debugging
        self._debug_obs_printed = False
        
        # Track coral holding state for each environment
        self.is_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.previous_is_holding = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Track distances for reward calculations
        self.prev_coral_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.prev_socket_distance = torch.full((self.num_envs,), float('inf'), device=self.device)
        self.prev_reef_distance = torch.full((self.num_envs,), float('inf'), device=self.device)  # Track reef distance consistently
        
        # Track which coral each robot is holding (None if not holding)
        self.holding_coral_idx = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        
        # Store current swerve commands for observations
        self.current_swerve_commands = torch.zeros((self.num_envs, 3), device=self.device)  # [vx, vy, omega]
        
        # Initialize previous actions for latency compensation in simplified observations
        self.previous_actions = torch.zeros((self.num_envs, 5), device=self.device)
        
        # Phase-based reward system
        self.PHASES = {
            0: "SEEK",          # Don't have coral; roaming toward coral
            1: "INTAKE",        # Don't have coral; close enough for careful approach
            2: "CARRY_LONG",    # Have coral; far from reef (transit)
            3: "CARRY_MED",     # Have coral; mid-range to reef (approach tightening)
            4: "FINAL_ALIGN",   # Close to reef; aligning heading and setting elevator
            5: "DOCK",          # At reef face, aligned, ready to eject
            6: "RESET"          # Brief grace period after scoring
        }
        
        # Current phase for each environment
        self.current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # Reset timer for RESET phase
        self.reset_timer = torch.zeros(self.num_envs, device=self.device)
        self.reset_duration = 30  # Steps to stay in RESET phase after scoring
        
        # Phase thresholds
        self.INTAKE_DISTANCE = 1.2      # Switch SEEK -> INTAKE when coral within this distance
        self.CARRY_MED_DISTANCE = 3.0   # Switch CARRY_LONG -> CARRY_MED when reef within this distance
        self.FINAL_ALIGN_DISTANCE = 1.0 # Switch CARRY_MED -> FINAL_ALIGN when reef within this distance
        self.DOCK_DISTANCE = 0.3        # Switch FINAL_ALIGN -> DOCK when reef within this distance
        self.ALIGNMENT_THRESHOLD = 0.8  # Dot product for "good alignment"
        self.ELEVATOR_READY_THRESHOLD = 0.2  # Radians for "elevator at target height"

    def _debug_print_joint_info(self):
        """Debug function to print all joint names and their indices."""
        print("\n" + "="*60)
        print("ROBOT JOINT DEBUG INFORMATION")
        print("="*60)
        print(f"Total number of joints: {self.robot.num_joints}")
        print(f"Robot name: {self.robot.cfg.prim_path}")
        print("-"*60)
        
        for i, joint_name in enumerate(self.robot.joint_names):
            print(f"Index {i:2d}: {joint_name}")
        
        print("-"*60)
        print("CURRENT JOINT INDEX ASSIGNMENTS:")
        print(f"Elevator indices     : {elevator_indices}")
        print(f"Swerve turret indices: {swerve_turret_indices}")
        print(f"Intake indices       : {intake_indices}")
        print(f"Wheel indices        : {wheel_indices}")
        print("="*60)
        
        # Validate that indices don't exceed joint count
        all_indices = elevator_indices + swerve_turret_indices + intake_indices + wheel_indices
        max_index = max(all_indices) if all_indices else -1
        if max_index >= self.robot.num_joints:
            print(f"⚠️  WARNING: Maximum index ({max_index}) exceeds joint count ({self.robot.num_joints})!")
        else:
            print(f"✅ All indices are valid (max: {max_index}, joint count: {self.robot.num_joints})")
        print("="*60 + "\n")


    def attach_objects(self, end_effector_prim, box_prim, local_pose_ee=None, local_pose_box=None):
        """Attach objects using PhysX joints - requires Isaac Sim runtime."""
        try:
            # This requires running within Isaac Sim where omni.physx is available
            from omni.physx import get_physx_scene_interface

            physx_scene = get_physx_scene_interface()

            # Use default poses if not provided
            if local_pose_ee is None:
                local_pose_ee = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Identity pose
            if local_pose_box is None:
                local_pose_box = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # Identity pose

            joint_prim = physx_scene.create_fixed_joint(
                prim_path="/World/Joints/Attachment",
                body0=end_effector_prim.GetPath(),
                body1=box_prim.GetPath(),
                local_pose0=local_pose_ee,
                local_pose1=local_pose_box,
                enable_collision=False  # Disable collision between attached bodies
            )
            print(f"🔗 Created PhysX joint between {end_effector_prim.GetPath()} and {box_prim.GetPath()}")
            return joint_prim
        except ImportError:
            print("⚠️  omni.physx not available - using fallback attachment method")
            # Fallback: implement attachment using Isaac Lab APIs
            return self._attach_objects_fallback(end_effector_prim, box_prim, local_pose_ee, local_pose_box)

    def _attach_objects_fallback(self, end_effector_prim, box_prim, local_pose_ee=None, local_pose_box=None):
        """Fallback attachment method using Isaac Lab APIs when omni.physx is not available."""
        if not USD_AVAILABLE:
            print("⚠️  USD libraries not available - cannot perform attachment")
            return None

        try:
            print(f"🔗 Attaching {box_prim.GetPath()} to {end_effector_prim.GetPath()} using fallback method")

            # Get positions of both prims
            ee_pos = end_effector_prim.GetAttribute('xformOp:translate').Get()
            box_pos = box_prim.GetAttribute('xformOp:translate').Get()

            # Calculate relative transformation
            if local_pose_ee is not None and len(local_pose_ee) >= 3:
                # Apply local pose offset to end effector position
                ee_pos = [ee_pos[i] + local_pose_ee[i] for i in range(3)]

            # Move box to end effector position
            box_prim.GetAttribute('xformOp:translate').Set(ee_pos)

            print(f"✅ Attached {box_prim.GetPath()} to {end_effector_prim.GetPath()} via position")
            return True
        except Exception as e:
            print(f"❌ Failed to attach objects: {e}")
            return None

    def detect_reef_welding(self):
        """Detect when corals are near BLUE reef sensors and create welds using attach_objects."""
        if not hasattr(self, 'coral_objects') or not hasattr(self.coral_objects.data, 'root_pos_w'):
            return

        # Initialize sensor occupancy tracking if not exists
        if not hasattr(self, 'sensor_coral_mapping'):
            self.sensor_coral_mapping = {}  # Maps (env_idx, sensor_name) -> coral_idx

        welding_events = []

        # Get all coral positions
        coral_positions = self.coral_objects.data.root_pos_w
        coral_orientations = self.coral_objects.data.root_quat_w

        for env_idx in range(self.num_envs):
            # Find corals in this environment
            corals_per_env = len(self.coral_prim_paths) // self.num_envs
            start_idx = env_idx * corals_per_env
            end_idx = start_idx + corals_per_env



            for coral_idx in range(start_idx, min(end_idx, len(coral_positions))):
                coral_pos = coral_positions[coral_idx]

                # Check proximity to reef sensors
                nearest_sensor, distance = self._find_nearest_reef_sensor(coral_pos, env_idx)

                if nearest_sensor and distance < 0.15:  # 15cm threshold for welding
                    # Check if this sensor already has a coral attached
                    sensor_key = (env_idx, nearest_sensor)
                    existing_coral = self.sensor_coral_mapping.get(sensor_key)
                    
                    # Check if this coral is already welded to any sensor
                    coral_already_welded = any(
                        coral_idx == mapped_coral 
                        for (env, sensor), mapped_coral in self.sensor_coral_mapping.items() 
                        if env == env_idx
                    )
                    
                    if existing_coral is not None:
                        # Sensor already occupied - silently reject
                        continue
                    
                    if coral_already_welded:
                        # This coral is already welded to another sensor - silently reject
                        continue

                    # Create weld between coral and sensor using attach_objects
                    weld_success = self._create_coral_weld(coral_idx, nearest_sensor, env_idx)
                    if weld_success:
                        # Mark this sensor as occupied by this coral
                        self.sensor_coral_mapping[sensor_key] = coral_idx
                        
                        # Update FRC scoring based on sensor level
                        sensor_level = self._get_sensor_level(nearest_sensor)
                        points = self.frc_scoring.get(sensor_level, 0)
                        self.total_score[env_idx] += points
                        self.level_counts[sensor_level][env_idx] += 1
                        
                        welding_events.append({
                            'env_idx': env_idx,
                            'coral_idx': coral_idx,
                            'sensor': nearest_sensor,
                            'distance': distance,
                            'level': sensor_level,
                            'points': points
                        })

        return welding_events

    def _get_sensor_level(self, sensor_name: str) -> str:
        """Extract sensor level (L1, L2, L3, L4) from sensor name like 'blue_A_L4'."""
        if '_L1' in sensor_name:
            return 'L1'
        elif '_L2' in sensor_name:
            return 'L2'
        elif '_L3' in sensor_name:
            return 'L3'
        elif '_L4' in sensor_name:
            return 'L4'
        else:
            return 'L1'  # Default fallback



    def inspect_usd_scene(self, env_idx=0, max_depth=3):
        """Inspect the USD scene structure to debug sensor detection issues."""
        print("\n" + "="*80)
        print(f"🔍 USD SCENE INSPECTION - Environment {env_idx}")
        print("="*80)

        if not USD_AVAILABLE:
            print("❌ USD libraries not available - cannot inspect scene")
            return

        try:
            from isaaclab.sim import SimulationContext
            sim_context = SimulationContext.instance()
            stage = sim_context.stage

            def print_prim_hierarchy(prim_path, depth=0, max_depth=max_depth):
                """Recursively print prim hierarchy."""
                if depth > max_depth:
                    return
                
                prim = stage.GetPrimAtPath(prim_path)
                if not prim.IsValid():
                    print("  " * depth + f"❌ Invalid prim: {prim_path}")
                    return

                prim_type = prim.GetTypeName()
                indent = "  " * depth
                print(f"{indent}📁 {prim.GetName()} ({prim_type}) - {prim_path}")

                # Print attributes for sensor-like prims
                if "sensor" in prim.GetName().lower() or SOCKET_NAME_RE.match(prim.GetName()):
                    if prim.HasAttribute('xformOp:translate'):
                        pos = prim.GetAttribute('xformOp:translate').Get()
                        print(f"{indent}   📍 Position: {pos}")

                # Recursively print children
                for child in prim.GetChildren():
                    print_prim_hierarchy(child.GetPath(), depth + 1, max_depth)

            # Start from the world level
            world_paths = [
                "/World",
                f"/World/envs/env_{env_idx}",
                f"/World/envs/env_{env_idx}/reefscape"
            ]

            for path in world_paths:
                print(f"\n🔍 Inspecting: {path}")
                print_prim_hierarchy(path, 0, max_depth)

        except Exception as e:
            print(f"❌ Error inspecting USD scene: {e}")
            import traceback
            traceback.print_exc()

        print("="*80 + "\n")

    def inspect_scene(self, env_idx=0):
        """Public method to inspect the USD scene structure for debugging."""
        self.inspect_usd_scene(env_idx)

    def _find_nearest_reef_sensor(self, coral_pos, env_idx):
        """Find the nearest BLUE reef sensor to a coral position using actual USD scene data."""
        # Implementation now uses actual sensor positions from the reefscape USD
        # Focuses on blue sensors only as requested
        # Uses the path: /World/envs/env_X/reefscape/REEFSCAPE_FIELD__FE_2025___1_/BlueReefSensor/

        sensor_positions = self._get_reef_sensor_positions(env_idx)

        min_distance = float('inf')
        nearest_sensor = None

        for sensor_name, sensor_pos in sensor_positions.items():
            # Only consider blue sensors (already filtered in _get_reef_sensor_positions)
            if sensor_name.startswith('blue'):
                distance = torch.norm(coral_pos - sensor_pos)
                if distance < min_distance:
                    min_distance = distance
                    nearest_sensor = sensor_name

        return nearest_sensor, min_distance.item() if nearest_sensor else float('inf')

    def _get_reef_sensor_positions(self, env_idx):
        """Get positions of all reef sensors in the environment from the actual USD scene."""
        # Check cache first
        if env_idx in self._sensor_positions_cache:
            return self._sensor_positions_cache[env_idx]

        sensor_positions = {}
        
        print(f"🔍 [DEBUG] Detecting sensors for environment {env_idx}")
        print(f"🔍 [DEBUG] USD_AVAILABLE = {USD_AVAILABLE}")

        if not USD_AVAILABLE:
            print("⚠️  [DEBUG] USD libraries not available - using placeholder sensor positions")
            # Fallback to placeholder positions
            placeholder_positions = [
                ("blue_A_L1", torch.tensor([-2.0, 2.0, 0.5], device=self.device)),
                ("blue_B_L1", torch.tensor([-1.0, 2.0, 0.5], device=self.device)),
                ("blue_C_L1", torch.tensor([0.0, 2.0, 0.5], device=self.device)),
                ("blue_D_L1", torch.tensor([1.0, 2.0, 0.5], device=self.device)),
            ]
            env_origin = self.scene.env_origins[env_idx]
            for sensor_name, pos in placeholder_positions:
                sensor_positions[sensor_name] = pos + env_origin
            
            print(f"✅ [DEBUG] Created {len(sensor_positions)} placeholder sensors")
            # Cache the result
            self._sensor_positions_cache[env_idx] = sensor_positions
            return sensor_positions

        try:
            print("🔍 [DEBUG] Attempting to access USD stage...")
            # Get the simulation stage
            from isaaclab.sim import SimulationContext
            sim_context = SimulationContext.instance()
            stage = sim_context.stage
            print("✅ [DEBUG] Successfully got simulation stage")

            # Query the actual BlueReefSensor folder
            blue_sensor_base_path = f"/World/envs/env_{env_idx}/reefscape/REEFSCAPE_FIELD__FE_2025___1_/BlueReefSensor"
            print(f"🔍 [DEBUG] Looking for BlueReefSensor at: {blue_sensor_base_path}")

            # Get all children of the BlueReefSensor folder
            blue_sensor_prim = stage.GetPrimAtPath(blue_sensor_base_path)
            if not blue_sensor_prim.IsValid():
                print(f"❌ [DEBUG] BlueReefSensor folder not found at: {blue_sensor_base_path}")
                print(f"🔍 [DEBUG] Expected sensor names: blue_[A-L]_L[1-4] (e.g., blue_A_L4)")
                print("🔍 [DEBUG] Available prims at reefscape level:")
                reefscape_path = f"/World/envs/env_{env_idx}/reefscape"
                reefscape_prim = stage.GetPrimAtPath(reefscape_path)
                if reefscape_prim.IsValid():
                    for child in reefscape_prim.GetChildren():
                        print(f"   - {child.GetName()}: {child.GetPath()}")
                else:
                    print(f"   ❌ Reefscape folder not found at: {reefscape_path}")
                
                # Inspect the scene structure to help debug
                print("\n🔍 [DEBUG] Inspecting USD scene structure to help debug sensor detection...")
                self.inspect_usd_scene(env_idx, max_depth=4)
                
                return sensor_positions

            print("✅ [DEBUG] Found BlueReefSensor folder")
            
            # Iterate through all child prims (these should be the individual sensors)
            child_count = 0
            for child_prim in blue_sensor_prim.GetChildren():
                child_count += 1
                sensor_name = child_prim.GetName()
                print(f"🔍 [DEBUG] Checking child {child_count}: {sensor_name}")

                # Check if this matches our sensor naming pattern
                if SOCKET_NAME_RE.match(sensor_name):
                    print(f"✅ [DEBUG] Sensor name matches pattern: {sensor_name}")
                    # Get the sensor's world position
                    if child_prim.HasAttribute('xformOp:translate'):
                        local_pos = child_prim.GetAttribute('xformOp:translate').Get()
                        print(f"📍 [DEBUG] Local position for {sensor_name}: {local_pos}")

                        # Convert to tensor and add environment offset
                        env_origin = self.scene.env_origins[env_idx]
                        world_pos = torch.tensor(local_pos, device=self.device) + env_origin

                        sensor_positions[sensor_name] = world_pos
                        print(f"📍 [DEBUG] World position for {sensor_name}: {world_pos}")
                    else:
                        print(f"⚠️  [DEBUG] Sensor '{sensor_name}' has no position attribute")
                else:
                    print(f"❌ [DEBUG] Sensor name does not match pattern: {sensor_name}")
            
            print(f"✅ [DEBUG] Found {len(sensor_positions)} valid blue sensors out of {child_count} total children")

        except Exception as e:
            print(f"❌ [DEBUG] Error querying sensor positions: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to placeholder positions
            print("🔄 [DEBUG] Falling back to placeholder sensor positions")
            placeholder_positions = [
                ("blue_A_L1", torch.tensor([-2.0, 2.0, 0.5], device=self.device)),
                ("blue_B_L1", torch.tensor([-1.0, 2.0, 0.5], device=self.device)),
                ("blue_C_L1", torch.tensor([0.0, 2.0, 0.5], device=self.device)),
                ("blue_D_L1", torch.tensor([1.0, 2.0, 0.5], device=self.device)),
            ]
            env_origin = self.scene.env_origins[env_idx]
            for sensor_name, pos in placeholder_positions:
                sensor_positions[sensor_name] = pos + env_origin

        # Cache the result
        self._sensor_positions_cache[env_idx] = sensor_positions
        print(f"💾 [DEBUG] Cached sensor positions for environment {env_idx}")
        return sensor_positions

    def _create_coral_weld(self, coral_idx, sensor_name, env_idx):
        """Create a weld between a coral and a reef sensor using attach_objects."""
        try:
            # Initialize joint tracking if not exists
            if not hasattr(self, 'coral_joints'):
                self.coral_joints = {}
            
            # Get coral and sensor prims
            coral_prim = self._get_coral_prim(coral_idx)
            sensor_prim = self._get_sensor_prim(sensor_name, env_idx)

            if coral_prim is None or sensor_prim is None:
                print(f"⚠️  Could not find prims for coral {coral_idx} or sensor {sensor_name}")
                return False

            # Use the attach_objects method to create the weld
            joint_result = self.attach_objects(sensor_prim, coral_prim)

            if joint_result is not None:
                # Track the created joint with detailed information for proper detachment
                joint_key = f"coral_{coral_idx}"
                
                # Store detailed joint information for detach_objects method
                if hasattr(joint_result, 'GetPath'):
                    # Store joint prim and related prims for detach_objects
                    self.coral_joints[joint_key] = {
                        'joint_prim': joint_result,
                        'coral_prim': coral_prim,
                        'sensor_prim': sensor_prim,
                        'sensor_name': sensor_name,
                        'env_idx': env_idx
                    }
                else:
                    # Fallback for boolean return values
                    self.coral_joints[joint_key] = joint_result
                
                return True
            else:
                return False

        except Exception as e:
            return False

    def _get_coral_prim(self, coral_idx):
        """Get the USD prim for a coral object."""
        try:
            if coral_idx < len(self.coral_prim_paths):
                coral_path = self.coral_prim_paths[coral_idx]

                if USD_AVAILABLE:
                    # Get the simulation stage
                    from isaaclab.sim import SimulationContext
                    sim_context = SimulationContext.instance()
                    stage = sim_context.stage
                    prim = stage.GetPrimAtPath(coral_path)

                    if prim.IsValid():
                        return prim
                    else:
                        # Try to find the actual coral mesh/geometry prim
                        # Often the physics body is at the root but the actual mesh is nested deeper
                        coral_children = stage.GetPrimAtPath(coral_path)
                        if coral_children.IsValid():
                            for child in coral_children.GetAllChildren():
                                child_path = child.GetPath()
                                if "Coral" in str(child_path) or "CORAL" in str(child_path):
                                    return child
                        return None
                else:
                    return None
            else:
                return None
        except Exception as e:
            return None

    def _get_sensor_prim(self, sensor_name, env_idx):
        """Get the USD prim for a reef sensor."""
        try:
            # Construct sensor path using the actual reefscape structure
            if sensor_name.startswith("blue_"):
                # All sensors we're working with are blue sensors, so just construct the path directly
                sensor_path = f"/World/envs/env_{env_idx}/reefscape/REEFSCAPE_FIELD__FE_2025___1_/BlueReefSensor/{sensor_name}"

                if USD_AVAILABLE:
                    # Get the simulation stage
                    from isaaclab.sim import SimulationContext
                    sim_context = SimulationContext.instance()
                    stage = sim_context.stage
                    prim = stage.GetPrimAtPath(sensor_path)

                    if prim.IsValid():
                        return prim
                    else:
                        return None
                else:
                    return None
            else:
                return None
        except Exception as e:
            print(f"Error getting sensor prim: {e}")
            return None
    

    def _setup_scene(self):
        # create robot
        self.robot = Articulation(self.cfg.robot_cfg)

        # Surface Gripper Configuration
        self.surface_gripper_cfg = SurfaceGripperCfg()
        self.surface_gripper_cfg.prim_expr = "/World/envs/env_.*/MasterRCSH/MasterRSCH/SurfaceGripper"
        self.surface_gripper_cfg.max_grip_distance = 0.25  # [m] (Maximum distance at which the gripper can grasp an object)
        self.surface_gripper_cfg.shear_force_limit = 50.0  # [N] (Force limit in the direction perpendicular direction)
        self.surface_gripper_cfg.coaxial_force_limit = 50.0  # [N] (Force limit in the direction of the gripper's axis)
        self.surface_gripper_cfg.retry_interval = 3  # seconds (Time the gripper will stay in a grasping state)
        self.surface_gripper = SurfaceGripper(cfg=self.surface_gripper_cfg)


        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Reefscape field & corals: author ONLY in env_0, then clone so others copy env_0
        cfg_reefscape = sim_utils.UsdFileCfg(
            usd_path="source/FRC_RL/assets/Reefscape_Field.usd",
        )
        cfg_coral = sim_utils.UsdFileCfg(usd_path="source/FRC_RL/assets/Coral.usd")

        # Spawn reefscape in env_0
        sim_utils.spawn_from_usd(prim_path="/World/envs/env_0/reefscape", cfg=cfg_reefscape)

        # Spawn corals only in env_0 (they will be cloned to other envs)
        for c in range(self.cfg.max_corals):
            prim_path = f"/World/envs/env_0/coral_{c}"
            random_x = random.uniform(-3, 3.0)
            random_y = random.uniform(9, 8.0)
            z = 0.5
            sim_utils.spawn_from_usd(
                prim_path=prim_path,
                cfg=cfg_coral,
                translation=(random_x, random_y, z),
            )

        # Clone env_0 -> env_1..N (copies reefscape + corals)
        self.scene.clone_environments(copy_from_source=False)

        # Build full coral prim path list (now that clones exist)
        self.coral_prim_paths = []
        for env_id in range(self.num_envs):
            for c in range(self.cfg.max_corals):
                self.coral_prim_paths.append(f"/World/envs/env_{env_id}/coral_{c}")

        # Register corals as rigid objects with collisions across all envs
        self.coral_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/coral_.*")
        self.coral_objects = RigidObject(self.coral_cfg)
        self.scene.rigid_objects["corals"] = self.coral_objects


        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.surface_grippers["gripper"] = self.surface_gripper

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        self.visualization_markers = define_markers()

        self.marker_locations = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset = torch.zeros((self.num_envs, 3), device=self.device)
        self.marker_offset[:, -1] = 0.5
        self.forward_marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        # self.command_marker_orientations = torch.zeros((self.num_envs, 4), device=self.device)

        #Sensors

        # Raycast sensor for environment perception
        # ray_caster_cfg = RayCasterCfg(
        #     prim_path="/World/envs/env_.*/MasterRCSH",
        #     update_period=1 / 60,
        #     offset=RayCasterCfg.OffsetCfg(pos=(0, 0, 0.5)),
        #     ray_alignment="yaw",
        #     pattern_cfg=patterns.LidarPatternCfg(
        #         channels=100, vertical_fov_range=[-90, 90], horizontal_fov_range=[-90, 90], horizontal_res=1.0
        #     ),
        #     mesh_prim_paths=["/World/envs/env_.*/reefscape"],  # Detect the reefscape field
        #     debug_vis=True
        # )
        #Intake Contact Sensor Configuration
        intake_contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/MasterRCSH/MasterRSCH/Gripper/Intake_Piece",
            update_period=0.0,
            history_length=6,
            debug_vis=True,
            filter_prim_paths_expr=["/World/envs/env_.*/coral_.*"],
        )
        #raycast sensor for collision detection
        # self.raycast_sensor = RayCaster(cfg=ray_caster_cfg)        

        #Intake Contact Sensor
        self.intake_contact_sensor = ContactSensor(cfg=intake_contact_sensor_cfg)

        # self.scene.sensors["raycast"] = self.raycast_sensor
        self.scene.sensors["intake_contact"] = self.intake_contact_sensor


    def _visualize_markers(self):
        # Use intake piece positions and orientations for visualization
        intake_positions = torch.zeros((self.num_envs, 3), device=self.device)
        intake_orientations = torch.zeros((self.num_envs, 4), device=self.device)
        
        for env_idx in range(self.num_envs):
            intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
            if intake_pos is not None and intake_quat is not None:
                intake_positions[env_idx] = intake_pos
                intake_orientations[env_idx] = intake_quat
            # If intake not available, leave as zeros (default tensor initialization)
        
        self.marker_locations = intake_positions
        self.forward_marker_orientations = intake_orientations
        # self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()

        loc = self.marker_locations + self.marker_offset
        rots = self.forward_marker_orientations

        all_envs = torch.arange(self.num_envs)
        indices = torch.zeros_like(all_envs)  # Only use forward markers (index 0)

        self.visualization_markers.visualize(loc, rots, marker_indices=indices)
        

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Store previous actions for latency compensation
        if hasattr(self, 'actions'):
            self.previous_actions = self.actions.clone()
        else:
            self.previous_actions = torch.zeros((self.num_envs, 5), device=self.device)
        
        self.actions = actions.clone()
        self._visualize_markers()

        # Detect and create coral welds when near sensors
        welding_events = self.detect_reef_welding()


   
    def _apply_action(self) -> None:
        # Ensure actions are on the correct device
        actions = self.actions.to(device=self.device)

        # Swerve Drive Control - handle each environment
        for env_idx in range(self.num_envs):
            # Extract swerve actions for this environment
            vx = actions[env_idx, 0].item() * max_wheel_speed
            vy = actions[env_idx, 1].item() * max_wheel_speed
            omega = actions[env_idx, 2].item() * math.pi * 2  # Scale to reasonable angular velocity

            # Store swerve commands for observations
            self.current_swerve_commands[env_idx] = torch.tensor([vx, vy, omega], device=self.device)

            # Use swerve kinematics to solve for wheel commands
            # Pass through lateral (vy) and angular (omega) commands so the
            # robot can crab/strafe and rotate rather than being forced forward.
            kin.solve(vx, vy, omega, dt=1/120)  # Use configured timestep

            # Get commands from modules
            wheel_speeds_ms = [modules[0].speed_cmd, modules[2].speed_cmd, modules[3].speed_cmd, modules[1].speed_cmd]
            swerve_positions = [modules[0].angle_cmd, modules[2].angle_cmd, modules[3].angle_cmd, modules[1].angle_cmd]
            
            
            # Convert linear speed (m/s) to angular velocity (rad/s)
            wheel_velocities = [speed / wheel_radius for speed in wheel_speeds_ms]
            
            # Clamp wheel velocities to prevent extreme values
            # No clamping of wheel velocities

            # Create tensors for this environment
            wheel_vel_tensor = torch.tensor(wheel_velocities, device=self.device).unsqueeze(0)
            swerve_pos_tensor = torch.tensor(swerve_positions, device=self.device).unsqueeze(0)
            
            # Create proper env_ids tensor
            env_ids_tensor = torch.tensor([env_idx], device=self.device)
            
            # Apply to robot joints for this specific environment
            self.robot.set_joint_velocity_target(wheel_vel_tensor, joint_ids=wheel_indices, env_ids=env_ids_tensor)
            self.robot.set_joint_position_target(swerve_pos_tensor, joint_ids=swerve_turret_indices, env_ids=env_ids_tensor)

        # Elevator and Intake Actions from neural network (discrete)
        # Use per-env column (vectorized) selection
        level_action = actions[:, 3]  # Single level control action per env

        # Normalize action from [-1, 1] to [0, 1]
        level_norm = (level_action + 1) / 2

        # Convert to discrete levels using specific ranges (vectorized)
        # Level 0: 0.0 to 0.25, Level 1: 0.25 to 0.5, Level 2: 0.5 to 0.75, Level 3: 0.75 to 1.0
        level = torch.where(level_norm < 0.25, 0,
                torch.where(level_norm < 0.5, 1,
                torch.where(level_norm < 0.75, 2, 3)))
        
        # ✅ Add bounds checking to prevent index out of bounds
        level = torch.clamp(level, 0, 3)

        # Position lookup tables
        elevator_positions = torch.tensor([
            [elevator_1_zero_position, elevator_2_zero_position],      # Level 0
            [elevator_1_level_1_position, elevator_2_level_1_position], # Level 1
            [elevator_1_level_2_position, elevator_2_level_2_position], # Level 2
            [elevator_1_level_3_position, elevator_2_level_3_position]  # Level 3
        ], device=self.device)

        intake_positions = torch.tensor([
            intake_zero_position,      # Level 0
            intake_level_1_position,   # Level 1
            intake_level_2_position,   # Level 2
            intake_level_3_position    # Level 3
        ], device=self.device)

        # Use the same level for both elevator and intake
        elevator_targets = elevator_positions[level]
        intake_targets = intake_positions[level].unsqueeze(1)

        # Clamp only intake targets to PhysX joint limits (elevators are linear slides, not rotational)
        intake_targets = torch.clamp(intake_targets, 0.2, 2*math.pi)

        # Apply targets to robot joints
        self.robot.set_joint_position_target(elevator_targets, joint_ids=elevator_indices)
        self.robot.set_joint_position_target(intake_targets, joint_ids=intake_indices)

        # Surface Gripper Actions - only allow when coral is contacted
        gripper_commands = actions[:, -1]  # Last action for gripper
        gripper_commands = torch.clamp(gripper_commands, -1.0, 1.0)  # Ensure commands are within [-1, 1]
        
        # Check if coral is contacted by the intake contact sensor
        coral_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Get contact sensor data - check if any contact forces are detected
        # ✅ Add validation for contact sensor data
        if hasattr(self.intake_contact_sensor, 'data') and hasattr(self.intake_contact_sensor.data, 'net_forces_w'):
            contact_forces = self.intake_contact_sensor.data.net_forces_w
            # Check if contact force magnitude is above threshold (indicating coral contact)
            force_magnitude = torch.norm(contact_forces, dim=-1)  # Shape: (num_envs, num_contacts)
            coral_contacted = torch.any(force_magnitude > 0.5, dim=-1)  # Shape: (num_envs,) - True if any contact detected

        # Only allow gripper commands when coral is contacted
        # Both gripper_commands and coral_contacted have shape [num_envs]
        filtered_gripper_commands = torch.where(
            coral_contacted,  # Shape: [num_envs]
            gripper_commands,  # Shape: [num_envs] - Use actual commands when coral contacted
            torch.zeros_like(gripper_commands)  # Shape: [num_envs] - Keep gripper idle when no contact
        )
        
        # Set gripper commands directly (Surface Gripper API)
        # Commands: -1 to -0.3 = Opening, -0.3 to 0.3 = Idle, 0.3 to 1 = Closing
        # filtered_gripper_commands already has shape [num_envs]
        self.surface_gripper.set_grippers_command(filtered_gripper_commands)
        self.surface_gripper.write_data_to_sim()
        
        # ✅ Removed duplicate detect_reef_welding() call - already called in _pre_physics_step()

    def _get_observations(self) -> dict:
        # Update surface gripper state from simulation
        sim_dt = 1/120  # Default to 120 FPS
        self.surface_gripper.update(sim_dt)
        
        # Simplified observation system based on user specifications
        
        # 1. Robot-frame motion: forward/lateral velocity and yaw rate (3 dimensions)
        robot_motion = self._get_robot_motion_obs()
        
        # 2. Goal-relative geometry: switches based on phase (3 dimensions)
        goal_geometry = self._get_goal_geometry_obs()
        
        # 3. Mechanism readiness: has_piece, elevator error, at_setpoint (3 dimensions)
        mechanism_readiness = self._get_mechanism_readiness_obs()
        
        # 4. Time context: normalized time remaining (1 dimension)
        time_context = self._get_time_context_obs()
        
        # 5. Latency compensation: previous action (5 dimensions)
        latency_compensation = self._get_latency_compensation_obs()
        
        # Combined simplified observation: 3 + 3 + 3 + 1 + 5 = 15 dimensions
        obs = torch.hstack((robot_motion, goal_geometry, mechanism_readiness, time_context, latency_compensation))
        
        self._debug_simplified_observations(obs, robot_motion, goal_geometry, mechanism_readiness, time_context, latency_compensation)
        
        observations = {"policy": obs}
        return observations

    def _get_robot_motion_obs(self) -> torch.Tensor:
        """Get robot-frame motion: forward/lateral velocity and yaw rate (3 dimensions)."""
        motion_obs = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Get robot velocity in world frame
        robot_vel_world = self.robot.data.root_com_vel_w[:, :3]  # Linear velocity only
        robot_angular_vel_world = self.robot.data.root_com_vel_w[:, 3:]  # Angular velocity
        
        # Get robot orientation to transform to robot frame
        robot_quat = self.robot.data.root_quat_w
        robot_quat_inv = math_utils.quat_conjugate(robot_quat)
        
        # Transform linear velocity to robot frame
        robot_vel_local = math_utils.quat_apply(robot_quat_inv, robot_vel_world)
        
        # Forward velocity is Y-axis in robot frame (intake forward direction)  
        forward_vel = robot_vel_local[:, 1]  # Y-axis
        lateral_vel = robot_vel_local[:, 0]  # X-axis
        yaw_rate = robot_angular_vel_world[:, 2]  # Z-axis angular velocity
        
        motion_obs[:, 0] = forward_vel
        motion_obs[:, 1] = lateral_vel  
        motion_obs[:, 2] = yaw_rate
        
        return motion_obs
    
    def _get_goal_geometry_obs(self) -> torch.Tensor:
        """Get goal-relative geometry switching by phase: coral when empty, reef when carrying (3 dimensions)."""
        geometry_obs = torch.zeros((self.num_envs, 3), device=self.device)
        
        for env_idx in range(self.num_envs):
            # Get intake position and orientation for reference frame
            intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
            if intake_pos is None or intake_quat is None:
                continue
                
            if self.is_holding[env_idx]:
                # When carrying: use reef sensor vector
                target_pos = self._get_nearest_reef_sensor_position(env_idx)
                if target_pos is not None:
                    # Vector from intake to target
                    target_vector_world = target_pos - intake_pos
                    distance = torch.norm(target_vector_world)
                    
                    # Transform to intake frame for bearing
                    intake_quat_inv = math_utils.quat_conjugate(intake_quat.unsqueeze(0)).squeeze(0)
                    target_vector_local = math_utils.quat_apply(intake_quat_inv.unsqueeze(0), target_vector_world.unsqueeze(0)).squeeze(0)
                    
                    # Calculate angle error (bearing)
                    angle_to_target = torch.atan2(target_vector_local[0], target_vector_local[1])  # X/Y for bearing
                    
                    geometry_obs[env_idx, 0] = distance
                    geometry_obs[env_idx, 1] = torch.sin(angle_to_target)
                    geometry_obs[env_idx, 2] = torch.cos(angle_to_target)
            else:
                # When empty: use coral vector
                coral_pos = self._get_nearest_coral_position(env_idx)
                if coral_pos is not None:
                    # Vector from intake to coral
                    coral_vector_world = coral_pos - intake_pos
                    distance = torch.norm(coral_vector_world)
                    
                    # Transform to intake frame for bearing
                    intake_quat_inv = math_utils.quat_conjugate(intake_quat.unsqueeze(0)).squeeze(0)
                    coral_vector_local = math_utils.quat_apply(intake_quat_inv.unsqueeze(0), coral_vector_world.unsqueeze(0)).squeeze(0)
                    
                    # Calculate angle error (bearing)
                    angle_to_coral = torch.atan2(coral_vector_local[0], coral_vector_local[1])  # X/Y for bearing
                    
                    geometry_obs[env_idx, 0] = distance
                    geometry_obs[env_idx, 1] = torch.sin(angle_to_coral)
                    geometry_obs[env_idx, 2] = torch.cos(angle_to_coral)
        
        return geometry_obs
    
    def _get_mechanism_readiness_obs(self) -> torch.Tensor:
        """Get mechanism readiness: has_piece, elevator error, at_setpoint (3 dimensions)."""
        mechanism_obs = torch.zeros((self.num_envs, 3), device=self.device)
        
        # Binary has_piece flag
        mechanism_obs[:, 0] = self.is_holding.float()
        
        # Normalized elevator error to target (average of both elevators)
        joint_pos = self.robot.data.joint_pos
        elevator_positions = joint_pos[:, [0, 1]]  # Elevator joint indices
        
        # Get target elevator positions based on current action level
        if hasattr(self, 'actions'):
            level_action = self.actions[:, 3]  # Level control action
            level_norm = (level_action + 1) / 2
            level = torch.where(level_norm < 0.25, 0,
                    torch.where(level_norm < 0.5, 1,
                    torch.where(level_norm < 0.75, 2, 3)))
            level = torch.clamp(level, 0, 3)
            
            # Position lookup for elevator targets
            elevator_positions_lookup = torch.tensor([
                [elevator_1_zero_position, elevator_2_zero_position],      # Level 0
                [elevator_1_level_1_position, elevator_2_level_1_position], # Level 1
                [elevator_1_level_2_position, elevator_2_level_2_position], # Level 2
                [elevator_1_level_3_position, elevator_2_level_3_position]  # Level 3
            ], device=self.device)
            
            elevator_targets = elevator_positions_lookup[level]
            
            # Normalized error (average absolute error divided by max possible range)
            elevator_error = torch.abs(elevator_positions - elevator_targets).mean(dim=1)
            max_elevator_range = max(elevator_1_level_3_position - elevator_1_zero_position,
                                   elevator_2_level_3_position - elevator_2_zero_position)
            normalized_error = elevator_error / max_elevator_range
            mechanism_obs[:, 1] = normalized_error
            
            # At setpoint flag (within tolerance)
            setpoint_tolerance = 0.05  # 5cm tolerance
            at_setpoint = (elevator_error < setpoint_tolerance).float()
            mechanism_obs[:, 2] = at_setpoint
        
        return mechanism_obs
    
    def _get_time_context_obs(self) -> torch.Tensor:
        """Get normalized time remaining (1 dimension)."""
        # Calculate time remaining based on episode progress
        time_remaining = (self.max_episode_length - self.episode_length_buf) / self.max_episode_length
        return time_remaining.unsqueeze(1)
    
    def _get_latency_compensation_obs(self) -> torch.Tensor:
        """Get previous action for latency compensation (5 dimensions)."""
        if not hasattr(self, 'previous_actions'):
            # Initialize with zeros if no previous action
            self.previous_actions = torch.zeros((self.num_envs, 5), device=self.device)
        
        return self.previous_actions.clone()
    
    def _get_nearest_coral_position(self, env_idx: int) -> torch.Tensor:
        """Get position of nearest coral in specified environment."""
        if not hasattr(self, 'coral_objects') or not hasattr(self.coral_objects.data, 'root_pos_w'):
            return None
            
        coral_positions = self.coral_objects.data.root_pos_w
        if len(coral_positions) == 0:
            return None
            
        # Get intake position for distance calculation
        intake_pos, _ = self._get_intake_piece_pose(env_idx)
        if intake_pos is None:
            return None
            
        # Find nearest coral in this environment
        corals_per_env = len(self.coral_prim_paths) // self.num_envs
        start_idx = env_idx * corals_per_env
        end_idx = start_idx + corals_per_env
        
        min_distance = float('inf')
        nearest_coral_pos = None
        
        for coral_idx in range(start_idx, min(end_idx, len(coral_positions))):
            coral_pos = coral_positions[coral_idx]
            distance = torch.norm(intake_pos - coral_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_coral_pos = coral_pos
                
        return nearest_coral_pos
    
    def _get_nearest_reef_sensor_position(self, env_idx: int) -> torch.Tensor:
        """Get position of nearest reef sensor in specified environment."""
        if not hasattr(self, 'reef_sensors') or len(self.reef_sensors) == 0:
            return None
            
        # Get intake position for distance calculation
        intake_pos, _ = self._get_intake_piece_pose(env_idx)
        if intake_pos is None:
            return None
            
        # Find nearest reef sensor in this environment
        sensors_per_env = len(self.reef_sensors) // self.num_envs
        start_idx = env_idx * sensors_per_env
        end_idx = start_idx + sensors_per_env
        
        min_distance = float('inf')
        nearest_sensor_pos = None
        
        for sensor_idx in range(start_idx, min(end_idx, len(self.reef_sensors))):
            sensor_info = self.reef_sensors[sensor_idx]
            sensor_pos = torch.tensor(sensor_info['position'], device=self.device)
            distance = torch.norm(intake_pos - sensor_pos)
            if distance < min_distance:
                min_distance = distance
                nearest_sensor_pos = sensor_pos
                
        return nearest_sensor_pos

    def _debug_simplified_observations(self, obs, robot_motion, goal_geometry, mechanism_readiness, time_context, latency_compensation):
        """Debug function for simplified observation system."""
        # Only print debug info every 120 steps (1 second at 120 FPS) for env 0
        if not hasattr(self, '_debug_step_counter'):
            self._debug_step_counter = 0
        
        self._debug_step_counter += 1
        
        if self._debug_step_counter % 120 == 0:  # Print every second
            env_idx = 0  # Focus on first environment
            print("\n" + "="*80)
            print(f"� SIMPLIFIED OBSERVATION DEBUG - Step {self._debug_step_counter} - Environment {env_idx}")
            print("="*80)
            
            # Robot motion (3D)
            motion = robot_motion[env_idx]
            print(f"🤖 Robot Motion - Forward: {motion[0]:.3f} m/s, Lateral: {motion[1]:.3f} m/s, Yaw Rate: {motion[2]:.3f} rad/s")
            
            # Goal geometry (3D)  
            geometry = goal_geometry[env_idx]
            goal_type = "REEF" if self.is_holding[env_idx] else "CORAL"
            print(f"🎯 Goal Geometry ({goal_type}) - Distance: {geometry[0]:.3f} m, Sin(angle): {geometry[1]:.3f}, Cos(angle): {geometry[2]:.3f}")
            
            # Mechanism readiness (3D)
            mechanism = mechanism_readiness[env_idx]
            has_piece = "YES" if mechanism[0] == 1.0 else "NO"
            at_setpoint = "YES" if mechanism[2] == 1.0 else "NO"
            print(f"⚙️  Mechanism - Has Piece: {has_piece}, Elevator Error: {mechanism[1]:.3f}, At Setpoint: {at_setpoint}")
            
            # Time context (1D)
            time_rem = time_context[env_idx, 0]
            print(f"⏱️  Time Remaining: {time_rem:.3f} (normalized)")
            
            # Latency compensation (5D)
            prev_action = latency_compensation[env_idx]
            print(f"🎮 Previous Action - Drive: [{prev_action[0]:.3f}, {prev_action[1]:.3f}, {prev_action[2]:.3f}], Level: {prev_action[3]:.3f}, Gripper: {prev_action[4]:.3f}")
            
            # Total observation vector
            print(f"📊 Total Observation (15D): {obs[env_idx].tolist()}")
            
            print("="*80 + "\n")

    # Rewards
    def _get_rewards(self) -> torch.Tensor:
        # Phase-based reward system with state machine
        
        # 1. Compute event flags and geometric features once per step
        event_flags = self._compute_event_flags()
        geo_features = self._compute_geometric_features()
        
        # 2. Determine current phase for each environment
        phases = self._determine_phases(geo_features)
        
        # 3. Apply phase-specific rewards using switchboard
        phase_rewards = self._apply_phase_rewards(phases, geo_features)
        
        # 4. Add global one-shot event rewards
        event_rewards = self._apply_event_rewards(event_flags)
        
        # 5. Update tracking state after calculating rewards
        self._update_tracking_state(geo_features)
        
        # 6. Combine all reward components
        total_reward = phase_rewards + event_rewards
        
        # 7. Optional: Log phase info periodically
        self._log_phase_info(phases, phase_rewards, event_rewards)
        
        return total_reward

    def _compute_event_flags(self) -> dict:
        """Compute one-shot event flags for this step."""
        # Detect grip/drop/score events
        new_grip_events = (~self.previous_is_holding) & self.is_holding
        drop_events = self.previous_is_holding & (~self.is_holding)
        
        # Detect scoring events
        score_events = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        welding_events = self.detect_reef_welding()
        for event in welding_events:
            env_idx = event['env_idx']
            score_events[env_idx] = True
        
        return {
            'picked_up': new_grip_events,
            'dropped': drop_events,
            'scored': score_events
        }
    
    def _compute_geometric_features(self) -> dict:
        """Compute geometric features once per step."""
        features = {
            'coral_distance': torch.full((self.num_envs,), float('inf'), device=self.device),
            'reef_distance': torch.full((self.num_envs,), float('inf'), device=self.device),
            'coral_alignment': torch.zeros(self.num_envs, device=self.device),
            'reef_alignment': torch.zeros(self.num_envs, device=self.device),
            'elevator_ready': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            'intake_pos_valid': torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        }
        
        if not hasattr(self, 'coral_objects') or not hasattr(self.coral_objects.data, 'root_pos_w'):
            return features
            
        coral_positions = self.coral_objects.data.root_pos_w
        
        # ✅ Add validation for coral objects existence
        if len(coral_positions) == 0 or not hasattr(self, 'coral_prim_paths') or len(self.coral_prim_paths) == 0:
            return features
            
        corals_per_env = len(self.coral_prim_paths) // self.num_envs
        
        for env_idx in range(self.num_envs):
            # Get intake piece position and orientation
            intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
            if intake_pos is None or intake_quat is None:
                continue
                
            features['intake_pos_valid'][env_idx] = True
            
            # Intake forward direction
            intake_forward = torch.tensor([0.0, 1.0, 0.0], device=self.device)
            intake_forward_world = math_utils.quat_apply(intake_quat.unsqueeze(0), intake_forward.unsqueeze(0)).squeeze(0)
            
            # Find nearest coral distance and alignment
            start_idx = env_idx * corals_per_env
            end_idx = start_idx + corals_per_env
            
            min_coral_distance = float('inf')
            best_coral_alignment = -1.0
            
            for coral_idx in range(start_idx, min(end_idx, len(coral_positions))):
                if coral_idx >= len(coral_positions):  # ✅ Add bounds check
                    break
                    
                coral_pos = coral_positions[coral_idx]
                distance = torch.norm(intake_pos - coral_pos).item()
                
                if distance < min_coral_distance:
                    min_coral_distance = distance
                    
                    # Calculate alignment to this coral
                    direction_to_coral = coral_pos - intake_pos
                    direction_to_coral = direction_to_coral / (torch.norm(direction_to_coral) + 1e-8)
                    alignment = torch.dot(intake_forward_world, direction_to_coral).item()
                    best_coral_alignment = alignment
            
            features['coral_distance'][env_idx] = min_coral_distance
            features['coral_alignment'][env_idx] = best_coral_alignment
            
            # Find nearest reef sensor distance and alignment
            sensor_positions = self._get_reef_sensor_positions(env_idx)
            min_reef_distance = float('inf')
            best_reef_alignment = -1.0
            
            for sensor_name, sensor_pos in sensor_positions.items():
                # Only consider unoccupied sensors
                sensor_key = (env_idx, sensor_name)
                if sensor_key not in getattr(self, 'sensor_coral_mapping', {}):
                    distance = torch.norm(intake_pos - sensor_pos).item()
                    
                    if distance < min_reef_distance:
                        min_reef_distance = distance
                        
                        # Calculate alignment to this sensor
                        direction_to_sensor = sensor_pos - intake_pos
                        direction_to_sensor = direction_to_sensor / (torch.norm(direction_to_sensor) + 1e-8)
                        alignment = torch.dot(intake_forward_world, direction_to_sensor).item()
                        best_reef_alignment = alignment
            
            features['reef_distance'][env_idx] = min_reef_distance
            features['reef_alignment'][env_idx] = best_reef_alignment
            
            # Check if elevator is at appropriate height for nearest sensor
            # This is a simplified check - in reality you'd want to match elevator height to sensor level
            current_elevator_angle = self.robot.data.joint_pos[env_idx, 6]  # Intake joint as proxy
            target_angles = [intake_zero_position, intake_level_1_position, intake_level_2_position, intake_level_3_position]
            
            min_angle_diff = float('inf')
            for target_angle in target_angles:
                angle_diff = abs(current_elevator_angle - target_angle)
                min_angle_diff = min(min_angle_diff, angle_diff)
            
            features['elevator_ready'][env_idx] = min_angle_diff < self.ELEVATOR_READY_THRESHOLD
        
        return features
    
    def _determine_phases(self, geo_features: dict) -> torch.Tensor:
        """Determine current phase for each environment based on geometric features."""
        new_phases = self.current_phase.clone()
        
        for env_idx in range(self.num_envs):
            if not geo_features['intake_pos_valid'][env_idx]:
                continue  # Keep current phase if we can't determine position
                
            current_phase = self.current_phase[env_idx].item()
            coral_dist = geo_features['coral_distance'][env_idx].item()
            reef_dist = geo_features['reef_distance'][env_idx].item()
            reef_alignment = geo_features['reef_alignment'][env_idx].item()
            elevator_ready = geo_features['elevator_ready'][env_idx].item()
            is_holding = self.is_holding[env_idx].item()
            
            # Phase transition logic
            if current_phase == 6:  # RESET
                # Count down reset timer
                self.reset_timer[env_idx] -= 1
                if self.reset_timer[env_idx] <= 0:
                    new_phases[env_idx] = 0  # SEEK
                    
            elif not is_holding:  # Not holding coral
                if coral_dist < self.INTAKE_DISTANCE:
                    new_phases[env_idx] = 1  # INTAKE
                else:
                    new_phases[env_idx] = 0  # SEEK
                    
            else:  # Holding coral
                if reef_dist > self.CARRY_MED_DISTANCE:
                    new_phases[env_idx] = 2  # CARRY_LONG
                elif reef_dist > self.FINAL_ALIGN_DISTANCE:
                    new_phases[env_idx] = 3  # CARRY_MED
                elif reef_dist > self.DOCK_DISTANCE:
                    new_phases[env_idx] = 4  # FINAL_ALIGN
                else:
                    # Close enough to dock - check if ready
                    if reef_alignment > self.ALIGNMENT_THRESHOLD and elevator_ready:
                        new_phases[env_idx] = 5  # DOCK
                    else:
                        new_phases[env_idx] = 4  # FINAL_ALIGN (not ready to dock yet)
        
        # Update current phases
        self.current_phase = new_phases
        return new_phases
    
    def _apply_phase_rewards(self, phases: torch.Tensor, geo_features: dict) -> torch.Tensor:
        """Apply phase-specific rewards using switchboard approach."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Base rewards/penalties that apply to all phases (with phase-specific scaling)
        alive_bonus = self.cfg.rew_scale_alive
        
        # Time penalty (varies by phase)
        max_episode_length = self.max_episode_length
        base_time_penalty = self.cfg.rew_scale_time_penalty * (self.episode_length_buf.float() / max_episode_length)
        
        # Tipover penalty (always full strength)
        body_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        up_world = math_utils.quat_apply(self.robot.data.root_link_quat_w, body_up)
        up_z = up_world[:, 2]
        tipping_penalty = torch.where(up_z < 0.3, self.cfg.rew_scale_tipping, 0.0)
        
        # Smoothness penalty (varies by phase)
        joint_vel_magnitude = torch.norm(self.robot.data.joint_vel, dim=-1)
        base_smoothness_penalty = self.cfg.rew_scale_smoothness * joint_vel_magnitude
        
        # Apply rewards based on current phase
        for env_idx in range(self.num_envs):
            phase = phases[env_idx].item()
            
            # Always apply these
            rewards[env_idx] += alive_bonus
            rewards[env_idx] += tipping_penalty[env_idx]
            
            if phase == 0:  # SEEK
                rewards[env_idx] += self._apply_seek_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 1.0  # Full time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 1.0  # Full smoothness penalty
                
            elif phase == 1:  # INTAKE
                rewards[env_idx] += self._apply_intake_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 0.8  # Reduced time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 0.7  # Reduced smoothness penalty
                
            elif phase == 2:  # CARRY_LONG
                rewards[env_idx] += self._apply_carry_long_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 1.0  # Full time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 1.0  # Full smoothness penalty
                
            elif phase == 3:  # CARRY_MED
                rewards[env_idx] += self._apply_carry_med_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 0.9  # Slightly reduced time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 0.8  # Reduced smoothness penalty
                
            elif phase == 4:  # FINAL_ALIGN
                rewards[env_idx] += self._apply_final_align_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 0.3  # Much reduced time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 0.5  # Reduced smoothness penalty
                
            elif phase == 5:  # DOCK
                rewards[env_idx] += self._apply_dock_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 0.0  # No time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 0.0  # No smoothness penalty
                
            elif phase == 6:  # RESET
                rewards[env_idx] += self._apply_reset_rewards(env_idx, geo_features)
                rewards[env_idx] += base_time_penalty[env_idx] * 0.5  # Reduced time pressure
                rewards[env_idx] += base_smoothness_penalty[env_idx] * 0.5  # Reduced smoothness penalty
        
        return rewards

    def _apply_seek_rewards(self, env_idx: int, geo_features: dict) -> float:
        """SEEK phase: Sequential 'lower → align → approach' strategy with prerequisites."""
        reward = 0.0
        
        coral_dist = geo_features['coral_distance'][env_idx].item()
        coral_alignment = geo_features['coral_alignment'][env_idx].item()
        reef_dist = geo_features['reef_distance'][env_idx].item()
        reef_alignment = geo_features['reef_alignment'][env_idx].item()
        
        # ✅ PENALTY: Going toward reef without coral
        if reef_alignment > 0.3 and reef_dist < 4.0:  # Facing reef and getting close
            reef_distraction_penalty = -0.2 * reef_alignment * (4.0 - reef_dist) / 4.0
            reward += reef_distraction_penalty
        
        # Get current mechanism positions
        current_intake_angle = self.robot.data.joint_pos[env_idx, 6]  # Intake joint
        current_elevator_1_pos = self.robot.data.joint_pos[env_idx, 0]  # Elevator 1 joint
        current_elevator_2_pos = self.robot.data.joint_pos[env_idx, 1]  # Elevator 2 joint
        
        # Calculate "lowered" state - mechanisms at stowed/zero position
        intake_zero_diff = abs(current_intake_angle - intake_zero_position)
        elevator_1_zero_diff = abs(current_elevator_1_pos - elevator_1_zero_position)
        elevator_2_zero_diff = abs(current_elevator_2_pos - elevator_2_zero_position)
        
        # Stage readiness flags
        mechanisms_lowered = (intake_zero_diff < 0.3 and 
                            elevator_1_zero_diff < 0.1 and 
                            elevator_2_zero_diff < 0.1)
        
        alignment_threshold = 0.7  # cos(~45 degrees) - reasonably aligned
        well_aligned = coral_alignment > alignment_threshold
        
        # 🎯 STAGE 1: LOWER MECHANISMS FIRST (Highest Priority)
        if not mechanisms_lowered:
            # Strong rewards for lowering mechanisms - must happen first
            
            # Intake lowering reward
            if intake_zero_diff < 0.3:
                intake_lower_reward = (0.3 - intake_zero_diff) / 0.3 * 0.4
                reward += intake_lower_reward
            
            # Elevator 1 lowering reward  
            if elevator_1_zero_diff < 0.1:
                elevator_1_lower_reward = (0.1 - elevator_1_zero_diff) / 0.1 * 0.3
                reward += elevator_1_lower_reward
            
            # Elevator 2 lowering reward
            if elevator_2_zero_diff < 0.1:
                elevator_2_lower_reward = (0.1 - elevator_2_zero_diff) / 0.1 * 0.3
                reward += elevator_2_lower_reward
            
            # Small alignment reward even when not lowered (encourage some turning)
            if coral_alignment > 0:
                basic_alignment_reward = coral_alignment * 0.1
                reward += basic_alignment_reward
            
            # PENALTY: Don't approach coral without lowering mechanisms first
            if coral_dist < 4.0 and not mechanisms_lowered:
                premature_approach_penalty = -0.3 * (4.0 - coral_dist) / 4.0
                reward += premature_approach_penalty
        
        # 🎯 STAGE 2: ALIGN WHEN LOWERED (Medium Priority)
        elif mechanisms_lowered and not well_aligned:
            # Maintain lowered position bonus
            mechanism_ready_bonus = 0.2
            reward += mechanism_ready_bonus
            
            # Strong alignment rewards when mechanisms are ready
            if coral_alignment > 0:
                alignment_reward = coral_alignment * 0.6  # Stronger than basic
                reward += alignment_reward
            
            # Penalty for approaching when lowered but not aligned
            if coral_dist < 3.0:
                misaligned_approach_penalty = -0.2 * (alignment_threshold - coral_alignment) * (3.0 - coral_dist) / 3.0
                reward += misaligned_approach_penalty
        
        # 🎯 STAGE 3: APPROACH WHEN LOWERED AND ALIGNED (Final Stage)
        elif mechanisms_lowered and well_aligned:
            # Maintain mechanism and alignment bonuses
            perfect_setup_bonus = 0.3
            reward += perfect_setup_bonus
            
            # Enhanced alignment reward for maintaining good alignment
            enhanced_alignment_reward = (coral_alignment - alignment_threshold) / (1.0 - alignment_threshold) * 0.5
            reward += enhanced_alignment_reward
            
            # NOW allow distance improvement rewards
            if hasattr(self, 'prev_coral_distance') and self.prev_coral_distance[env_idx] != float('inf'):
                if coral_dist < self.prev_coral_distance[env_idx]:  # Only reward getting CLOSER
                    improvement = self.prev_coral_distance[env_idx] - coral_dist  # Positive value
                    # Scale distance reward by alignment quality
                    aligned_distance_reward = improvement * self.cfg.rew_scale_distance_improvement * coral_alignment
                    reward += aligned_distance_reward
                elif coral_dist > self.prev_coral_distance[env_idx]:  # Penalize moving AWAY
                    regression = coral_dist - self.prev_coral_distance[env_idx]  # Positive value
                    # Penalty for moving away (scaled by alignment)
                    distance_regression_penalty = -regression * self.cfg.rew_scale_distance_improvement * 0.5 * max(coral_alignment, 0.1)
                    reward += distance_regression_penalty
            
            # Proximity bonus when properly set up and aligned
            if coral_dist < 3.0:
                aligned_proximity_bonus = max(0, (2.0 - coral_dist) / 2.0) * 0.4 * coral_alignment
                reward += aligned_proximity_bonus
        
        # Handle edge cases - basic alignment reward for any other states
        else:
            if coral_alignment > 0:
                basic_alignment_reward = coral_alignment * 0.1
                reward += basic_alignment_reward
        
        return reward

    def _apply_intake_rewards(self, env_idx: int, geo_features: dict) -> float:
        """INTAKE phase: Precise alignment and careful approach - final alignment before pickup."""
        reward = 0.0
        
        coral_dist = geo_features['coral_distance'][env_idx].item()
        coral_alignment = geo_features['coral_alignment'][env_idx].item()
        reef_dist = geo_features['reef_distance'][env_idx].item()
        reef_alignment = geo_features['reef_alignment'][env_idx].item()
        
        # ✅ PENALTY: Going toward reef without coral (stronger in INTAKE phase)
        if reef_alignment > 0.2 and reef_dist < 3.0:  # Facing reef and getting close
            reef_distraction_penalty = -0.3 * reef_alignment * (3.0 - reef_dist) / 3.0
            reward += reef_distraction_penalty
        
        # CRITICAL: Very high alignment requirement for intake phase
        high_alignment_threshold = 0.85  # cos(~32 degrees) - very well aligned
        
        if coral_alignment > high_alignment_threshold:
            # Excellent alignment - allow final approach
            precise_alignment_reward = (coral_alignment - high_alignment_threshold) / (1.0 - high_alignment_threshold) * 1.0
            reward += precise_alignment_reward
            
            # Final approach reward only when excellently aligned
            if coral_dist < 0.8:
                aligned_close_bonus = (0.8 - coral_dist) / 0.8 * 0.6 * coral_alignment
                reward += aligned_close_bonus
                
            # Reward for proper intake positioning when aligned
            current_intake_angle = self.robot.data.joint_pos[env_idx, 6]
            optimal_angles = [intake_zero_position, intake_level_1_position]
            min_angle_diff = min(abs(current_intake_angle - angle) for angle in optimal_angles)
            
            if min_angle_diff < 0.5:
                aligned_intake_reward = (0.5 - min_angle_diff) / 0.5 * 0.4 * coral_alignment
                reward += aligned_intake_reward
                
        elif coral_alignment > 0.5:  # Moderate alignment
            # Still reward alignment improvement but discourage approach
            moderate_alignment_reward = coral_alignment * 0.4
            reward += moderate_alignment_reward
            
            # Penalty for getting too close when not precisely aligned
            if coral_dist < 0.5:
                premature_approach_penalty = -0.2 * (high_alignment_threshold - coral_alignment)
                reward += premature_approach_penalty
                
        else:
            # Poor alignment in INTAKE phase is bad - encourage backing off and realigning
            if coral_dist < 1.0:
                too_close_penalty = -0.3 * (1.0 - coral_dist)  # Penalty for being close when misaligned
                reward += too_close_penalty
            
            # Small reward for any alignment improvement
            if coral_alignment > 0:
                basic_alignment_reward = coral_alignment * 0.1
                reward += basic_alignment_reward
        
        return reward

    def _apply_carry_long_rewards(self, env_idx: int, geo_features: dict) -> float:
        """CARRY_LONG phase: Reward progress toward reef and basic alignment."""
        reward = 0.0
        
        reef_dist = geo_features['reef_distance'][env_idx].item()
        reef_alignment = geo_features['reef_alignment'][env_idx].item()
        
        # Carrying bonus
        reward += self.cfg.rew_scale_coral_carrying
        
        # Reward for facing reef
        if reef_alignment > 0:
            alignment_reward = reef_alignment * 0.3
            reward += alignment_reward
        
        # Distance improvement toward reef - use consistent reef_distance tracking
        if hasattr(self, 'prev_reef_distance') and self.prev_reef_distance[env_idx] != float('inf'):
            if reef_dist < self.prev_reef_distance[env_idx]:  # Only reward getting CLOSER
                improvement = self.prev_reef_distance[env_idx] - reef_dist  # Positive value
                reward += improvement * self.cfg.rew_scale_distance_improvement
            elif reef_dist > self.prev_reef_distance[env_idx]:  # Penalize moving AWAY
                regression = reef_dist - self.prev_reef_distance[env_idx]  # Positive value
                # Penalty for moving away from reef while carrying coral
                reef_regression_penalty = -regression * self.cfg.rew_scale_distance_improvement * 0.3
                reward += reef_regression_penalty
        
        return reward

    def _apply_carry_med_rewards(self, env_idx: int, geo_features: dict) -> float:
        """CARRY_MED phase: Stronger emphasis on alignment and mechanism prep."""
        reward = 0.0
        
        reef_alignment = geo_features['reef_alignment'][env_idx].item()
        elevator_ready = geo_features['elevator_ready'][env_idx].item()
        
        # Carrying bonus
        reward += self.cfg.rew_scale_coral_carrying
        
        # Stronger reward for reef alignment
        if reef_alignment > 0:
            alignment_reward = reef_alignment * 0.5  # Stronger than CARRY_LONG
            reward += alignment_reward
        
        # Begin rewarding elevator preparation
        if elevator_ready:
            reward += 0.2
        
        return reward

    def _apply_final_align_rewards(self, env_idx: int, geo_features: dict) -> float:
        """FINAL_ALIGN phase: Strong rewards for precise alignment and elevator readiness."""
        reward = 0.0
        
        reef_alignment = geo_features['reef_alignment'][env_idx].item()
        elevator_ready = geo_features['elevator_ready'][env_idx].item()
        
        # Very strong alignment reward
        if reef_alignment > self.ALIGNMENT_THRESHOLD:
            alignment_reward = reef_alignment * 1.0  # Very strong
            reward += alignment_reward
        
        # Strong elevator readiness reward
        if elevator_ready:
            reward += 0.5
        
        # Bonus for being in the "perfect" state
        if reef_alignment > self.ALIGNMENT_THRESHOLD and elevator_ready:
            reward += 0.3  # Perfect setup bonus
        
        return reward

    def _apply_dock_rewards(self, env_idx: int, geo_features: dict) -> float:
        """DOCK phase: Maintain neutral rewards, let scoring bonus handle success."""
        reward = 0.0
        
        # Just a small bonus for being in dock phase (ready to score)
        reward += 0.1
        
        return reward

    def _apply_reset_rewards(self, env_idx: int, geo_features: dict) -> float:
        """RESET phase: Mild movement reward and low costs."""
        reward = 0.0
        
        # Small bonus for quick transition back to seeking
        reward += 0.05
        
        return reward

    def _apply_event_rewards(self, event_flags: dict) -> torch.Tensor:
        """Apply global one-shot event rewards."""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # Pickup bonus
        rewards[event_flags['picked_up']] += self.cfg.rew_scale_coral_grip
        
        # Drop penalty
        rewards[event_flags['dropped']] += self.cfg.rew_scale_coral_drop
        
        # Score bonus
        for env_idx in range(self.num_envs):
            if event_flags['scored'][env_idx]:
                # Set RESET phase and timer
                self.current_phase[env_idx] = 6  # RESET
                self.reset_timer[env_idx] = self.reset_duration
                
                # Apply scoring reward
                welding_events = self.detect_reef_welding()
                for event in welding_events:
                    if event['env_idx'] == env_idx:
                        points = event.get('points', 3)
                        rewards[env_idx] += float(points) + self.cfg.rew_scale_weld_success
        
        return rewards

    def _log_phase_info(self, phases: torch.Tensor, phase_rewards: torch.Tensor, event_rewards: torch.Tensor):
        """Log phase information periodically."""
        if not hasattr(self, '_phase_log_counter'):
            self._phase_log_counter = 0
        
        self._phase_log_counter += 1
        
        if self._phase_log_counter % 240 == 0:  # Every 2 seconds at 120 FPS
            env_idx = 0  # Focus on first environment
            phase_name = self.PHASES.get(phases[env_idx].item(), "UNKNOWN")
            current_phase = phases[env_idx].item()
            
            print(f"\n🔄 PHASE DEBUG - Step {self._phase_log_counter} - Environment {env_idx}")
            print(f"   Current Phase: {phase_name}")
            print(f"   Holding Coral: {self.is_holding[env_idx].item()}")
            print(f"   Phase Reward: {phase_rewards[env_idx].item():.3f}")
            print(f"   Event Reward: {event_rewards[env_idx].item():.3f}")
            print(f"   Total Reward: {(phase_rewards[env_idx] + event_rewards[env_idx]).item():.3f}")
            
            # Get geometric features for detailed logging
            geo_features = self._compute_geometric_features()
            
            if geo_features['intake_pos_valid'][env_idx]:
                coral_dist = geo_features['coral_distance'][env_idx].item()
                coral_alignment = geo_features['coral_alignment'][env_idx].item()
                reef_dist = geo_features['reef_distance'][env_idx].item()
                reef_alignment = geo_features['reef_alignment'][env_idx].item()
                elevator_ready = geo_features['elevator_ready'][env_idx].item()
                
                # Phase-specific information
                if current_phase in [0, 1]:  # SEEK or INTAKE - focus on coral
                    print(f"   🪸 Coral Distance: {coral_dist:.3f}m")
                    print(f"   🪸 Coral Alignment: {coral_alignment:.3f} (1.0=perfect, -1.0=opposite)")
                    if current_phase == 0:  # SEEK
                        print(f"   📏 Intake Threshold: {self.INTAKE_DISTANCE:.3f}m")
                        print(f"   ⚡ Ready to INTAKE: {'YES' if coral_dist < self.INTAKE_DISTANCE else 'NO'}")
                    elif current_phase == 1:  # INTAKE
                        print(f"   🎯 Close Enough (<0.8m): {'YES' if coral_dist < 0.8 else 'NO'}")
                        print(f"   🎯 Well Aligned (>0): {'YES' if coral_alignment > 0 else 'NO'}")
                
                elif current_phase in [2, 3, 4, 5]:  # CARRY phases - focus on reef
                    print(f"   🏔️  Reef Distance: {reef_dist:.3f}m")
                    print(f"   🏔️  Reef Alignment: {reef_alignment:.3f} (1.0=perfect, -1.0=opposite)")
                    print(f"   🏗️  Elevator Ready: {'YES' if elevator_ready else 'NO'}")
                    
                    if current_phase == 2:  # CARRY_LONG
                        print(f"   📏 Med Range Threshold: {self.CARRY_MED_DISTANCE:.3f}m")
                        print(f"   ⚡ Ready for CARRY_MED: {'YES' if reef_dist <= self.CARRY_MED_DISTANCE else 'NO'}")
                    elif current_phase == 3:  # CARRY_MED
                        print(f"   📏 Final Align Threshold: {self.FINAL_ALIGN_DISTANCE:.3f}m")
                        print(f"   ⚡ Ready for FINAL_ALIGN: {'YES' if reef_dist <= self.FINAL_ALIGN_DISTANCE else 'NO'}")
                    elif current_phase == 4:  # FINAL_ALIGN
                        print(f"   📏 Dock Threshold: {self.DOCK_DISTANCE:.3f}m")
                        print(f"   📏 Alignment Threshold: {self.ALIGNMENT_THRESHOLD:.3f}")
                        print(f"   ⚡ Ready to DOCK: {'YES' if reef_dist <= self.DOCK_DISTANCE and reef_alignment > self.ALIGNMENT_THRESHOLD and elevator_ready else 'NO'}")
                    elif current_phase == 5:  # DOCK
                        print(f"   🎯 Perfect Position: Distance={reef_dist:.3f}m, Alignment={reef_alignment:.3f}")
                        
                elif current_phase == 6:  # RESET
                    reset_time_left = self.reset_timer[env_idx].item()
                    print(f"   ⏰ Reset Timer: {reset_time_left:.0f} steps remaining")
                    print(f"   ⚡ Ready to SEEK: {'YES' if reset_time_left <= 0 else 'NO'}")
            else:
                print(f"   ⚠️  Intake position invalid - geometric features unavailable")



    def _update_tracking_state(self, geo_features: dict):
        """Update coral holding state and distance tracking for all environments."""
        if not hasattr(self, 'coral_objects') or not hasattr(self.coral_objects.data, 'root_pos_w'):
            return
            
        coral_positions = self.coral_objects.data.root_pos_w
        
        # ✅ Add validation for coral objects existence  
        if len(coral_positions) == 0 or not hasattr(self, 'coral_prim_paths') or len(self.coral_prim_paths) == 0:
            return
            
        corals_per_env = len(self.coral_prim_paths) // self.num_envs
        
        # Store previous state
        self.previous_is_holding = self.is_holding.clone()
        
        # Get contact sensor data for coral detection
        coral_contacted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self.intake_contact_sensor, 'data') and hasattr(self.intake_contact_sensor.data, 'net_forces_w'):
            contact_forces = self.intake_contact_sensor.data.net_forces_w
            force_magnitude = torch.norm(contact_forces, dim=-1)  # Shape: (num_envs,)
            coral_contacted = force_magnitude > 0.5  # Threshold for contact detection (0.5N) - match _apply_action threshold
        
        for env_idx in range(self.num_envs):
            # Get intake piece position for tracking
            intake_pos, _ = self._get_intake_piece_pose(env_idx)
            if intake_pos is None:
                continue  # Skip tracking if intake position unavailable
            
            # Check surface gripper state (1 = Closed/Grasping, 0 = Closing, -1 = Open)
            gripper_closed = self.surface_gripper.state[env_idx] == 1
            
            # Find nearest coral in this environment
            start_idx = env_idx * corals_per_env
            end_idx = start_idx + corals_per_env
            
            min_coral_distance = float('inf')
            nearest_coral_idx = -1
            
            for coral_idx in range(start_idx, min(end_idx, len(coral_positions))):
                coral_pos = coral_positions[coral_idx]
                distance = torch.norm(intake_pos - coral_pos).item()
                if distance < min_coral_distance:
                    min_coral_distance = distance
                    nearest_coral_idx = coral_idx
            
            # Update coral holding state based on contact sensor AND gripper state
            if gripper_closed and coral_contacted[env_idx]:  # Contact sensor detects coral AND gripper is closed
                self.is_holding[env_idx] = True
                self.holding_coral_idx[env_idx] = nearest_coral_idx
            else:
                self.is_holding[env_idx] = False
                self.holding_coral_idx[env_idx] = -1
            
            # Find nearest free socket distance
            min_socket_distance = float('inf')
            sensor_positions = self._get_reef_sensor_positions(env_idx)
            
            for sensor_name, sensor_pos in sensor_positions.items():
                # Check if this sensor is already occupied
                sensor_key = (env_idx, sensor_name)
                if sensor_key not in getattr(self, 'sensor_coral_mapping', {}):
                    distance = torch.norm(intake_pos - sensor_pos).item()
                    min_socket_distance = min(min_socket_distance, distance)
            
            # Store current distances as previous for next step
            # (This happens after reward calculation uses the previous values)
            self.prev_coral_distance[env_idx] = min_coral_distance
            self.prev_socket_distance[env_idx] = min_socket_distance
            # Use the reef_distance from geo_features to ensure consistency
            self.prev_reef_distance[env_idx] = geo_features['reef_distance'][env_idx].item()
    

    # Observation Helpers
    def _get_intake_piece_pose(self, env_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Get the position and orientation of the intake piece for the specified environment."""
        # Get robot root position and orientation for this environment
        robot_pos = self.robot.data.root_pos_w[env_idx]  # [3]
        robot_quat = self.robot.data.root_quat_w[env_idx]  # [4] (w, x, y, z)
    
        
        # Define intake piece offset from robot base in robot's local frame
        # This should be adjusted based on your robot's actual geometry
        intake_base_offset = torch.tensor([0, -0.5, 0], device=self.device)  
        
        # Calculate intake piece position by transforming offset to world frame
        intake_offset_world = math_utils.quat_apply(robot_quat.unsqueeze(0), intake_base_offset.unsqueeze(0)).squeeze(0)
        intake_pos = robot_pos + intake_offset_world

        
        # Combine robot orientation with intake joint rotation
        # Rotate clockwise 90 degrees around Z-axis (negative rotation for clockwise)
        z_rotation_quat = torch.tensor([0.7071068, 0.0, 0.0, -0.7071068], device=self.device)  # 90° clockwise around Z
        intake_quat = math_utils.quat_mul(robot_quat.unsqueeze(0), z_rotation_quat.unsqueeze(0)).squeeze(0)
        
        return intake_pos, intake_quat


    def _get_coral_observations(self) -> torch.Tensor:
        """Get nearest coral positions and orientations relative to intake piece with alignment, height difference, and distance."""
        coral_obs = torch.zeros((self.num_envs, 9), device=self.device)  # [rel_x, rel_y, rel_z, qx, qy, qz, alignment, height_diff, distance]
        
        if hasattr(self, 'coral_objects') and hasattr(self.coral_objects.data, 'root_pos_w'):            
            coral_positions = self.coral_objects.data.root_pos_w
            coral_orientations = self.coral_objects.data.root_quat_w
            
            # ✅ Add validation for coral objects
            if len(coral_positions) == 0 or not hasattr(self, 'coral_prim_paths') or len(self.coral_prim_paths) == 0:
                return coral_obs
            
            corals_per_env = len(self.coral_prim_paths) // self.num_envs
            
            for env_idx in range(self.num_envs):
                # Get intake piece position and orientation - only use actual intake piece prim
                intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
                
                # Skip if intake piece position is not available
                if intake_pos is None or intake_quat is None:
                    continue  # Leave observations as zeros for this environment
                
                # Find nearest coral in this environment
                start_idx = env_idx * corals_per_env
                end_idx = start_idx + corals_per_env
                
                min_distance = float('inf')
                nearest_coral_idx = start_idx
                
                for coral_idx in range(start_idx, min(end_idx, len(coral_positions))):
                    coral_pos = coral_positions[coral_idx]
                    distance = torch.norm(intake_pos - coral_pos)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_coral_idx = coral_idx
                
                if nearest_coral_idx < len(coral_positions):
                    # Get relative position from intake piece to coral
                    coral_pos_world = coral_positions[nearest_coral_idx]
                    rel_pos_world = coral_pos_world - intake_pos
                    
                    # Transform to intake piece's local frame
                    intake_quat_inv = math_utils.quat_conjugate(intake_quat.unsqueeze(0)).squeeze(0)
                    rel_pos_local = math_utils.quat_apply(intake_quat_inv.unsqueeze(0), rel_pos_world.unsqueeze(0)).squeeze(0)
                    
                    # Get relative orientation
                    coral_quat = coral_orientations[nearest_coral_idx]
                    rel_quat = math_utils.quat_mul(intake_quat_inv.unsqueeze(0), coral_quat.unsqueeze(0)).squeeze(0)
                    
                    # Calculate alignment: how well the intake is pointing toward the coral
                    # Intake forward direction (positive Y in intake's local frame)
                    intake_forward_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)
                    intake_forward_world = math_utils.quat_apply(intake_quat.unsqueeze(0), intake_forward_local.unsqueeze(0)).squeeze(0)
                    
                    # Direction from intake to coral
                    direction_to_coral = coral_pos_world - intake_pos
                    direction_to_coral = direction_to_coral / (torch.norm(direction_to_coral) + 1e-8)  # Normalize
                    
                    # Alignment is dot product (cosine of angle between intake forward and direction to coral)
                    alignment = torch.dot(intake_forward_world, direction_to_coral)
                    
                    # Calculate height difference (coral height - intake height)
                    height_diff = coral_pos_world[2] - intake_pos[2]
                    
                    # Calculate distance to coral
                    distance = torch.norm(rel_pos_local)
                    
                    # Store in observation (position + orientation xyz components + alignment + height difference + distance)
                    coral_obs[env_idx, :3] = rel_pos_local
                    coral_obs[env_idx, 3:6] = rel_quat[1:]  # Skip w component, use x,y,z
                    coral_obs[env_idx, 6] = alignment  # Alignment (-1 to 1, where 1 is perfect alignment)
                    coral_obs[env_idx, 7] = height_diff  # Height difference for intake level decisions
                    coral_obs[env_idx, 8] = distance  # Distance to coral
        
        return coral_obs

    def _get_target_socket_observations(self) -> torch.Tensor:
        """Get target socket position, yaw, and distance relative to intake when carrying coral."""
        target_obs = torch.zeros((self.num_envs, 5), device=self.device)  # [rel_x, rel_y, rel_z, yaw, distance]
        
        for env_idx in range(self.num_envs):
            # Only provide target info when holding coral
            if not self.is_holding[env_idx]:
                continue  # Leave as zeros when not holding
                
            # Get intake piece position and orientation
            intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
            if intake_pos is None:
                continue
            
            # Find nearest available socket
            sensor_positions = self._get_reef_sensor_positions(env_idx)
            min_distance = float('inf')
            nearest_socket_pos = None
            
            for sensor_name, sensor_pos in sensor_positions.items():
                # Check if this sensor is already occupied
                sensor_key = (env_idx, sensor_name)
                if sensor_key not in getattr(self, 'sensor_coral_mapping', {}):
                    distance = torch.norm(intake_pos - sensor_pos).item()
                    if distance < min_distance:
                        min_distance = distance
                        nearest_socket_pos = sensor_pos
            
            if nearest_socket_pos is not None:
                # Calculate relative position from intake to target socket
                rel_pos = nearest_socket_pos - intake_pos
                
                # Calculate required yaw angle to face the socket
                # Direction vector from intake to socket (in world frame)
                direction_to_socket = nearest_socket_pos - intake_pos
                
                # Calculate yaw angle (rotation around Z-axis needed to face socket)
                target_yaw = torch.atan2(direction_to_socket[1], direction_to_socket[0])
                
                # Get current intake yaw from quaternion
                roll, pitch, yaw = math_utils.euler_xyz_from_quat(intake_quat.unsqueeze(0))
                current_yaw = yaw[0]  # Extract the yaw component for this environment
                
                # Calculate relative yaw (how much to turn to face socket)
                relative_yaw = target_yaw - current_yaw
                
                # Normalize yaw to [-pi, pi] range
                relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))
                
                # Calculate distance to socket
                distance = torch.norm(rel_pos)
                
                # Store relative position, yaw, and distance
                target_obs[env_idx, :3] = rel_pos
                target_obs[env_idx, 3] = relative_yaw
                target_obs[env_idx, 4] = distance
        
        return target_obs

    def _get_reef_sensor_observations(self) -> torch.Tensor:
        """Get reef sensor positions and alignment relative to robot intake piece."""
        # Return positions and alignment of nearest reef sensors relative to intake piece
        # Format: [nearest_L1_x, nearest_L1_y, nearest_L1_z, nearest_L1_alignment, nearest_L2_x, ...]
        reef_obs = torch.zeros((self.num_envs, 16), device=self.device)  # 4 levels * (3 coords + 1 alignment) = 16
        
        for env_idx in range(self.num_envs):
            # Get intake piece position and orientation
            intake_pos, intake_quat = self._get_intake_piece_pose(env_idx)
            if intake_pos is None or intake_quat is None:
                continue  # Leave observations as zeros for this environment
            
            # Get all reef sensor positions for this environment
            sensor_positions = self._get_reef_sensor_positions(env_idx)
            
            # Find nearest sensor for each level
            level_sensors = {'L1': [], 'L2': [], 'L3': [], 'L4': []}
            
            # Group sensors by level
            for sensor_name, sensor_pos in sensor_positions.items():
                if '_L1' in sensor_name:
                    level_sensors['L1'].append(sensor_pos)
                elif '_L2' in sensor_name:
                    level_sensors['L2'].append(sensor_pos)
                elif '_L3' in sensor_name:
                    level_sensors['L3'].append(sensor_pos)
                elif '_L4' in sensor_name:
                    level_sensors['L4'].append(sensor_pos)
            
            # Find nearest sensor for each level and store relative position
            obs_idx = 0
            for level in ['L1', 'L2', 'L3', 'L4']:
                if level_sensors[level]:
                    # Find nearest sensor of this level
                    min_distance = float('inf')
                    nearest_sensor_pos = None
                    
                    for sensor_pos in level_sensors[level]:
                        distance = torch.norm(intake_pos - sensor_pos)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_sensor_pos = sensor_pos
                    
                    if nearest_sensor_pos is not None:
                        # Calculate relative position from intake to sensor
                        rel_pos_world = nearest_sensor_pos - intake_pos
                        
                        # Transform to intake piece's local frame
                        intake_quat_inv = math_utils.quat_conjugate(intake_quat.unsqueeze(0)).squeeze(0)
                        rel_pos_local = math_utils.quat_apply(intake_quat_inv.unsqueeze(0), rel_pos_world.unsqueeze(0)).squeeze(0)
                        
                        # Calculate alignment: how well the intake is pointing toward the sensor
                        # Intake forward direction (positive Y in intake's local frame)
                        intake_forward_local = torch.tensor([0.0, 1.0, 0.0], device=self.device)  # ✅ Fixed: was [0,0,0]
                        intake_forward_world = math_utils.quat_apply(intake_quat.unsqueeze(0), intake_forward_local.unsqueeze(0)).squeeze(0)
                        
                        # Direction from intake to sensor
                        direction_to_sensor = nearest_sensor_pos - intake_pos
                        direction_to_sensor = direction_to_sensor / (torch.norm(direction_to_sensor) + 1e-8)  # Normalize
                        
                        # Alignment is dot product (cosine of angle between intake forward and direction to sensor)
                        alignment = torch.dot(intake_forward_world, direction_to_sensor)
                        
                        # Store relative position (normalized by distance) and alignment
                        distance_norm = torch.clamp(min_distance, 0.1, 5.0)  # Clamp to reasonable range
                        reef_obs[env_idx, obs_idx:obs_idx+3] = rel_pos_local / distance_norm
                        reef_obs[env_idx, obs_idx+3] = alignment  # Alignment (-1 to 1, where 1 is perfect alignment)
                
                obs_idx += 4  # Move to next level's position slots (3 coords + 1 alignment)
        
        return reef_obs

    def detach_objects(self, joint_prim, sensor_prim, coral_prim):
        """Detach objects by removing the joint between them."""
        try:
            if USD_AVAILABLE:
                from isaaclab.sim import SimulationContext
                sim_context = SimulationContext.instance()
                stage = sim_context.stage
                
                if joint_prim and hasattr(joint_prim, 'GetPath'):
                    stage.RemovePrim(joint_prim.GetPath())
                    print(f"🔓 Detached joint: {joint_prim.GetPath()}")
                    return True
            return False
        except Exception as e:
            print(f"⚠️  Error detaching objects: {e}")
            return False

    # Coral Respawn and Cleanup
    def _respawn_corals(self, env_ids: Sequence[int]):
        """Respawn corals in new random positions for the specified environments."""
        if not hasattr(self, 'coral_objects') or not hasattr(self, 'coral_prim_paths'):
            return

        max_corals = self.cfg.max_corals  # Should match the value used in _setup_scene

        # 🔧 CLEANUP: Break any existing joints/attachments before respawning corals
        self._cleanup_coral_joints(env_ids)
        
        for env_idx in env_ids:
            # Calculate coral indices for this environment
            start_coral_idx = env_idx * max_corals
            
            # Generate new random positions for corals in this environment
            for coral_local_idx in range(max_corals):
                coral_global_idx = start_coral_idx + coral_local_idx
                
                if coral_global_idx < len(self.coral_objects.data.root_pos_w):

                    # Generate new random position near Coral StationS
                    if(coral_local_idx % 2 == 0):
                        random_x = random.uniform(-7.5, -8.5)
                        random_y = random.uniform(-3.0, -1.0)
                    else:
                        random_x = random.uniform(-7.5, -8.5)
                        random_y = random.uniform(1.0, 3.0)
                    z = 0.5
                    
                    # Get environment origin offset
                    env_origin = self.scene.env_origins[env_idx]
                    new_pos = torch.tensor([random_x, random_y, z], device=self.device) + env_origin
                    
                    # Set new coral position
                    self.coral_objects.data.root_pos_w[coral_global_idx] = new_pos
                    
                    # Reset coral velocity to zero
                    if hasattr(self.coral_objects.data, 'root_lin_vel_w'):
                        self.coral_objects.data.root_lin_vel_w[coral_global_idx] = torch.zeros(3, device=self.device)
                    if hasattr(self.coral_objects.data, 'root_ang_vel_w'):
                        self.coral_objects.data.root_ang_vel_w[coral_global_idx] = torch.zeros(3, device=self.device)
        
        # Write the new positions to simulation
        if hasattr(self.coral_objects, 'write_root_pose_to_sim'):
            # Convert env_ids to tensor if it's not already one
            if isinstance(env_ids, torch.Tensor):
                coral_env_ids = env_ids.to(device=self.device)
            else:
                coral_env_ids = torch.tensor(env_ids, device=self.device)
            coral_indices = []
            for env_idx in env_ids:
                for c in range(max_corals):
                    coral_indices.append(env_idx * max_corals + c)
            
            if coral_indices:
                coral_indices_tensor = torch.tensor(coral_indices, device=self.device)
                # Get positions and orientations for the corals to respawn
                coral_poses = torch.zeros((len(coral_indices), 7), device=self.device)
                for i, coral_idx in enumerate(coral_indices):
                    if coral_idx < len(self.coral_objects.data.root_pos_w):
                        coral_poses[i, :3] = self.coral_objects.data.root_pos_w[coral_idx]
                        coral_poses[i, 3:] = self.coral_objects.data.root_quat_w[coral_idx]
                
                try:
                    self.coral_objects.write_root_pose_to_sim(coral_poses, coral_indices_tensor)
                except Exception as e:
                    print(f"Warning: Could not write coral poses to simulation: {e}")

    def _cleanup_coral_joints(self, env_ids: Sequence[int]):
        """Clean up any existing coral joints/attachments for the specified environments."""
        if not hasattr(self, 'coral_joints'):
            self.coral_joints = {}  # Initialize joint tracking if not exists
            return
            
        # Initialize sensor mapping if not exists
        if not hasattr(self, 'sensor_coral_mapping'):
            self.sensor_coral_mapping = {}
        
        # Remove joints for corals in the specified environments
        max_corals = 10
        joints_to_remove = []
        
        for env_idx in env_ids:
            for coral_local_idx in range(max_corals):
                coral_global_idx = env_idx * max_corals + coral_local_idx
                joint_key = f"coral_{coral_global_idx}"
                
                if joint_key in self.coral_joints:
                    joint_info = self.coral_joints[joint_key]
                    joints_to_remove.append((joint_key, joint_info))
        
        # Clear sensor mappings for these environments
        mapping_keys_to_remove = []
        for (env_idx, sensor_name), coral_idx in self.sensor_coral_mapping.items():
            if env_idx in env_ids:
                mapping_keys_to_remove.append((env_idx, sensor_name))
        
        for key in mapping_keys_to_remove:
            del self.sensor_coral_mapping[key]
        
        # Remove joints from USD stage using detach_objects
        if USD_AVAILABLE and joints_to_remove:
            try:
                from isaaclab.sim import SimulationContext
                sim_context = SimulationContext.instance()
                stage = sim_context.stage
                
                for joint_key, joint_info in joints_to_remove:
                    # Handle different joint storage formats
                    if isinstance(joint_info, dict):
                        joint_prim = joint_info.get('joint_prim')
                        coral_prim = joint_info.get('coral_prim')
                        sensor_prim = joint_info.get('sensor_prim')
                        
                        if joint_prim and coral_prim and sensor_prim:
                            self.detach_objects(joint_prim, sensor_prim, coral_prim)
                        del self.coral_joints[joint_key]
                    elif joint_info and hasattr(joint_info, 'IsValid') and joint_info.IsValid():
                        # Fallback for direct joint prim
                        stage.RemovePrim(joint_info.GetPath())
                        del self.coral_joints[joint_key]
                    else:
                        if joint_key in self.coral_joints:
                            del self.coral_joints[joint_key]
                
            except Exception as e:
                # Clear tracking even if USD removal fails
                for joint_key, _ in joints_to_remove:
                    if joint_key in self.coral_joints:
                        del self.coral_joints[joint_key]
        else:
            # Clear tracking for simulation without USD
            for joint_key, _ in joints_to_remove:
                if joint_key in self.coral_joints:
                    del self.coral_joints[joint_key]

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # For now, never terminate episodes
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if the robot is on its side (i.e., base 'up' vector has low Z component)
        # Compute base up vector in world frame by rotating body-space up (0,0,1)
        body_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1)
        up_world = math_utils.quat_apply(self.robot.data.root_link_quat_w, body_up)
        up_z = up_world[:, 2]
        tipped_threshold = 0.5
        terminated = torch.where(up_z < tipped_threshold, -1.0, 0.0)



        return terminated, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)



        # Reset to default joint positions and velocities
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        
        # Reset tracking variables for the reset environments
        self.is_holding[env_ids] = False
        self.previous_is_holding[env_ids] = False
        self.prev_coral_distance[env_ids] = float('inf')
        self.prev_socket_distance[env_ids] = float('inf')
        self.prev_reef_distance[env_ids] = float('inf')
        self.holding_coral_idx[env_ids] = -1
        
        # ✅ Reset phase system for reset environments
        self.current_phase[env_ids] = 0  # Start in SEEK phase
        self.reset_timer[env_ids] = 0.0  # Clear reset timers

        # Reset root state to default position
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Write states to simulation
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        
        # Reset surface gripper state for specified environments
        self.surface_gripper.reset()  # Reset gripper to open state
        
        # Respawn corals in new random positions for reset environments
        self._respawn_corals(env_ids)
        
        # Reset coral spawning timers for the reset environments
        if hasattr(self, 'coral_spawn_timer'):
            self.coral_spawn_timer[env_ids] = 0.0
            
        # Reset FRC scoring for reset environments
        for env_idx in env_ids:
            self.total_score[env_idx] = 0.0
            self.level_counts['L1'][env_idx] = 0.0
            self.level_counts['L2'][env_idx] = 0.0
            self.level_counts['L3'][env_idx] = 0.0
            self.level_counts['L4'][env_idx] = 0.0
            
        # Clear welding events for reset environments
        if hasattr(self, 'welding_events'):
            env_ids_list = list(env_ids) if hasattr(env_ids, '__iter__') else [env_ids]
            keys_to_remove = [key for key in self.welding_events.keys() 
                            if any(f"env_{env_id}" in key for env_id in env_ids_list)]
            for key in keys_to_remove:
                del self.welding_events[key]
            
            # Clear debug flags for reset environments
            for env_id in env_ids_list:
                debug_attr = f'_debugged_env_{env_id}'
                warn_attr = f'_warned_env_{env_id}'
                if hasattr(self, debug_attr):
                    delattr(self, debug_attr)
                if hasattr(self, warn_attr):
                    delattr(self, warn_attr)


    def debug_sensor_info(self):
        """Debug method to display information about all detected reef sensors."""
        print("\n" + "="*80)
        print(" REEF SENSOR DEBUG INFORMATION")
        print("="*80)

        print(" SENSOR PATH STRUCTURE:")
        print("   Base Path: /World/envs/env_X/reefscape/REEFSCAPE_FIELD__FE_2025___1_/BlueReefSensor/")
        print("   Sensor Pattern: blue[A-L]L\\d+ (e.g., blueAL1, blueBL2, blueCL3, etc.)")
        print("   Regex: ^(blue)[A-L]L\\d+$")

        print("\n DETECTED SENSORS:")

        # Show cache status
        cache_size = len(self._sensor_positions_cache)
        print(f"   Cache Status: {cache_size} environments cached")

        # Get sensor positions for each environment
        for env_idx in range(self.num_envs):
            print(f"\n   Environment {env_idx}:")
            sensor_positions = self._get_reef_sensor_positions(env_idx)

            if not sensor_positions:
                print("    No sensors detected")
                continue

            print(f"      ✅ Found {len(sensor_positions)} blue sensors:")

            for sensor_name, position in sensor_positions.items():
                print(f"         📍 {sensor_name}: {position}")

        print("\nSUMMARY:")
        total_sensors = 0
        for env_idx in range(self.num_envs):
            sensor_positions = self._get_reef_sensor_positions(env_idx)
            total_sensors += len(sensor_positions)

        print(f"   Total Environments: {self.num_envs}")
        print(f"   Total Blue Sensors: {total_sensors}")
        print(f"   Average Sensors per Environment: {total_sensors / self.num_envs:.1f}")