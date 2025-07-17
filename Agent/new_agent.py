"""
Advanced Coral-Focused FRC Agent for Reefscape Game

This agent implements sophisticated reward shaping and control logic to:
1. Focus exclusively on coral pieces (higher value than algae)
2. Use front-facing intake with proper alignment
3. Navigate efficiently with heading alignment
4. Score at reef with mechanism positioning
5. Handle time pressure and multi-objective optimization

Based on advanced reward shaping principles with heading alignment,
mechanism positioning, drop penalties, and time pressure.
"""
import time
import numpy as np
import logging
from typing import List, Tuple
import ntcore
import heapq
from basic_agent import BasicFRCAgent
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

logger = logging.getLogger(__name__)

def normalize_angle(angle):
    """Normalize any angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class HeuristicAgent(BasicFRCAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # === STATE INDICES (Verified from Robot.java and basic_agent.py) ===
        self.state_indices = {
            # Basic robot state (0-10)
            'x': 0, 'y': 1, 'heading': 2,
            'vel_x': 3, 'vel_y': 4, 'ang_vel': 5,
            'elevator': 6, 'arm_angle': 7, 'at_setpoint': 8,
            'has_piece': 9, 'timestamp': 10,  # CORRECTED: has_piece is at index 9
            
            # Game state (11-24)
            'algae_count': 11, 'coral_count': 12,
            'algae_dist': 13, 'algae_x': 14, 'algae_y': 15,
            'coral_dist': 16, 'coral_x': 17, 'coral_y': 18,
            'red_score': 19, 'blue_score': 20, 'score_diff': 21,
            'red_reef_dist': 22, 'blue_reef_dist': 23, 'nearest_reef_dist': 24,
            
            # Orientation state (25-26)
            'intake_alignment': 25, 'heading_sin': 26
        }
        
        # === CORE GAME PARAMETERS ===
        self.reef_center = np.array([8.0, 4.0])  # Updated to correct reef center
        self.coral_only = True  # Focus exclusively on coral (higher value)
        
        # === REEF WALL PARAMETERS ===
        # Reef dimensions (approximate FRC 2025 reef structure)
        self.reef_width = 2.0   # Width of reef structure
        self.reef_height = 1.5  # Height of reef structure
        self.wall_detection_distance = 4.0  # How close to wall to trigger ejection (increased)
        
        # === CONTROL THRESHOLDS ===
        self.alignment_tolerance = 0.15  # Radians (~8.6 degrees) for heading alignment
        self.intake_distance = 0.8       # Distance to start intake approach
        self.scoring_distance = 5.0      # Distance to start scoring approach (increased)
        self.eject_distance = 5.0        # Distance to start ejecting (increased to allow reef approach)
        self.speed_scale = 1.5          # Base movement speed multiplier (reduced from 3.5)
        self.rotation_scale = 1.5       # Base rotation speed multiplier (reduced from 2.8)
        
        # === REWARD WEIGHTS (Advanced Shaping) ===
        # Core objectives
        self.k1_approach = 2.0      # Coral approach reward
        self.k2_pickup = 15.0       # Coral pickup bonus (increased for coral focus)
        self.k3_goal_approach = 2.0 # Reef approach with coral
        self.k4_deposit = 25.0      # Scoring bonus (increased for coral focus)
        
        # Advanced shaping
        self.k5_heading = 1.5       # Heading alignment reward
        self.k6_mechanism = 3.0     # Elevator/arm positioning reward
        self.p5_drop = 8.0          # Drop piece penalty
        self.p3_time = 0.02         # Base time penalty
        self.alpha_time = 0.5       # Time pressure multiplier
        
        # Movement and efficiency
        self.p1_collision = 5.0     # Collision penalty
        self.p2_turn = 0.3          # Turn rate penalty
        self.move_reward = 0.08     # Movement reward scale
        self.pause_penalty = 0.05   # Stationary penalty
        
        # === MECHANISM PARAMETERS ===
        self.desired_elevator_height = 2  # Optimal height for reef scoring
        self.elevator_tolerance = 0.5    # Tolerance for "good" positioning
        self.intake_direction = 1.0         # 1.0 for intake, -1.0 for eject
        
        # === EPISODE TRACKING ===
        self.episode_start_time = 0.0
        self.teleop_duration = 120.0  # 2 minutes for realistic time pressure
        
        # === COLLISION DETECTION AND MAPPING ===
        self.position_history = []  # Track recent positions for stuck detection
        self.velocity_history = []  # Track recent velocities
        self.history_length = 10   # Number of recent states to track
        self.stuck_threshold = 0.05  # Minimum movement required to not be considered stuck (reduced from 0.1)
        self.velocity_threshold = 0.8  # Minimum velocity to consider movement intended (increased from 0.2)
        
        # Pseudo map for obstacle detection (grid-based)
        self.map_resolution = 0.1  # meters per grid cell
        self.map_width = int(16.54 / self.map_resolution) + 1  # Field width in grid cells
        self.map_height = int(8.21 / self.map_resolution) + 1  # Field length in grid cells
        self.obstacle_map = np.zeros((self.map_height, self.map_width))  # 0 = unknown, 1 = obstacle, -1 = free
        self.obstacle_confidence = np.zeros((self.map_height, self.map_width))  # Confidence in obstacle detection
        
        # Pathfinding parameters
        self.path_cache = {}  # Cache computed paths
        self.path_cache_max_size = 100
        self.current_path = []  # Current path being followed
        self.path_target = None  # Current pathfinding target
        self.path_recalc_threshold = 1.0  # Recalculate path if target moves this much
        self.stuck_waypoint_threshold = 250  # Max attempts at same waypoint before abandoning
        self.waypoint_attempts = 0  # Counter for current waypoint attempts
        self.last_waypoint = None  # Track last attempted waypoint
        self.emergency_escape_mode = False  # Flag for emergency random movement
        self.emergency_escape_counter = 0  # Counter for emergency mode duration
        
        # Directional stuck detection
        self.direction_attempts = {}  # Track attempts in each direction
        self.max_direction_attempts = 3  # Max attempts in same direction before reversing
        self.direction_history = []  # Track recent movement directions
        self.direction_history_length = 25
        self.last_stuck_direction = None  # Remember which direction got us stuck
        
        # Desperation mode for when completely stuck
        self.desperation_counter = 0  # Count consecutive stuck detections
        self.desperation_threshold = 25  # Enter desperation mode after this many stuck detections (increased)
        self.desperation_mode = False  # Flag for desperation mode
        self.last_position_for_desperation = None  # Track position for desperation detection
        
        # === VELOCITY SMOOTHING ===
        self.last_velocity_command = [0.0, 0.0, 0.0]  # [vx, vy, angular]
        self.velocity_smoothing_factor = 0.5  # How much to blend with previous command (0-1)
        self.max_velocity_change = 0.7  # Maximum change in velocity per timestep (m/s)
        self.max_angular_change = 0.9   # Maximum change in angular velocity per timestep (rad/s)
        self.command_hold_time = 5      # Hold same command for this many timesteps
        self.command_hold_counter = 0   # Counter for holding commands
        
        # === NETWORKTABLES CONNECTION ===
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.control_table = None
        self.state_table = None
        self.connected = False
        
        logger.info("Advanced Coral-Focused Agent initialized")
        logger.info(f"Focus: Coral only={self.coral_only}, Reef center={self.reef_center}")
        logger.info(f"Thresholds: align={self.alignment_tolerance:.2f}, intake_dist={self.intake_distance:.2f}")

    def _smooth_velocity_commands(self, vx: float, vy: float, ang_vel: float) -> Tuple[float, float, float]:
        """
        Smooth velocity commands to prevent rapid changes that confuse the robot.
        
        Args:
            vx, vy, ang_vel: Raw velocity commands
            
        Returns:
            Smoothed velocity commands
        """
        # Skip smoothing in desperation mode to allow sharp movements
        if self.desperation_mode:
            logger.debug(f"DESPERATION: Bypassing velocity smoothing")
            self.last_velocity_command = [vx, vy, ang_vel]
            return vx, vy, ang_vel
        
        # Get previous commands
        prev_vx, prev_vy, prev_ang = self.last_velocity_command
        
        # Calculate velocity changes
        dvx = vx - prev_vx
        dvy = vy - prev_vy
        dang = ang_vel - prev_ang
        
        # Limit velocity changes per timestep
        if abs(dvx) > self.max_velocity_change:
            dvx = np.sign(dvx) * self.max_velocity_change
        if abs(dvy) > self.max_velocity_change:
            dvy = np.sign(dvy) * self.max_velocity_change
        if abs(dang) > self.max_angular_change:
            dang = np.sign(dang) * self.max_angular_change
        
        # Apply rate-limited changes
        smooth_vx = prev_vx + dvx
        smooth_vy = prev_vy + dvy
        smooth_ang = prev_ang + dang
        
        # Optional: Apply smoothing filter (blend with previous command)
        smooth_vx = prev_vx * self.velocity_smoothing_factor + smooth_vx * (1 - self.velocity_smoothing_factor)
        smooth_vy = prev_vy * self.velocity_smoothing_factor + smooth_vy * (1 - self.velocity_smoothing_factor)
        smooth_ang = prev_ang * self.velocity_smoothing_factor + smooth_ang * (1 - self.velocity_smoothing_factor)
        
        # Store for next iteration
        self.last_velocity_command = [smooth_vx, smooth_vy, smooth_ang]
        
        logger.debug(f"SMOOTH: Raw ({vx:.2f},{vy:.2f},{ang_vel:.2f}) -> "
                    f"Smooth ({smooth_vx:.2f},{smooth_vy:.2f},{smooth_ang:.2f})")
        
        return smooth_vx, smooth_vy, smooth_ang

    def _safe_state_access(self, state: np.ndarray, key: str, default=0.0) -> float:
        """Safely access state values using verified indices."""
        index = self.state_indices.get(key, -1)
        return state[index] if 0 <= index < len(state) else default

    def _field_to_robot_centric(self, vx_field: float, vy_field: float, heading: float) -> Tuple[float, float]:
        """Converts field-centric velocities to robot-centric."""
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        vx_robot = vx_field * cos_h + vy_field * sin_h
        vy_robot = -vx_field * sin_h + vy_field * cos_h
        return vx_robot, vy_robot

    def _get_target_coral(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Get the coordinates and distance of target coral (coral-only focus)."""
        coral_x = self._safe_state_access(state, 'coral_x')
        coral_y = self._safe_state_access(state, 'coral_y')
        coral_dist = self._safe_state_access(state, 'coral_dist', np.inf)
        
        return coral_x, coral_y, coral_dist

    def _detect_reef_wall_proximity(self, x: float, y: float, heading: float) -> Tuple[bool, float]:
        """
        Detect if robot is close to reef wall and properly positioned for ejection.
        
        Returns:
            (is_at_wall, distance_to_wall): Boolean indicating wall proximity and actual distance
        """
        reef_x, reef_y = self.reef_center
        
        # Calculate robot's position relative to reef center
        dx = x - reef_x
        dy = y - reef_y
        
        # Calculate distance to reef edges (assuming rectangular reef)
        half_width = self.reef_width / 2.0
        half_height = self.reef_height / 2.0
        
        # Find closest point on reef perimeter
        closest_x = np.clip(dx, -half_width, half_width)
        closest_y = np.clip(dy, -half_height, half_height)
        
        # If robot is inside reef bounds, find distance to nearest edge
        if abs(dx) <= half_width and abs(dy) <= half_height:
            # Robot is inside reef area - find distance to nearest wall
            dist_to_left = dx + half_width
            dist_to_right = half_width - dx
            dist_to_bottom = dy + half_height  
            dist_to_top = half_height - dy
            
            wall_distance = min(dist_to_left, dist_to_right, dist_to_bottom, dist_to_top)
        else:
            # Robot is outside reef - calculate distance to closest point on perimeter
            wall_distance = np.hypot(dx - closest_x, dy - closest_y)
        
        # Check if robot is close enough to wall and facing roughly toward reef
        is_at_wall = wall_distance <= self.wall_detection_distance
        
        # Additional check: robot should be facing generally toward reef center for ejection
        angle_to_reef = np.arctan2(reef_y - y, reef_x - x)
        heading_diff = abs(normalize_angle(angle_to_reef - heading))
        facing_reef = heading_diff < (np.pi / 2)  # Within 90 degrees of reef direction
        
        return is_at_wall and facing_reef, wall_distance

    def select_action(self, state: np.ndarray) -> List[float]:
        """
        Advanced action selection with coral focus and proper alignment.
        """
        # DESPERATION MODE: If completely stuck, use very simple movement
        if self.desperation_mode:
            return self._desperation_mode_action(state)
        
        has_piece = self._safe_state_access(state, 'has_piece') > 0.5
        
        if has_piece:
            return self._scoring_phase_action(state)
        else:
            return self._seeking_phase_action(state)

    def _desperation_mode_action(self, state: np.ndarray) -> List[float]:
        """
        Simple desperation mode when completely stuck - bypasses all complex logic.
        Uses very basic movement commands to try to break free.
        """
        
        x = self._safe_state_access(state, 'x')
        y = self._safe_state_access(state, 'y')
        
        logger.warning(f"DESPERATION: Using simple movement at ({x:.2f}, {y:.2f})")
        
        # Try different desperation strategies
        desperation_strategy = self.desperation_counter % 4
        
        if desperation_strategy == 0:
            # Strategy 1: Move away from walls toward field center
            field_center_x, field_center_y = 8.27, 4.105  # FRC field center
            dx = field_center_x - x
            dy = field_center_y - y
            
            # Normalize and scale to moderate speed
            dist = max(np.hypot(dx, dy), 0.1)
            vx = (dx / dist) * 1.0
            vy = (dy / dist) * 1.0
            
            logger.warning(f"DESPERATION: Moving toward field center ({vx:.2f}, {vy:.2f})")
            return [vx, vy, 0.0, 0.0, 0.0, 0.0]
        
        elif desperation_strategy == 1:
            # Strategy 2: Try backing up
            vx = random.uniform(-1.5, -0.5)
            vy = random.uniform(-1.5, 1.5)
            
            logger.warning(f"DESPERATION: Backing up ({vx:.2f}, {vy:.2f})")
            return [vx, vy, 0.0, 0.0, 0.0, 0.0]
        
        elif desperation_strategy == 2:
            # Strategy 3: Rotate and move
            angle = random.uniform(0, 2 * np.pi)
            vx = np.cos(angle) * 0.8
            vy = np.sin(angle) * 0.8
            ang_vel = random.uniform(-1.0, 1.0)
            
            logger.warning(f"DESPERATION: Rotate and move ({vx:.2f}, {vy:.2f}, {ang_vel:.2f})")
            return [vx, vy, ang_vel, 0.0, 0.0, 0.0]
        
        else:
            # Strategy 4: Pure rotation to break orientation lock
            ang_vel = random.choice([-2.0, 2.0])
            
            logger.warning(f"DESPERATION: Pure rotation ({ang_vel:.2f})")
            return [0.0, 0.0, ang_vel, 0.0, 0.0, 0.0]

    def _seeking_phase_action(self, state: np.ndarray) -> List[float]:
        """
        Handle seeking coral with front-facing intake alignment.
        Now allows simultaneous rotation and movement for fluid motion.
        Includes wall avoidance to prevent getting stuck.
        """
        x = self._safe_state_access(state, 'x')
        y = self._safe_state_access(state, 'y')
        heading = self._safe_state_access(state, 'heading')
        
        # Get target coral
        tx, ty, dist = self._get_target_coral(state)
        
        if np.isinf(dist):
            logger.debug("No coral available, using exploration policy")
            # Exploration policy is robot-centric, so no conversion needed
            return self.random_policy()
        
        # Calculate direction and alignment
        dx, dy = tx - x, ty - y
        target_angle = np.arctan2(dy, dx)
        angle_diff = normalize_angle(target_angle - heading)
        
        # Use intake alignment from state (front-facing)
        intake_alignment = self._safe_state_access(state, 'intake_alignment', np.cos(angle_diff))
        is_aligned = abs(angle_diff) < self.alignment_tolerance
        
        logger.debug(f"SEEKING: coral at ({tx:.2f}, {ty:.2f}), dist={dist:.2f}, "
                    f"angle_diff={angle_diff:.2f}, aligned={is_aligned}")
        
        # Calculate base movement toward coral
        base_vx = np.clip(dx * self.speed_scale, -2.0, 2.0)  # Reduced max speed
        base_vy = np.clip(dy * self.speed_scale, -2.0, 2.0)  # Reduced max speed
        
        # Check for wall proximity and get avoidance velocities
        near_wall, avoid_vx, avoid_vy = self._detect_wall_proximity_and_avoid(x, y, base_vx, base_vy)
        
        # Combine coral-seeking and wall avoidance
        if near_wall:
            logger.debug(f"SEEKING: Near wall, applying avoidance: avoid_vx={avoid_vx:.2f}, avoid_vy={avoid_vy:.2f}")
            
            # Check if we need pathfinding for complex navigation
            if abs(avoid_vx) > 2.0 or abs(avoid_vy) > 2.0:
                # Strong avoidance needed - use pathfinding to coral
                logger.debug(f"SEEKING: Using pathfinding to coral due to obstacles")
                path_vx, path_vy = self._get_pathfinding_velocity(x, y, tx, ty)
                vx = path_vx * 0.7  # Scale down for safety
                vy = path_vy * 0.7
            else:
                # Mild avoidance - blend coral movement with wall avoidance
                vx = base_vx * 0.7 + avoid_vx * 0.3
                vy = base_vy * 0.7 + avoid_vy * 0.3
        else:
            vx = base_vx
            vy = base_vy
        
        # Calculate rotation to align with coral
        ang_vel = np.clip(angle_diff * self.rotation_scale, -1.5, 1.5)  # Reduced max rotation
        
        # Apply velocity smoothing to prevent erratic movement
        smooth_vx, smooth_vy, smooth_ang_vel = self._smooth_velocity_commands(vx, vy, ang_vel)
        
        # Determine final command based on distance
        if dist > self.intake_distance + 1.0:  # Beyond intake + buffer zone
            final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, smooth_ang_vel
            intake_cmd = 0.0
        elif dist > self.intake_distance:
            # Moderate speed with rotation for better control
            final_vx = smooth_vx * 0.7
            final_vy = smooth_vy * 0.7
            final_ang_vel = smooth_ang_vel * 0.7
            intake_cmd = 0.0
        else:
            # Slow approach with intake running and continued alignment
            final_vx = smooth_vx * 0.4  # Slower for precision
            final_vy = smooth_vy * 0.4
            # Reduce rotation speed when very close for stability
            final_ang_vel = smooth_ang_vel * 0.5
            intake_cmd = self.intake_direction

        # Convert field-centric velocity to robot-centric
        vx_robot, vy_robot = self._field_to_robot_centric(final_vx, final_vy, heading)

        return [vx_robot, vy_robot, final_ang_vel, 0.0, 0.0, intake_cmd]

    def _scoring_phase_action(self, state: np.ndarray) -> List[float]:
        """
        Handle scoring with reef wall alignment and mechanism positioning.
        Includes wall avoidance to prevent getting stuck.
        """
        x = self._safe_state_access(state, 'x')
        y = self._safe_state_access(state, 'y')
        heading = self._safe_state_access(state, 'heading')
        elevator = self._safe_state_access(state, 'elevator')
        
        # Target reef center for approach
        tx, ty = self.reef_center
        dx, dy = tx - x, ty - y
        dist = np.hypot(dx, dy)
        
        target_angle = np.arctan2(dy, dx)
        angle_diff = normalize_angle(target_angle - heading)
        is_aligned = abs(angle_diff) < self.alignment_tolerance
        
        # Check elevator readiness
        elevator_ready = abs(elevator - self.desired_elevator_height) < self.elevator_tolerance
        
        # Check reef wall proximity for ejection
        at_reef_wall, wall_distance = self._detect_reef_wall_proximity(x, y, heading)
        
        # Check for field wall proximity and get avoidance velocities
        near_field_wall, avoid_vx, avoid_vy = self._detect_wall_proximity_and_avoid(x, y, 0.0, 0.0)
        
        logger.debug(f"SCORING: reef at ({tx:.2f}, {ty:.2f}), dist={dist:.2f}, "
                    f"angle_diff={angle_diff:.2f}, aligned={is_aligned}, "
                    f"elevator={elevator:.2f}, target={self.desired_elevator_height:.2f}, ready={elevator_ready}, "
                    f"at_wall={at_reef_wall}, wall_dist={wall_distance:.2f}, near_field_wall={near_field_wall}")
        
        # Initialize action components
        final_vx, final_vy, final_ang_vel = 0.0, 0.0, 0.0
        elevator_cmd = self.desired_elevator_height
        arm_cmd = 0.0
        intake_cmd = 0.0

        # Calculate base movement toward reef
        base_vx = np.clip(dx * self.speed_scale * 0.6, -2.0, 2.0)  # Reduced speed and max
        base_vy = np.clip(dy * self.speed_scale * 0.6, -2.0, 2.0)  # Reduced speed and max
        base_ang_vel = np.clip(angle_diff * self.rotation_scale * 0.8, -1.5, 1.5)  # Reduced rotation
        
        # Phase 1: Long range approach to reef
        if dist > self.scoring_distance:
            logger.debug("SCORING: Phase 1 - Long range approach")
            # Apply wall avoidance if near field walls
            if near_field_wall:
                vx = base_vx * 0.5 + avoid_vx * 0.5
                vy = base_vy * 0.5 + avoid_vy * 0.5
            else:
                vx = base_vx * 0.8
                vy = base_vy * 0.8
            
            # Apply velocity smoothing
            smooth_vx, smooth_vy, smooth_ang_vel = self._smooth_velocity_commands(vx, vy, base_ang_vel)
            
            if not is_aligned:
                # Rotate while approaching
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, smooth_ang_vel
            else:
                # Move toward reef
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, 0.0
        
        # Phase 2: Medium range approach - get closer to reef
        elif dist > self.eject_distance and not at_reef_wall:
            logger.debug("SCORING: Phase 2 - Medium range approach")
            # Apply wall avoidance if near field walls
            if near_field_wall:
                vx = base_vx * 0.4 + avoid_vx * 0.3
                vy = base_vy * 0.4 + avoid_vy * 0.3
            else:
                vx = base_vx * 0.6
                vy = base_vy * 0.6
            
            # Apply velocity smoothing
            smooth_vx, smooth_vy, smooth_ang_vel = self._smooth_velocity_commands(vx, vy, base_ang_vel)
            
            if not is_aligned:
                # Fine alignment while approaching
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, smooth_ang_vel * 0.8
            else:
                # Slow approach to reef wall
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, 0.0
        
        # Phase 3: Close to reef but not at wall - final approach to wall
        elif not at_reef_wall:
            logger.debug(f"SCORING: Phase 3 - Close approach to wall, current wall distance: {wall_distance:.2f}")
            
            # Apply wall avoidance if near field walls (more conservative near reef)
            if near_field_wall:
                vx = base_vx * 0.3 + avoid_vx * 0.2
                vy = base_vy * 0.3 + avoid_vy * 0.2
            else:
                vx = base_vx * 0.5  # Moderate speed for controlled approach
                vy = base_vy * 0.5
            
            # Apply velocity smoothing
            smooth_vx, smooth_vy, smooth_ang_vel = self._smooth_velocity_commands(vx, vy, base_ang_vel)
            
            if not is_aligned:
                # Fine alignment while moving to wall
                logger.debug(f"SCORING: Phase 3 movement: vx={smooth_vx:.2f}, vy={smooth_vy:.2f}, ang_vel={smooth_ang_vel:.2f}")
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, smooth_ang_vel * 0.6
            else:
                # Controlled approach to wall
                logger.debug(f"SCORING: Phase 3 approach: vx={smooth_vx:.2f}, vy={smooth_vy:.2f}")
                final_vx, final_vy, final_ang_vel = smooth_vx, smooth_vy, 0.0
        
        # Phase 4: At reef wall - final alignment check
        elif not is_aligned:
            logger.debug("SCORING: Phase 4 - Final alignment")
            # Use smoothing for alignment rotation only
            _, _, smooth_ang_vel = self._smooth_velocity_commands(0.0, 0.0, base_ang_vel)
            final_vx, final_vy, final_ang_vel = 0.0, 0.0, smooth_ang_vel * 0.8
        
        # Phase 5: EJECT - All conditions met (at wall, aligned, elevator ready)
        elif at_reef_wall and elevator_ready and is_aligned:
            logger.info(f"SCORING: AT REEF WALL - EJECTING! wall_dist={wall_distance:.2f}, "
                       f"aligned={is_aligned}, elevator_ready={elevator_ready}")
            eject_command = -self.intake_direction
            logger.info(f"EJECT DEBUG: Sending intake_command={eject_command:.2f} (intake_direction={self.intake_direction:.2f})")
            final_vx, final_vy, final_ang_vel = 0.0, 0.0, 0.0
            intake_cmd = eject_command
        else:
            # Wait for all conditions
            reasons = []
            if not at_reef_wall:
                reasons.append(f"not at reef wall (wall_dist={wall_distance:.2f} > {self.wall_detection_distance:.2f})")
            if not elevator_ready:
                reasons.append(f"elevator not ready ({elevator:.2f} vs {self.desired_elevator_height:.2f})")
            if not is_aligned:
                reasons.append(f"not aligned ({angle_diff:.2f} rad)")
            
            logger.debug(f"SCORING: Waiting - {', '.join(reasons)}")
            final_vx, final_vy, final_ang_vel = 0.0, 0.0, 0.0

        # Convert field-centric velocities to robot-centric
        vx_robot, vy_robot = self._field_to_robot_centric(final_vx, final_vy, heading)

        return [vx_robot, vy_robot, final_ang_vel, elevator_cmd, arm_cmd, intake_cmd]

    def calculate_advanced_reward(self, prev_state: np.ndarray, action: List[float], 
                                 next_state: np.ndarray) -> float:
        """
        Advanced reward calculation incorporating all sophisticated shaping terms.
        """
        reward = 0.0
        
        # Extract action components
        vx, vy, ang_vel = action[0], action[1], action[2]
        elevator_cmd = action[3] if len(action) > 3 else 0.0
        
        # Extract state information
        prev_has_piece = self._safe_state_access(prev_state, 'has_piece') > 0.5
        has_piece = self._safe_state_access(next_state, 'has_piece') > 0.5
        prev_score = self._safe_state_access(prev_state, 'score_diff')
        curr_score = self._safe_state_access(next_state, 'score_diff')
        
        # === 1. CORAL APPROACH REWARD (only when seeking) ===
        if not prev_has_piece and self.coral_only:
            prev_coral_dist = self._safe_state_access(prev_state, 'coral_dist', np.inf)
            curr_coral_dist = self._safe_state_access(next_state, 'coral_dist', np.inf)
            
            if not np.isinf(prev_coral_dist) and not np.isinf(curr_coral_dist):
                approach_progress = prev_coral_dist - curr_coral_dist
                reward += self.k1_approach * max(0.0, approach_progress)
        
        # === 2. CORAL PICKUP BONUS ===
        if has_piece and not prev_has_piece:
            reward += self.k2_pickup
            logger.debug(f"REWARD: +{self.k2_pickup} for coral pickup")
        
        # === 3. REEF APPROACH REWARD (only when carrying) ===
        if has_piece:
            prev_x = self._safe_state_access(prev_state, 'x')
            prev_y = self._safe_state_access(prev_state, 'y')
            curr_x = self._safe_state_access(next_state, 'x')
            curr_y = self._safe_state_access(next_state, 'y')
            
            prev_reef_dist = np.hypot(prev_x - self.reef_center[0], prev_y - self.reef_center[1])
            curr_reef_dist = np.hypot(curr_x - self.reef_center[0], curr_y - self.reef_center[1])
            
            approach_progress = prev_reef_dist - curr_reef_dist
            reward += self.k3_goal_approach * max(0.0, approach_progress)
        
        # === 4. SCORING BONUS ===
        if curr_score > prev_score:
            reward += self.k4_deposit
            logger.debug(f"REWARD: +{self.k4_deposit} for scoring coral")
        
        # === 5. HEADING ALIGNMENT BONUS ===
        # Determine target based on current phase
        if has_piece:
            target_x, target_y = self.reef_center
        else:
            target_x = self._safe_state_access(next_state, 'coral_x')
            target_y = self._safe_state_access(next_state, 'coral_y')
        
        if not np.isinf(target_x):
            curr_x = self._safe_state_access(next_state, 'x')
            curr_y = self._safe_state_access(next_state, 'y')
            curr_heading = self._safe_state_access(next_state, 'heading')
            
            angle_to_target = np.arctan2(target_y - curr_y, target_x - curr_x)
            heading_error = normalize_angle(angle_to_target - curr_heading)
            alignment_quality = 1.0 - abs(heading_error) / np.pi;
            
            reward += self.k5_heading * alignment_quality
        
        # === 6. MECHANISM POSITIONING REWARD ===
        if has_piece:
            curr_elevator = self._safe_state_access(next_state, 'elevator')
            height_error = abs(curr_elevator - self.desired_elevator_height)
            max_height = 2.5  # Assumed max elevator height
            
            if height_error <= self.elevator_tolerance:
                # Perfect positioning bonus
                reward += self.k6_mechanism
            else:
                # Partial credit for approaching correct height
                positioning_quality = 1.0 - (height_error / max_height)
                reward += self.k6_mechanism * positioning_quality * 0.5
        
        # === 7. DROP PIECE PENALTY ===
        if prev_has_piece and not has_piece and curr_score == prev_score:
            reward -= self.p5_drop
            logger.debug(f"PENALTY: -{self.p5_drop} for dropping coral")
        
        # === 8. TIME PRESSURE PENALTY ===
        time_elapsed = time.time() - self.episode_start_time
        time_factor = min(time_elapsed / self.teleop_duration, 1.0)
        step_penalty = self.p3_time + self.alpha_time * time_factor
        reward -= step_penalty
        
        # === 9. MOVEMENT AND EFFICIENCY ===
        # Movement reward
        speed = np.hypot(vx, vy)
        if speed > 0.01:
            reward += self.move_reward * (speed ** 2)
        else:
            reward -= self.pause_penalty
        
        # Turn penalty
        reward -= self.p2_turn * abs(ang_vel)
        
        # Excessive spinning penalty
        if abs(ang_vel) > 0.5 and speed < 0.01:
            reward -= self.p2_turn * abs(ang_vel)
        
        # === 10. COLLISION PENALTY ===
        if hasattr(self, 'detect_wall_collision') and self.detect_wall_collision(prev_state, action, next_state):
            reward -= self.p1_collision
            logger.debug(f"PENALTY: -{self.p1_collision} for collision")
        
        return reward

    def detect_wall_collision(self, prev_state: np.ndarray, action: List[float], 
                            next_state: np.ndarray) -> bool:
        """
        Revised collision and stuck detection logic.
        Focuses on a simpler, more robust definition of "stuck":
        - Is the robot commanding movement?
        - Is the robot's position failing to change over a time window?
        This avoids overly strict single-tick checks.
        """
        # --- State Extraction ---
        curr_x = self._safe_state_access(next_state, 'x')
        curr_y = self._safe_state_access(next_state, 'y')
        cmd_vx, cmd_vy = action[0], action[1]
        cmd_speed = np.hypot(cmd_vx, cmd_vy)
        
        # --- History Management ---
        self.position_history.append((curr_x, curr_y))
        if len(self.position_history) > self.history_length:
            self.position_history.pop(0)

        # We need history to determine if we are stuck
        if len(self.position_history) < self.history_length:
            return False

        # --- Core Stuck Detection ---
        is_stuck = False
        # Condition 1: The agent is actively trying to move.
        is_commanding_move = cmd_speed > self.velocity_threshold

        if is_commanding_move:
            # Condition 2: The robot's position has not changed significantly over the history window.
            start_pos = self.position_history[0]
            end_pos = self.position_history[-1]
            movement_over_history = np.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
            
            if movement_over_history < self.stuck_threshold:
                is_stuck = True
                logger.debug(
                    f"COLLISION: Stuck detected. "
                    f"Movement over last {self.history_length} steps: {movement_over_history:.3f}m "
                    f"(Threshold: {self.stuck_threshold:.3f}m) "
                    f"while commanding speed {cmd_speed:.2f} m/s."
                )
                # Mark the obstacle in the direction of intended movement
                self._update_obstacle_map(curr_x, curr_y, cmd_vx, cmd_vy, is_obstacle=True)

        # --- Boundary Collision Check ---
        # This is a hard check for hitting the physical field walls.
        field_width, field_length = 16.54, 8.21
        wall_buffer = 0.1  # A small buffer to detect wall proximity

        at_boundary = (curr_x < wall_buffer or curr_x > field_width - wall_buffer or
                       curr_y < wall_buffer or curr_y > field_length - wall_buffer)

        if at_boundary:
            # Check if trying to move further out of bounds
            moving_out_of_bounds = (
                (curr_x < wall_buffer and cmd_vx < 0) or
                (curr_x > field_width - wall_buffer and cmd_vx > 0) or
                (curr_y < wall_buffer and cmd_vy < 0) or
                (curr_y > field_length - wall_buffer and cmd_vy > 0)
            )
            if moving_out_of_bounds:
                is_stuck = True
                logger.debug(f"COLLISION: Field boundary hit at ({curr_x:.2f}, {curr_y:.2f}).")
                self._mark_field_boundaries() # Reinforce boundary knowledge

        # --- Update Free Space ---
        # If we are moving successfully, mark the current area as clear.
        if not is_stuck:
            # Check for actual movement to be sure
            prev_x = self._safe_state_access(prev_state, 'x')
            prev_y = self._safe_state_access(prev_state, 'y')
            actual_movement = np.hypot(curr_x - prev_x, curr_y - prev_y)
            if actual_movement > 0.01:
                 self._update_obstacle_map(curr_x, curr_y, cmd_vx, cmd_vy, is_obstacle=False)

        # --- Desperation Mode Logic ---
        # Enter desperation mode if stuck in the same small area for too long.
        if is_stuck:
            if self.last_position_for_desperation is None:
                # First time we've been stuck recently, start tracking
                self.last_position_for_desperation = (curr_x, curr_y)
                self.desperation_counter = 1
            else:
                # Check if we're still stuck in the same spot
                stuck_area_radius = 0.3 # 30cm radius to define the "stuck area"
                dist_from_last_stuck = np.hypot(curr_x - self.last_position_for_desperation[0],
                                                curr_y - self.last_position_for_desperation[1])

                if dist_from_last_stuck < stuck_area_radius:
                    # Still in the same area, increment counter
                    self.desperation_counter += 1
                else:
                    # Got stuck in a new area, reset counter
                    self.last_position_for_desperation = (curr_x, curr_y)
                    self.desperation_counter = 1
            
            # Check if we've crossed the threshold to enter desperation mode
            if self.desperation_counter >= self.desperation_threshold:
                if not self.desperation_mode:
                    logger.warning(
                        f"DESPERATION: Entering desperation mode. "
                        f"Stuck for {self.desperation_counter} steps in area around ({curr_x:.2f}, {curr_y:.2f})."
                    )
                self.desperation_mode = True
        else:
            # Not stuck, so reset all desperation tracking
            if self.desperation_mode:
                logger.info("DESPERATION: Exiting desperation mode.")
            self.desperation_mode = False
            self.desperation_counter = 0
            self.last_position_for_desperation = None
            
        return is_stuck

    def _update_obstacle_map(self, x: float, y: float, vx: float, vy: float, is_obstacle: bool):
        """
        Update the pseudo obstacle map based on collision detection.
        
        Args:
            x, y: Current position
            vx, vy: Intended velocity (for obstacle detection)
            is_obstacle: True if obstacle detected, False if free space
        """
        # Convert world coordinates to grid coordinates
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        
        # Ensure coordinates are within map bounds
        grid_x = np.clip(grid_x, 0, self.map_width - 1)
        grid_y = np.clip(grid_y, 0, self.map_height - 1)
        
        if is_obstacle:
            # If obstacle detected, also check the intended movement direction
            if abs(vx) > 0.1 or abs(vy) > 0.1:
                # Calculate where robot was trying to go
                intended_x = x + vx * 0.1  # Look ahead by 0.1 meters
                intended_y = y + vy * 0.1
                
                intended_grid_x = int(intended_x / self.map_resolution)
                intended_grid_y = int(intended_y / self.map_resolution)
                
                intended_grid_x = np.clip(intended_grid_x, 0, self.map_width - 1)
                intended_grid_y = np.clip(intended_grid_y, 0, self.map_height - 1)
                
                # Mark intended location as obstacle
                self.obstacle_map[intended_grid_y, intended_grid_x] = 1
                self.obstacle_confidence[intended_grid_y, intended_grid_x] += 0.3
                
                logger.debug(f"MAPPING: Obstacle at grid ({intended_grid_x}, {intended_grid_y}) "
                           f"world ({intended_x:.2f}, {intended_y:.2f})")
            
            # Also mark current position as potentially blocked
            self.obstacle_map[grid_y, grid_x] = 0.5  # Partial obstacle
            self.obstacle_confidence[grid_y, grid_x] += 0.1
        else:
            # Mark as free space if we successfully moved here
            if self.obstacle_map[grid_y, grid_x] != 1:  # Don't override confirmed obstacles
                self.obstacle_map[grid_y, grid_x] = -1
                self.obstacle_confidence[grid_y, grid_x] = max(0, self.obstacle_confidence[grid_y, grid_x] - 0.05)
        
        # Clamp confidence values
        self.obstacle_confidence = np.clip(self.obstacle_confidence, 0, 1.0)

    def _mark_field_boundaries(self):
        """Mark the field boundaries as obstacles in the map."""
        field_width = 16.54
        field_length = 8.21
        
        # Mark left and right boundaries
        for y in range(self.map_height):
            self.obstacle_map[y, 0] = 1  # Left boundary
            self.obstacle_map[y, self.map_width - 1] = 1  # Right boundary
            self.obstacle_confidence[y, 0] = 1.0
            self.obstacle_confidence[y, self.map_width - 1] = 1.0
        
        # Mark top and bottom boundaries
        for x in range(self.map_width):
            self.obstacle_map[0, x] = 1  # Bottom boundary
            self.obstacle_map[self.map_height - 1, x] = 1  # Top boundary
            self.obstacle_confidence[0, x] = 1.0
            self.obstacle_confidence[self.map_height - 1, x] = 1.0

    def _get_obstacle_avoidance_force(self, x: float, y: float) -> Tuple[float, float]:
        """
        Calculate avoidance force based on the pseudo obstacle map.
        
        Returns:
            (avoid_vx, avoid_vy): Avoidance velocity components
        """
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        
        # Ensure coordinates are within map bounds
        grid_x = np.clip(grid_x, 0, self.map_width - 1)
        grid_y = np.clip(grid_y, 0, self.map_height - 1)
        
        avoid_vx, avoid_vy = 0.0, 0.0
        
        # Check surrounding area for obstacles
        search_radius = 3  # Grid cells to check around current position
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                check_x = grid_x + dx
                check_y = grid_y + dy
                
                # Skip out of bounds
                if (check_x < 0 or check_x >= self.map_width or 
                    check_y < 0 or check_y >= self.map_height):
                    continue
                
                # Skip center cell
                if dx == 0 and dy == 0:
                    continue
                
                # If there's an obstacle, calculate repulsive force
                obstacle_strength = self.obstacle_map[check_y, check_x]
                confidence = self.obstacle_confidence[check_y, check_x]
                
                if obstacle_strength > 0 and confidence > 0.2:
                    # Distance to obstacle
                    dist = np.hypot(dx, dy)
                    if dist < 0.1:
                        dist = 0.1  # Prevent division by zero
                    
                    # Repulsive force (inversely proportional to distance)
                    force_magnitude = (obstacle_strength * confidence) / (dist * dist)
                    
                    # Direction away from obstacle
                    force_x = -dx / dist * force_magnitude
                    force_y = -dy / dist * force_magnitude
                    
                    avoid_vx += force_x
                    avoid_vy += force_y
        
        # Normalize and scale avoidance force
        avoid_magnitude = np.hypot(avoid_vx, avoid_vy)
        if avoid_magnitude > 3.0:
            avoid_vx = (avoid_vx / avoid_magnitude) * 3.0
            avoid_vy = (avoid_vy / avoid_magnitude) * 3.0
        
        return avoid_vx, avoid_vy

    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates."""
        grid_x = int(x / self.map_resolution)
        grid_y = int(y / self.map_resolution)
        grid_x = np.clip(grid_x, 0, self.map_width - 1)
        grid_y = np.clip(grid_y, 0, self.map_height - 1)
        return grid_x, grid_y

    def _grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates."""
        x = (grid_x + 0.5) * self.map_resolution
        y = (grid_y + 0.5) * self.map_resolution
        return x, y

    def _is_cell_passable(self, grid_x: int, grid_y: int) -> bool:
        """Check if a grid cell is passable (not an obstacle)."""
        if (grid_x < 0 or grid_x >= self.map_width or 
            grid_y < 0 or grid_y >= self.map_height):
            return False
        
        obstacle_strength = self.obstacle_map[grid_y, grid_x]
        confidence = self.obstacle_confidence[grid_y, grid_x]
        
        # Cell is blocked if it's a confident obstacle
        return not (obstacle_strength > 0.3 and confidence > 0.3)

    def _get_neighbors(self, grid_x: int, grid_y: int) -> List[Tuple[int, int, float]]:
        """Get passable neighbor cells with movement costs."""
        neighbors = []
        
        # 8-directional movement (including diagonals)
        directions = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            (0, -1, 1.0),                   (0, 1, 1.0),
            (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)
        ]
        
        for dx, dy, cost in directions:
            new_x, new_y = grid_x + dx, grid_y + dy
            
            if self._is_cell_passable(new_x, new_y):
                # Add penalty for cells near obstacles to encourage safer paths
                penalty = 0.0
                for check_dx in range(-1, 2):
                    for check_dy in range(-1, 2):
                        check_x = new_x + check_dx
                        check_y = new_y + check_dy
                        if (0 <= check_x < self.map_width and 0 <= check_y < self.map_height):
                            obs_strength = self.obstacle_map[check_y, check_x]
                            obs_confidence = self.obstacle_confidence[check_y, check_x]
                            if obs_strength > 0.3 and obs_confidence > 0.3:
                                penalty += 0.5
                
                total_cost = cost + penalty
                neighbors.append((new_x, new_y, total_cost))
        
        return neighbors

    def _dijkstra_pathfind(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> List[Tuple[float, float]]:
        """
        Find a path from start to goal using Dijkstra's algorithm.
        
        Returns:
            List of (x, y) waypoints in world coordinates, or empty list if no path found.
        """
        # Convert to grid coordinates
        start_grid_x, start_grid_y = self._world_to_grid(start_x, start_y)
        goal_grid_x, goal_grid_y = self._world_to_grid(goal_x, goal_y)
        
        # Check cache first
        cache_key = (start_grid_x, start_grid_y, goal_grid_x, goal_grid_y)
        if cache_key in self.path_cache:
            return self.path_cache[cache_key]
        
        # Initialize Dijkstra's algorithm
        distances = {}
        previous = {}
        visited = set()
        pq = [(0.0, start_grid_x, start_grid_y)]
        
        distances[(start_grid_x, start_grid_y)] = 0.0
        
        while pq:
            current_dist, current_x, current_y = heapq.heappop(pq)
            
            # Skip if already visited
            if (current_x, current_y) in visited:
                continue
            
            visited.add((current_x, current_y))
            
            # Check if we reached the goal
            if current_x == goal_grid_x and current_y == goal_grid_y:
                break
            
            # Check all neighbors
            for neighbor_x, neighbor_y, move_cost in self._get_neighbors(current_x, current_y):
                if (neighbor_x, neighbor_y) in visited:
                    continue
                
                new_distance = current_dist + move_cost
                
                if ((neighbor_x, neighbor_y) not in distances or 
                    new_distance < distances[(neighbor_x, neighbor_y)]):
                    
                    distances[(neighbor_x, neighbor_y)] = new_distance
                    previous[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    heapq.heappush(pq, (new_distance, neighbor_x, neighbor_y))
        
        # Reconstruct path
        path = []
        if (goal_grid_x, goal_grid_y) in previous or (goal_grid_x, goal_grid_y) == (start_grid_x, start_grid_y):
            current = (goal_grid_x, goal_grid_y)
            
            while current is not None:
                grid_x, grid_y = current
                world_x, world_y = self._grid_to_world(grid_x, grid_y)
                path.append((world_x, world_y))
                current = previous.get(current)
            
            path.reverse()
        
        # Cache the result (manage cache size)
        if len(self.path_cache) >= self.path_cache_max_size:
            # Remove oldest entries
            keys_to_remove = list(self.path_cache.keys())[:10]
            for key in keys_to_remove:
                del self.path_cache[key]
        
        self.path_cache[cache_key] = path
        
        if path:
            logger.debug(f"PATHFIND: Found path with {len(path)} waypoints from ({start_x:.1f},{start_y:.1f}) to ({goal_x:.1f},{goal_y:.1f})")
        else:
            logger.debug(f"PATHFIND: No path found from ({start_x:.1f},{start_y:.1f}) to ({goal_x:.1f},{goal_y:.1f})")
        
        return path

    def _get_pathfinding_velocity(self, current_x: float, current_y: float, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        Get velocity commands using pathfinding for obstacle avoidance with stuck detection.
        
        Returns:
            (vx, vy): Velocity components to follow the path
        """
        # Emergency escape mode - use random movement to get unstuck
        if self.emergency_escape_mode:
            self.emergency_escape_counter += 1
            if self.emergency_escape_counter > 10:  # Exit emergency mode after 10 attempts
                self.emergency_escape_mode = False
                self.emergency_escape_counter = 0
                self.current_path = []  # Clear path to force recalculation
                logger.debug("PATHFIND: Exiting emergency escape mode")
            # Random movement to break out of stuck situation
            escape_angle = random.uniform(0, 2 * np.pi)
            escape_speed = 1.5  # Reduced from 2.0 for smoother movement
            escape_vx = np.cos(escape_angle) * escape_speed
            escape_vy = np.sin(escape_angle) * escape_speed
            logger.debug(f"PATHFIND: Emergency escape - random movement ({escape_vx:.2f},{escape_vy:.2f})")
            return escape_vx, escape_vy
        
        # Check if we need to recalculate the path
        need_new_path = (
            not self.current_path or 
            self.path_target is None or
            np.hypot(target_x - self.path_target[0], target_y - self.path_target[1]) > self.path_recalc_threshold
        )
        
        if need_new_path:
            self.current_path = self._dijkstra_pathfind(current_x, current_y, target_x, target_y)
            self.path_target = (target_x, target_y)
            self.waypoint_attempts = 0  # Reset waypoint attempts counter
        
        if not self.current_path:
            # No path found, try emergency escape
            logger.debug("PATHFIND: No path found, entering emergency escape mode")
            self.emergency_escape_mode = True
            self.emergency_escape_counter = 0
            return self._get_pathfinding_velocity(current_x, current_y, target_x, target_y)
        
        # Find the next waypoint to follow
        next_waypoint = None
        min_distance = float('inf')
        waypoint_index = 0
        
        for i, (wx, wy) in enumerate(self.current_path):
            dist = np.hypot(wx - current_x, wy - current_y)
            if dist < min_distance:
                min_distance = dist
                waypoint_index = i
        
        # Look ahead to next waypoint if current one is close
        if min_distance < self.map_resolution and waypoint_index + 1 < len(self.current_path):
            next_waypoint = self.current_path[waypoint_index + 1]
        else:
            next_waypoint = self.current_path[waypoint_index]
        
        # Check if we're stuck trying the same waypoint
        if self.last_waypoint is not None:
            waypoint_distance = np.hypot(next_waypoint[0] - self.last_waypoint[0], 
                                       next_waypoint[1] - self.last_waypoint[1])
            if waypoint_distance < 0.5:  # Same waypoint (within 0.5m)
                self.waypoint_attempts += 1
            else:
                self.waypoint_attempts = 0  # Reset counter for new waypoint
        
        self.last_waypoint = next_waypoint
        
        # If we've been stuck on the same waypoint too long, enter emergency mode
        if self.waypoint_attempts > self.stuck_waypoint_threshold:
            logger.debug(f"PATHFIND: Stuck on waypoint ({next_waypoint[0]:.1f},{next_waypoint[1]:.1f}) for {self.waypoint_attempts} attempts, entering emergency escape")
            self.emergency_escape_mode = True
            self.emergency_escape_counter = 0
            self.waypoint_attempts = 0
            self.current_path = []  # Clear the problematic path
            return self._get_pathfinding_velocity(current_x, current_y, target_x, target_y)
        
        # Calculate velocity toward next waypoint
        dx = next_waypoint[0] - current_x
        dy = next_waypoint[1] - current_y
        
        # Scale velocity based on distance to waypoint
        waypoint_distance = np.hypot(dx, dy)
        if waypoint_distance < 0.1:
            # Very close to waypoint, move to next one
            if waypoint_index + 1 < len(self.current_path):
                next_waypoint = self.current_path[waypoint_index + 1]
                dx = next_waypoint[0] - current_x
                dy = next_waypoint[1] - current_y
        
        # Scale velocity - reduced for smoother movement
        velocity_scale = 1.5  # Reduced from 3.0
        vx = dx * velocity_scale
        vy = dy * velocity_scale
        
        # Clamp velocities to reasonable limits
        vx = np.clip(vx, -2.0, 2.0)
        vy = np.clip(vy, -2.0, 2.0)
        
        logger.debug(f"PATHFIND: Following waypoint ({next_waypoint[0]:.1f},{next_waypoint[1]:.1f}), "
                    f"velocity ({vx:.2f},{vy:.2f}), attempts={self.waypoint_attempts}")
        
        return vx, vy

    def _detect_wall_proximity_and_avoid(self, x: float, y: float, vx: float, vy: float) -> Tuple[bool, float, float]:
        """
        Fuzzy logic obstacle avoidance: blend desired velocity with repulsive force from nearby obstacles.
        
        Returns:
            (near_wall, avoid_vx, avoid_vy): Boolean and avoidance velocity components
        """
        speed = np.hypot(vx, vy)
        # If hardly moving, no avoidance
        if speed < 0.01:
            return False, vx, vy
        # Unit direction of desired motion
        ux, uy = vx / speed, vy / speed
        # Lookahead distance and sampling
        lookahead = 0.5  # meters ahead to scan
        samples = 5
        total_strength = 0.0
        avoid_x, avoid_y = 0.0, 0.0
        for i in range(1, samples + 1):
            d = lookahead * i / samples
            sx = x + ux * d
            sy = y + uy * d
            gx, gy = self._world_to_grid(sx, sy)
            conf = self.obstacle_confidence[gy, gx]
            # weight decreases with distance
            w = conf * (1 - d / lookahead)
            if w > 0.0:
                # perpendicular direction
                px, py = -uy, ux
                avoid_x += px * w
                avoid_y += py * w
                total_strength += w
        # Determine if avoidance needed
        near_wall = total_strength > 0.2
        if near_wall:
            # fuzzy blend between forward and avoidance
            blend = min(1.0, total_strength)
            fx = ux * (1 - blend) + avoid_x * blend
            fy = uy * (1 - blend) + avoid_y * blend
            # scale to original speed
            fn = np.hypot(fx, fy)
            if fn > 1e-3:
                fx, fy = fx / fn * speed, fy / fn * speed
            else:
                # fallback orthogonal
                fx, fy = -uy * speed, ux * speed
            return True, fx, fy
        # no obstacles nearby
        return False, vx, vy

    def get_obstacle_map_summary(self) -> str:
        """
        Get a summary of the current obstacle map for debugging.
        """
        total_cells = self.map_width * self.map_height
        obstacle_cells = np.sum(self.obstacle_map > 0.5)
        free_cells = np.sum(self.obstacle_map < -0.5)
        unknown_cells = total_cells - obstacle_cells - free_cells
        
        max_confidence = np.max(self.obstacle_confidence)
        avg_confidence = np.mean(self.obstacle_confidence[self.obstacle_confidence > 0])
        
        return (f"Map: {self.map_width}x{self.map_height} cells, "
                f"Obstacles: {obstacle_cells}, Free: {free_cells}, Unknown: {unknown_cells}, "
                f"Max confidence: {max_confidence:.2f}, Avg confidence: {avg_confidence:.2f}")

    def save_obstacle_map(self, filename: str = "obstacle_map.txt"):
        """
        Save the current obstacle map to a file for analysis.
        """
        try:
            with open(filename, 'w') as f:
                f.write(f"# Obstacle Map - {self.map_width}x{self.map_height}\n")
                f.write(f"# Resolution: {self.map_resolution}m per cell\n")
                f.write(f"# Values: 1=obstacle, 0=unknown, -1=free\n\n")
                
                for y in range(self.map_height):
                    for x in range(self.map_width):
                        obstacle_val = self.obstacle_map[y, x]
                        confidence_val = self.obstacle_confidence[y, x]
                        
                        if obstacle_val > 0.5:
                            f.write(f"X")  # Obstacle
                        elif obstacle_val < -0.5:
                            f.write(f".")  # Free space
                        else:
                            f.write(f" ")  # Unknown
                    f.write(f"\n")
                
                logger.info(f"Obstacle map saved to {filename}")
                
        except Exception as e:
            logger.error(f"Failed to save obstacle map: {e}")

    def reset_obstacle_map(self):
        """Reset the obstacle map (useful for new episodes)."""
        self.obstacle_map = np.zeros((self.map_height, self.map_width))
        self.obstacle_confidence = np.zeros((self.map_height, self.map_width))
        self.position_history = []
        self.velocity_history = []
        
        # Clear pathfinding cache and current path
        self.path_cache = {}
        self.current_path = []
        self.path_target = None
        
        # Reset stuck detection variables
        self.waypoint_attempts = 0
        self.last_waypoint = None
        self.emergency_escape_mode = False
        self.emergency_escape_counter = 0
        
        # Reset desperation mode
        self.desperation_counter = 0
        self.desperation_mode = False
        self.last_position_for_desperation = None
        
        # Reset velocity smoothing
        self.last_velocity_command = [0.0, 0.0, 0.0]
        self.command_hold_counter = 0
        
        logger.info("Obstacle map and pathfinding cache reset")

    def send_action(self, action: List[float]):
        """
        Send action to robot via NetworkTables - same method as basic_agent.
        """
        if not self.connected or not self.control_table:
            logger.warning("Not connected to robot - cannot send action")
            return
        
        try:
            # Ensure action has correct size
            while len(action) < 6:
                action.append(0.0)
            
            # Clamp actions to safe limits
            clamped_action = [
                np.clip(action[0], -5.0, 5.0),    # velocity_x
                np.clip(action[1], -5.0, 5.0),    # velocity_y  
                np.clip(action[2], -6.0, 6.0),    # angular_velocity
                np.clip(action[3], 0.0, 2.5),     # elevator_height
                np.clip(action[4], -90.0, 120.0), # arm_angle
                np.clip(action[5], -1.0, 1.0)     # intake_command
            ]
            
            # Send commands using same keys as basic_agent
            self.control_table.putNumber("velocity_x", clamped_action[0])
            self.control_table.putNumber("velocity_y", clamped_action[1]) 
            self.control_table.putNumber("angular_velocity", clamped_action[2])
            self.control_table.putNumber("elevator_height", clamped_action[3])
            self.control_table.putNumber("arm_angle", clamped_action[4])
            self.control_table.putNumber("intake_command", clamped_action[5])
            
            # Debug: Log when intake command is non-zero
            if abs(clamped_action[5]) > 0.1:
                logger.info(f"INTAKE COMMAND: {clamped_action[5]:.2f} ({'EJECT' if clamped_action[5] < 0 else 'INTAKE'})")
            
        except Exception as e:
            logger.error(f"Failed to send action: {e}")

    def get_state(self) -> np.ndarray:
        """
        Get current state from NetworkTables - same method as basic_agent.
        """
        if not self.connected or not self.state_table:
            logger.warning("Not connected or no state table available")
            return np.zeros(36)
        
        try:
            # Basic robot state (11 values) - same as basic_agent
            pose_x = self.state_table.getNumber("pose_x", 0.0)
            pose_y = self.state_table.getNumber("pose_y", 0.0)
            vel_x = self.state_table.getNumber("velocity_x", 0.0)
            vel_y = self.state_table.getNumber("velocity_y", 0.0)
            
            basic_state = [
                pose_x,
                pose_y,
                self.state_table.getNumber("pose_rotation", 0.0),
                vel_x,
                vel_y,
                self.state_table.getNumber("angular_velocity", 0.0),
                self.state_table.getNumber("elevator_height", 0.0),
                self.state_table.getNumber("arm_angle", 0.0),
                1.0 if self.state_table.getBoolean("at_setpoint", False) else 0.0,
                1.0 if self.state_table.getBoolean("has_game_piece", False) else 0.0,
                self.state_table.getNumber("timestamp", 0.0)
            ]
            
            # Enhanced game state information (16 additional values)
            closest_algae_x = self.state_table.getNumber("closest_algae_x", 0.0)
            closest_algae_y = self.state_table.getNumber("closest_algae_y", 0.0)
            closest_coral_x = self.state_table.getNumber("closest_coral_x", 0.0)
            closest_coral_y = self.state_table.getNumber("closest_coral_y", 0.0)
            pose_rotation = self.state_table.getNumber("pose_rotation", 0.0)
            
            game_state = [
                # Game piece counts and distances
                self.state_table.getNumber("algae_count", 0.0),
                self.state_table.getNumber("coral_count", 0.0),
                self.state_table.getNumber("closest_algae_distance", 999.0),
                closest_algae_x,
                closest_algae_y,
                self.state_table.getNumber("closest_coral_distance", 999.0),
                closest_coral_x,
                closest_coral_y,
                
                # Scoring information
                self.state_table.getNumber("red_score", 0.0),
                self.state_table.getNumber("blue_score", 0.0),
                self.state_table.getNumber("score_difference", 0.0),
                
                # Field positioning
                self.state_table.getNumber("distance_to_red_reef", 999.0),
                self.state_table.getNumber("distance_to_blue_reef", 999.0),
                self.state_table.getNumber("distance_to_nearest_reef", 999.0)
            ]
            
            # Orientation and intake direction awareness
            closest_piece_x, closest_piece_y = closest_algae_x, closest_algae_y
            closest_algae_dist = self.state_table.getNumber("closest_algae_distance", 999.0)
            closest_coral_dist = self.state_table.getNumber("closest_coral_distance", 999.0)
            
            if closest_coral_dist < closest_algae_dist:
                closest_piece_x, closest_piece_y = closest_coral_x, closest_coral_y
            
            # Calculate intake alignment
            if closest_piece_x != 0 or closest_piece_y != 0:
                dx, dy = closest_piece_x - pose_x, closest_piece_y - pose_y
                angle_to_piece = np.arctan2(dy, dx)
                heading_diff = normalize_angle(angle_to_piece - pose_rotation)
                intake_alignment = np.cos(heading_diff)  # -1 to 1, 1 is perfect alignment
            else:
                intake_alignment = 0.0
            
            orientation_state = [
                intake_alignment,  # How well aligned for intake (-1 to 1)
                np.sin(pose_rotation),  # Robot heading components for neural network
            ]
            
            # Combine basic and enhanced state and additional stuck detection (11 + 14 + 2 + 9 = 36)
            # Additional stuck detection and wall distance info (9 values)
            is_stuck = 1.0 if self.state_table.getBoolean("is_stuck", False) else 0.0
            average_velocity = self.state_table.getNumber("average_velocity", 0.0)
            total_recent_movement = self.state_table.getNumber("total_recent_movement", 0.0)
            time_since_movement = self.state_table.getNumber("time_since_movement", 0.0)
            distance_to_nearest_wall = self.state_table.getNumber("distance_to_nearest_wall", 0.0)
            distance_to_left_wall = self.state_table.getNumber("distance_to_left_wall", 0.0)
            distance_to_right_wall = self.state_table.getNumber("distance_to_right_wall", 0.0)
            distance_to_bottom_wall = self.state_table.getNumber("distance_to_bottom_wall", 0.0)
            distance_to_top_wall = self.state_table.getNumber("distance_to_top_wall", 0.0)
            additional_state = [
                is_stuck,
                average_velocity,
                total_recent_movement,
                time_since_movement,
                distance_to_nearest_wall,
                distance_to_left_wall,
                distance_to_right_wall,
                distance_to_bottom_wall,
                distance_to_top_wall
            ]
            state = np.array(basic_state + game_state + orientation_state + additional_state)
            return state
            
        except Exception as e:
            logger.error(f"Error reading state: {e}")
            return np.zeros(36)

    def connect(self) -> bool:
        """
        Connect to robot via NetworkTables.
        """
        try:
            # Use same connection method as basic_agent
            self.inst.startClient4("RL_Agent")
            self.inst.setServer("localhost")  # For simulation
            
            # Wait for connection
            start_time = time.time()
            while not self.inst.isConnected() and (time.time() - start_time) < 5.0:
                time.sleep(0.1)
            
            if self.inst.isConnected():
                # Get NetworkTables - same as basic_agent
                self.control_table = self.inst.getTable("RL_Control")
                self.state_table = self.inst.getTable("RL_State")
                
                # Enable robot control
                self.control_table.putBoolean("enabled", True)
                
                self.connected = True
                logger.info("Successfully connected to robot NetworkTables")
                
                # Debug: Check if tables are working
                logger.info(f"Control table keys: {list(self.control_table.getKeys())}")
                logger.info(f"State table keys: {list(self.state_table.getKeys())}")
                
                return True
            else:
                logger.error("Failed to connect to robot NetworkTables")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False

    def disconnect(self):
        """
        Disconnect from NetworkTables.
        """
        try:
            if self.connected and self.control_table:
                # Disable robot control and send zero commands
                self.send_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                self.control_table.putBoolean("enabled", False)
                
            self.inst.stopClient()
            self.connected = False
            logger.info("Disconnected from robot")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")

    def reset_environment(self) -> np.ndarray:
        """
        Reset the environment and return initial state.
        """
        if not self.connected or not self.control_table:
            return np.zeros(36)
        
        try:
            # Request environment reset
            self.control_table.putBoolean("reset_environment", True)
            time.sleep(0.1)  # Give time for reset
            
            # Reset collision detection history
            self.position_history = []
            self.velocity_history = []
            
            # Reset stuck detection variables
            self.waypoint_attempts = 0
            self.last_waypoint = None
            self.emergency_escape_mode = False
            self.emergency_escape_counter = 0
            
            # Reset desperation mode
            self.desperation_counter = 0
            self.desperation_mode = False
            self.last_position_for_desperation = None
            
            # Reset velocity smoothing
            self.last_velocity_command = [0.0, 0.0, 0.0]
            self.command_hold_counter = 0
            
            # Clear pathfinding cache and current path
            self.current_path = []
            self.path_target = None
            
            # Optionally reset obstacle map (comment out to preserve learning across episodes)
            # self.reset_obstacle_map()
            
            # Get initial state
            return self.get_state()
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return np.zeros(36)

    def random_policy(self) -> List[float]:
        """
        Generate a random action for exploration.
        """
        import random
        action = []
        
        # Random drivetrain velocities
        action.append(random.uniform(-3.0, 3.0))  # vx
        action.append(random.uniform(-3.0, 3.0))  # vy
        action.append(random.uniform(-2.0, 2.0))  # angular velocity
        
        # Random mechanism commands
        action.append(random.uniform(0.0, 2.0))   # elevator
        action.append(random.uniform(-1.0, 1.0))  # intake
        action.append(random.uniform(-1.0, 1.0))  # shooter
        
        return action

    def run_episode(self, max_steps: int = 1000, timestep: float = 0.02):
        """
        Run a single episode with advanced reward calculation.
        """
        try:
            state = self.reset_environment()
            if state is None:
                logger.error("Failed to get initial state")
                return
        except Exception as e:
            logger.error(f"Failed to reset environment: {e}")
            return
        
        self.episode_start_time = time.time()
        episode_reward = 0.0
        corals_collected = 0
        scores = 0
        
        for step in range(max_steps):
            prev_state = state
            
            # Select and execute action
            action = self.select_action(state)
            self.send_action(action)
            time.sleep(timestep)
            state = self.get_state()
            
            # Calculate advanced reward
            step_reward = self.calculate_advanced_reward(prev_state, action, state)
            episode_reward += step_reward
            
            # Track metrics
            if self._safe_state_access(state, 'has_piece') > 0.5 and not (self._safe_state_access(prev_state, 'has_piece') > 0.5):
                corals_collected += 1
                logger.info(f"Coral collected! Total: {corals_collected}")
            
            if self._safe_state_access(state, 'score_diff') > self._safe_state_access(prev_state, 'score_diff'):
                scores += 1
                logger.info(f"Scored! Total: {scores}")
                # End episode after successful score
                break
        
        # Stop robot
        self.send_action([0.0] * 6)
        
        episode_duration = time.time() - self.episode_start_time
        
        # Get obstacle map summary
        map_summary = self.get_obstacle_map_summary()
        
        logger.info(f"Episode complete: {episode_duration:.1f}s, "
                   f"Reward: {episode_reward:.2f}, Corals: {corals_collected}, Scores: {scores}")
        logger.info(f"Obstacle mapping: {map_summary}")
        
        # Save obstacle map periodically
        if hasattr(self, '_episode_count'):
            self._episode_count += 1
        else:
            self._episode_count = 1
            
        if self._episode_count % 5 == 0:  # Save every 5 episodes
            self.save_obstacle_map(f"obstacle_map_ep_{self._episode_count}.txt")
        
        return {
            'duration': episode_duration,
            'reward': episode_reward,
            'corals_collected': corals_collected,
            'scores': scores
        }

    def run_training_session(self, num_episodes: int = 50):
        """
        Run multiple episodes for training/evaluation.
        """
        if not self.connect():
            logger.error("Could not connect to robot")
            return
        
        results = []
        
        for ep in range(1, num_episodes + 1):
            logger.info(f"\n=== Episode {ep}/{num_episodes} ===")
            result = self.run_episode()
            if result:
                results.append(result)
            
            time.sleep(1.0)  # Brief pause between episodes
        
        # Summary statistics
        if results:
            avg_reward = np.mean([r['reward'] for r in results])
            total_scores = sum(r['scores'] for r in results)
            success_rate = sum(1 for r in results if r['scores'] > 0) / len(results)
            
            logger.info(f"\n=== TRAINING SESSION COMPLETE ===")
            logger.info(f"Episodes: {len(results)}")
            logger.info(f"Average reward: {avg_reward:.2f}")
            logger.info(f"Total scores: {total_scores}")
            logger.info(f"Success rate: {success_rate:.2%}")
        
        self.disconnect()

    # ==== DQN Agent Skeleton ====
    class DQNNetwork(nn.Module):
        def __init__(self, state_dim: int, action_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        def forward(self, x):
            return self.net(x)

    class DQNAgent:
        """
        A simple DQN agent for discrete actions.
        """
        def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3,
                     gamma: float = 0.99, epsilon_start: float = 1.0,
                     epsilon_end: float = 0.01, epsilon_decay: int = 500):
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.gamma = gamma
            self.epsilon = epsilon_start
            self.epsilon_min = epsilon_end
            self.epsilon_decay = epsilon_decay
            self.policy_net = HeuristicAgent.DQNNetwork(state_dim, action_dim)
            self.target_net = HeuristicAgent.DQNNetwork(state_dim, action_dim)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.memory = deque(maxlen=10000)
            self.steps_done = 0

        def select_action(self, state: np.ndarray) -> int:
            # Epsilon-greedy action selection
            sample = random.random()
            self.epsilon = max(self.epsilon_min, self.epsilon - (1.0 - self.epsilon_min) / self.epsilon_decay)
            if sample < self.epsilon:
                return random.randrange(self.action_dim)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                    q_values = self.policy_net(state_tensor)
                    return int(q_values.argmax().item())

        def store_transition(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))

        def optimize_model(self, batch_size: int = 64):
            if len(self.memory) < batch_size:
                return
            transitions = random.sample(self.memory, batch_size)
            batch = list(zip(*transitions))
            states = torch.tensor(np.array(batch[0]), dtype=torch.float32)
            actions = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1)
            rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1)
            next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32)
            dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1)

            # Compute Q(s_t, a)
            state_action_values = self.policy_net(states).gather(1, actions)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)
            expected_state_action_values = rewards + (self.gamma * next_state_values * (1 - dones))

            # Compute loss
            criterion = nn.MSELoss()
            loss = criterion(state_action_values, expected_state_action_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def update_target_network(self):
            self.target_net.load_state_dict(self.policy_net.state_dict())
    # ==== End DQN Agent Skeleton ====
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    agent = HeuristicAgent()
    agent.run_training_session(num_episodes=20)
