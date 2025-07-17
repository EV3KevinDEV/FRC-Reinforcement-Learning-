#!/usr/bin/env python3
"""
Basic Reinforcement Learning Agent for FRC Robot Control

This agent connects to the robot via NetworkTables and provides a simple
interface for controlling the robot and receiving state feedback.
"""

import time
import numpy as np
import ntcore
from typing import Dict, List, Tuple
import logging
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class BasicFRCAgent:
    """
    A basic RL agent that can control an FRC robot via NetworkTables.
    
    This agent provides:
    - Connection to robot NetworkTables
    - State observation from robot sensors
    - Action sending to robot actuators
    - Basic random policy for testing
    """
    
    def __init__(self, team_number: int = 0000, server_ip: str = "localhost"):
        """
        Initialize the agent.
        
        Args:
            team_number: FRC team number (e.g., 1234)
            server_ip: IP address of robot (None for automatic detection)
        """
        self.team_number = team_number
        self.server_ip = server_ip
        
        # NetworkTables setup
        self.inst = ntcore.NetworkTableInstance.getDefault()
        self.control_table = None
        self.state_table = None
        
        # State and action spaces
        self.state_size = 25  # Enhanced state with game piece and scoring info
        self.action_size = 6  # vel_x, vel_y, ang_vel, elev_height, arm_angle, intake_cmd
        
        # Action limits (matching robot safety limits)
        self.action_limits = {
            'velocity_x': (-5.0, 5.0),      # m/s
            'velocity_y': (-5.0, 5.0),      # m/s
            'angular_velocity': (-6.0, 6.0), # rad/s
            'elevator_height': (0.0, 2.5),   # m
            'arm_angle': (-90.0, 120.0),     # degrees
            'intake_command': (-1.0, 1.0)    # -1=eject, 0=stop, 1=intake
        }
        
        # Current state and action
        self.current_state = np.zeros(self.state_size)
        self.current_action = np.zeros(self.action_size)
        
        # === IMPROVED STUCK DETECTION AND PUNISHMENT TRACKING ===
        self.position_history = []  # Track recent positions for stuck detection
        self.stuck_threshold = 0.001  # Very lenient - robot moves at 0.0005m avg
        self.stuck_check_window = 20  # Longer window for better averaging
        self.consecutive_stuck_steps = 0  # Count of consecutive stuck steps
        self.max_stuck_penalty = -5.0  # Reduced max penalty
        self.stuck_reset_threshold = 0.005  # Very low movement needed to reset
        
        # Wall collision memory
        self.recent_wall_collisions = []  # Track recent wall collisions
        self.wall_collision_memory_time = 5.0  # Remember wall collisions for 5 seconds
        
        # Debug tracking
        self._stuck_debug_counter = 0
        
        # Oscillation detection - improved
        self.action_history = []  # Track recent actions for oscillation detection
        self.oscillation_threshold = 0.95  # Much higher to avoid false positives
        self.oscillation_window = 6  # Shorter window for faster detection
        self.oscillation_penalty = -0.2  # Much reduced penalty
        
        # Efficiency tracking - less aggressive
        self.total_energy_used = 0.0  # Track energy consumption
        self.steps_without_progress = 0  # Steps without meaningful progress
        self.max_steps_without_progress = 30  # Reduced threshold
        self.last_meaningful_position = None  # Track last position with good progress
        self.progress_reset_distance = 0.5  # Distance needed to reset progress counter
        
        # Connection status
        self.connected = False
        # DQN components
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        # Neural networks
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.update_target_steps = 1000
        self.learn_step_counter = 0
        
    def connect(self) -> bool:
        """
        Connect to the robot's NetworkTables.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self.server_ip:
                self.inst.startClient4("RL_Agent")
                self.inst.setServer(self.server_ip)
            else:
                self.inst.startClient4("RL_Agent")
                if self.team_number > 0:
                    self.inst.setServerTeam(self.team_number)
                else:
                    # Default to localhost for simulation
                    self.inst.setServer("localhost")
            
            # Wait for connection
            start_time = time.time()
            while not self.inst.isConnected() and (time.time() - start_time) < 5.0:
                time.sleep(0.1)
            
            if self.inst.isConnected():
                # Get NetworkTables
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
        """Disconnect from NetworkTables and disable robot control."""
        if self.connected and self.control_table:
            # Disable robot control and send zero commands
            self.send_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.control_table.putBoolean("enabled", False)
            
        self.inst.stopClient()
        self.connected = False
        logger.info("Disconnected from robot")
    
    def get_state(self) -> np.ndarray:
        """
        Get current robot state from NetworkTables.
        
        Returns:
            Enhanced state vector as numpy array with game piece and scoring information
        """
        if not self.connected or not self.state_table:
            logger.warning("Not connected or no state table available")
            return self.current_state
        
        try:
            # Basic robot state (11 values) - with debug logging
            pose_x = self.state_table.getNumber("pose_x", 0.0)
            pose_y = self.state_table.getNumber("pose_y", 0.0)
            vel_x = self.state_table.getNumber("velocity_x", 0.0)
            vel_y = self.state_table.getNumber("velocity_y", 0.0)
            
            # Debug: Log position occasionally to check if it's updating
            if hasattr(self, '_debug_step_count'):
                self._debug_step_count += 1
            else:
                self._debug_step_count = 0
            
            if self._debug_step_count % 50 == 0:  # Log every 50 steps
                logger.info(f"STATE DEBUG: pos=({pose_x:.3f}, {pose_y:.3f}), vel=({vel_x:.3f}, {vel_y:.3f})")
                logger.info(f"Available keys: {list(self.state_table.getKeys())}")
            
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
            
            # Enhanced game state information (14 additional values)
            game_state = [
                # Game piece counts and distances
                self.state_table.getNumber("algae_count", 0.0),
                self.state_table.getNumber("coral_count", 0.0),
                self.state_table.getNumber("closest_algae_distance", 999.0),
                self.state_table.getNumber("closest_algae_x", 0.0),
                self.state_table.getNumber("closest_algae_y", 0.0),
                self.state_table.getNumber("closest_coral_distance", 999.0),
                self.state_table.getNumber("closest_coral_x", 0.0),
                self.state_table.getNumber("closest_coral_y", 0.0),
                
                # Scoring information
                self.state_table.getNumber("red_score", 0.0),
                self.state_table.getNumber("blue_score", 0.0),
                self.state_table.getNumber("score_difference", 0.0),
                
                # Field positioning
                self.state_table.getNumber("distance_to_red_reef", 999.0),
                self.state_table.getNumber("distance_to_blue_reef", 999.0),
                self.state_table.getNumber("distance_to_nearest_reef", 999.0)
            ]
            
            # Combine basic and enhanced state
            state = np.array(basic_state + game_state)
            self.current_state = state
            return state
            
        except Exception as e:
            logger.error(f"Error reading state: {e}")
            return self.current_state
    
    def send_action(self, action: List[float]):
        """
        Send action to robot via NetworkTables.
        
        Args:
            action: List of 6 action values [vel_x, vel_y, ang_vel, elev_height, arm_angle, intake_cmd]
        """
        if not self.connected or not self.control_table:
            logger.warning("Not connected to robot - cannot send action")
            return
        
        # Clamp actions to safe limits
        clamped_action = self._clamp_action(action)
        
        try:
            # Send action to robot
            self.control_table.putNumber("velocity_x", clamped_action[0])
            self.control_table.putNumber("velocity_y", clamped_action[1])
            self.control_table.putNumber("angular_velocity", clamped_action[2])
            self.control_table.putNumber("elevator_height", clamped_action[3])
            self.control_table.putNumber("arm_angle", clamped_action[4])
            self.control_table.putNumber("intake_command", clamped_action[5])
            
            self.current_action = np.array(clamped_action)
            
            # Debug: Log what we're sending
            logger.debug(f"Sent action: vel_x={clamped_action[0]:.2f}, vel_y={clamped_action[1]:.2f}, "
                        f"ang_vel={clamped_action[2]:.2f}")
            
        except Exception as e:
            logger.error(f"Error sending action: {e}")
    
    def _clamp_action(self, action: List[float]) -> List[float]:
        """Clamp action values to safe limits."""
        limits = [
            self.action_limits['velocity_x'],
            self.action_limits['velocity_y'],
            self.action_limits['angular_velocity'],
            self.action_limits['elevator_height'],
            self.action_limits['arm_angle'],
            self.action_limits['intake_command']
        ]
        
        clamped = []
        for i, (val, (min_val, max_val)) in enumerate(zip(action, limits)):
            clamped.append(max(min_val, min(max_val, val)))
        
        return clamped
    
    def random_policy(self) -> List[float]:
        """
        Generate a random action for testing.
        
        Returns:
            Random action within safe limits
        """
        action = []
        for limit in [
            self.action_limits['velocity_x'],
            self.action_limits['velocity_y'],
            self.action_limits['angular_velocity'],
            self.action_limits['elevator_height'],
            self.action_limits['arm_angle'],
            self.action_limits['intake_command']
        ]:
            # Generate random value within limits (with bias toward zero for safety)
            if np.random.random() < 0.7:  # 70% chance of small movement
                action.append(np.random.uniform(limit[0] * 0.3, limit[1] * 0.3))
            else:  # 30% chance of larger movement
                action.append(np.random.uniform(limit[0], limit[1]))
        
        return action
    
    def select_action(self, state: np.ndarray) -> List[float]:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return self.random_policy()
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_t).cpu().numpy()[0]
            return q_values.tolist()
    
    def remember(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.append(Transition(state, action, reward, next_state, done))
    
    def optimize_model(self):
        """Sample batch and optimize DQN."""
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.FloatTensor(np.array(batch.state))
        action_batch = torch.FloatTensor(np.array(batch.action))
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state))
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1)

        # Compute Q(s,a)
        current_q = (self.policy_net(state_batch) * action_batch).sum(dim=1, keepdim=True)
        # Compute target
        next_q = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1)
        target_q = reward_batch + self.gamma * next_q * (1 - done_batch)

        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def run_test_episode(self, duration: float = 10.0, control_rate: float = 50.0):
        """
        Run a test episode with random actions and reward calculation.
        
        Args:
            duration: Episode duration in seconds
            control_rate: Control loop frequency in Hz
        """
        if not self.connected:
            logger.error("Not connected to robot - cannot run episode")
            return
        
        logger.info(f"Starting enhanced test episode for {duration} seconds at {control_rate} Hz")
        
        dt = 1.0 / control_rate
        start_time = time.time()
        step_count = 0
        total_reward = 0.0
        
        # Get initial state
        prev_state = self.get_state()
        
        try:
            while (time.time() - start_time) < duration:
                step_start = time.time()
                
                # Generate random action
                action = self.random_policy()
                
                # Send action to robot
                self.send_action(action)
                
                # Wait for action to take effect
                time.sleep(dt * 0.5)
                
                # Get new state
                current_state = self.get_state()
                
                # Calculate reward
                reward = self.calculate_reward(prev_state, np.array(action), current_state)
                total_reward += reward
                
                # Get game info for logging
                game_info = self.get_game_info()
                
                # Log progress with enhanced information
                if step_count % int(control_rate) == 0:  # Log every second
                    logger.info(f"Step {step_count}: Reward: {reward:.2f}, "
                              f"Total Reward: {total_reward:.2f}")
                    logger.info(f"  Game Info: Algae={game_info.get('algae_count', 0)}, "
                              f"Coral={game_info.get('coral_count', 0)}, "
                              f"Has Piece={game_info.get('robot_has_piece', False)}")
                    logger.info(f"  Position: ({current_state[0]:.2f}, {current_state[1]:.2f}), "
                              f"Closest Algae: {game_info.get('closest_algae_distance', 999):.2f}m")
                    logger.info(f"  Score - Red: {game_info.get('red_score', 0)}, "
                              f"Blue: {game_info.get('blue_score', 0)}")
                
                # Update previous state
                prev_state = current_state.copy()
                step_count += 1
                
                # Maintain control rate
                elapsed = time.time() - step_start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
        
        except KeyboardInterrupt:
            logger.info("Episode interrupted by user")
        
        finally:
            # Stop robot
            self.send_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            logger.info(f"Enhanced test episode completed!")
            logger.info(f"Total steps: {step_count}, Total reward: {total_reward:.2f}, "
                       f"Average reward: {total_reward/max(step_count, 1):.3f}")
            
            # Final game state summary
            final_game_info = self.get_game_info()
            logger.info(f"Final game state: {final_game_info}")
    
    def test_simple_movement(self):
        """Test simple movement to debug connection issues."""
        if not self.connected:
            logger.error("Not connected to robot")
            return
        
        logger.info("Testing simple movement...")
        
        # Test 1: Send a simple forward command
        logger.info("Test 1: Moving forward at 0.5 m/s")
        self.send_action([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(2.0)
        
        # Check if robot state changed
        state = self.get_state()
        logger.info(f"Current state: pos=({state[0]:.2f}, {state[1]:.2f}), "
                   f"vel=({state[3]:.2f}, {state[4]:.2f})")
        
        # Test 2: Stop
        logger.info("Test 2: Stopping")
        self.send_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(1.0)
        
        # Test 3: Check if NetworkTables are being updated
        logger.info("Test 3: Checking NetworkTable updates")
        test_val = 12.345
        self.control_table.putNumber("test_value", test_val)
        time.sleep(0.1)
        read_val = self.control_table.getNumber("test_value", 0.0)
        logger.info(f"Wrote {test_val}, read back {read_val} - {'SUCCESS' if abs(read_val - test_val) < 0.001 else 'FAILED'}")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        Calculate reward for RL training based on game state.
        Prioritizes: 1) Find and pick up game pieces, 2) Score them at reefs
        Includes punishment for stuck behavior, oscillation, and inefficiency.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Resulting state
            
        Returns:
            Reward value for training
        """
        reward = 0.0
        
        # === UPDATE BEHAVIOR TRACKING ===
        self._update_behavior_tracking(state, action, next_state)
        
        # === WALL COLLISION DETECTION ===
        wall_collision = self.detect_wall_collision(state, action, next_state)
        if wall_collision:
            # Significant penalty for wall collisions
            reward -= 3.0
            logger.debug("WALL COLLISION PENALTY: -3.0")
        
        # === PUNISHMENT MECHANISMS ===
        stuck_penalty = self._detect_stuck_behavior(state, next_state)
        # Temporarily disable oscillation detection to help learning
        oscillation_penalty = 0.0  # self._detect_oscillation()
        inefficiency_penalty = self._detect_inefficiency(state, action, next_state)
        
        # === POSITIVE REINFORCEMENT FOR BREAKING FREE ===
        # Reward the agent for successfully getting unstuck
        unstuck_bonus = 0.0
        if hasattr(self, '_previous_stuck_steps'):
            if self._previous_stuck_steps > 5 and self.consecutive_stuck_steps < self._previous_stuck_steps:
                unstuck_bonus = min(2.0, 0.2 * (self._previous_stuck_steps - self.consecutive_stuck_steps))
                logger.debug(f"UNSTUCK BONUS: {unstuck_bonus:.2f} for reducing stuck steps from {self._previous_stuck_steps} to {self.consecutive_stuck_steps}")
        self._previous_stuck_steps = self.consecutive_stuck_steps
        
        reward += stuck_penalty + oscillation_penalty + inefficiency_penalty + unstuck_bonus
        
        # === MOVEMENT ENCOURAGEMENT ===
        # Bonus for any movement at all to encourage exploration
        position_change = np.linalg.norm(next_state[:2] - state[:2])
        if position_change > 0.001:  # Any detectable movement
            movement_bonus = min(1.0, position_change * 10.0)  # Scale up small movements
            reward += movement_bonus
            logger.debug(f"MOVEMENT BONUS: {movement_bonus:.3f} for {position_change:.4f}m movement")
        
        # === BOUNDARY AWARENESS AND WALL APPROACH PREVENTION ===
        # Penalty for being near walls, larger penalty for being stuck at walls
        pose_x, pose_y = next_state[0], next_state[1]
        prev_x, prev_y = state[0], state[1]
        
        # Check if robot is at field boundaries
        field_margin = 0.5
        at_boundary = (pose_x < field_margin or pose_x > (16 - field_margin) or 
                      pose_y < field_margin or pose_y > (8 - field_margin))
        
        near_boundary = (pose_x < 1.5 or pose_x > 14.5 or 
                        pose_y < 1.5 or pose_y > 6.5)  # More conservative
        
        # Check for wall approach after recent collision
        approaching_wall_after_collision = False
        if hasattr(self, 'recent_wall_collisions') and self.recent_wall_collisions:
            # If we recently had a wall collision, check if we're moving back toward walls
            current_dist_to_wall = min(pose_x, 16-pose_x, pose_y, 8-pose_y)
            prev_dist_to_wall = min(prev_x, 16-prev_x, prev_y, 8-prev_y)
            
            if current_dist_to_wall < prev_dist_to_wall and current_dist_to_wall < 2.0:
                approaching_wall_after_collision = True
        
        if at_boundary:
            reward -= 2.0  # Penalty for being at boundary
            logger.debug(f"BOUNDARY PENALTY: At boundary pos=({pose_x:.2f}, {pose_y:.2f})")
            
            # Extra penalty if stuck at boundary
            if position_change < 0.01:
                reward -= 1.0
                logger.debug("BOUNDARY STUCK: Robot stuck at boundary")
                
            # Bonus for moving away from boundary
            distance_from_boundary_now = min(pose_x, 16-pose_x, pose_y, 8-pose_y)
            distance_from_boundary_prev = min(prev_x, 16-prev_x, prev_y, 8-prev_y)
            
            if distance_from_boundary_now > distance_from_boundary_prev:
                boundary_escape_bonus = 3.0 * (distance_from_boundary_now - distance_from_boundary_prev)
                reward += boundary_escape_bonus
                logger.debug(f"BOUNDARY ESCAPE BONUS: {boundary_escape_bonus:.2f}")
        
        elif near_boundary:
            reward -= 0.5  # Light penalty for being near boundary
            
            # Penalty for approaching wall after recent collision (prevent bouncing)
            if approaching_wall_after_collision:
                wall_approach_penalty = -2.0
                reward += wall_approach_penalty
                logger.debug(f"WALL APPROACH PENALTY: {wall_approach_penalty:.2f} for approaching wall after recent collision")
            
            # Bonus for moving toward center
            center_x, center_y = 8.0, 4.0
            dist_to_center_now = np.linalg.norm([pose_x - center_x, pose_y - center_y])
            dist_to_center_prev = np.linalg.norm([prev_x - center_x, prev_y - center_y])
            
            if dist_to_center_now < dist_to_center_prev:
                center_bonus = 0.5 * (dist_to_center_prev - dist_to_center_now)
                reward += center_bonus
                logger.debug(f"CENTER BONUS: {center_bonus:.2f}")
        
        # Extract relevant state information
        pose_x, pose_y = next_state[0], next_state[1]
        has_game_piece = next_state[9] > 0.5
        closest_algae_dist = next_state[13]
        closest_coral_dist = next_state[16]
        score_diff = next_state[21]
        distance_to_nearest_reef = next_state[24]
        
        # Previous state for comparison
        prev_closest_algae_dist = state[13] if len(state) > 13 else closest_algae_dist
        prev_closest_coral_dist = state[16] if len(state) > 16 else closest_coral_dist
        prev_score_diff = state[21] if len(state) > 21 else score_diff
        prev_has_game_piece = state[9] > 0.5 if len(state) > 9 else has_game_piece
        prev_distance_to_reef = state[24] if len(state) > 24 else distance_to_nearest_reef
        
        # === PHASE 1: PRIORITIZE GAME PIECE ACQUISITION ===
        
        if not has_game_piece:
            # HIGHEST PRIORITY: Getting closer to game pieces
            min_piece_distance = min(closest_algae_dist, closest_coral_dist)
            prev_min_piece_distance = min(prev_closest_algae_dist, prev_closest_coral_dist)
            
            # Strong reward for approaching game pieces
            if min_piece_distance < prev_min_piece_distance:
                distance_improvement = prev_min_piece_distance - min_piece_distance
                reward += 3.0 * distance_improvement  # Reduced from 5.0 to balance punishment
            
            # Extra reward for being very close to game pieces
            if min_piece_distance < 1.5:
                reward += 2.0  # Close proximity bonus
            if min_piece_distance < 0.8:
                reward += 3.0  # Very close proximity bonus
                
            # Moderate penalty for being far from game pieces (reduced)
            if min_piece_distance > 5.0:
                reward -= 1.0  # Reduced from 2.0
            elif min_piece_distance > 8.0:
                reward -= 2.0  # Reduced from 5.0
            
            # Encourage active searching - slight penalty for not moving toward game pieces
            if min_piece_distance >= prev_min_piece_distance and min_piece_distance > 2.0:
                reward -= 0.2  # Reduced from 0.5
            
            # Additional penalty for being near reef without game piece
            if distance_to_nearest_reef < 3.0:  # Close to reef
                reef_proximity_penalty = -1.5 * (3.0 - distance_to_nearest_reef)  # Stronger penalty the closer to reef
                reward += reef_proximity_penalty
                logger.debug(f"REEF PROXIMITY PENALTY: {reef_proximity_penalty:.2f} for being {distance_to_nearest_reef:.2f}m from reef without piece")
            
            # MASSIVE REWARD for successfully picking up a game piece
            if has_game_piece and not prev_has_game_piece:
                reward += 50.0  # Keep this high as it's the main objective
            
            # STRONG PENALTY for going toward reef without game piece (waste of time)
            if distance_to_nearest_reef < prev_distance_to_reef:
                distance_wasted = prev_distance_to_reef - distance_to_nearest_reef
                reef_penalty = -3.0 * distance_wasted  # Strong penalty proportional to distance moved toward reef
                reward += reef_penalty
                logger.debug(f"REEF WITHOUT PIECE PENALTY: {reef_penalty:.2f} for moving {distance_wasted:.2f}m toward reef without game piece")
        
        # === PHASE 2: SCORING WITH GAME PIECE ===
        
        else:  # has_game_piece is True
            # Now prioritize getting to reef for scoring
            
            # Strong reward for approaching reef with game piece
            if distance_to_nearest_reef < prev_distance_to_reef:
                distance_improvement = prev_distance_to_reef - distance_to_nearest_reef
                reward += 6.0 * distance_improvement  # Reduced from 8.0
            
            # Extra reward for being close to reef with game piece
            if distance_to_nearest_reef < 2.0:
                reward += 6.0  # Reduced from 8.0
            if distance_to_nearest_reef < 1.0:
                reward += 12.0  # Reduced from 15.0
            
            # Moderate penalty for moving away from reef when carrying piece
            if distance_to_nearest_reef > prev_distance_to_reef:
                reward -= 1.5  # Reduced from 3.0
            
            # Penalty for going toward game pieces when already carrying one
            if min(closest_algae_dist, closest_coral_dist) < min(prev_closest_algae_dist, prev_closest_coral_dist):
                reward -= 0.5  # Reduced from 1.0
            
            # MASSIVE REWARD for successful scoring
            if score_diff > prev_score_diff:
                reward += 100.0  # Keep this high
            elif score_diff < prev_score_diff:
                reward -= 25.0  # Reduced from 50.0
        
        # === GENERAL PENALTIES AND CONSTRAINTS ===
        
        # Keep robot on field (boundary penalty) - reduced
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            reward -= 5.0  # Reduced from 10.0
        
        # Moderate penalty for excessive movement (energy efficiency) - reduced
        action_magnitude = np.linalg.norm(action[:3])  # Only drivetrain actions
        if action_magnitude > 3.0:  # Higher threshold
            reward -= 0.1 * action_magnitude  # Much reduced penalty
        
        # === ADDITIONAL PUNISHMENT MECHANISMS (REDUCED) ===
        
        # Moderate penalty for remaining stationary when action commands movement
        commanded_movement = np.linalg.norm(action[:2])  # Commanded velocity
        actual_movement = np.linalg.norm(next_state[:2] - state[:2])
        if commanded_movement > 0.8 and actual_movement < 0.005:  # Higher threshold
            reward -= 2.0  # Reduced from 5.0
            logger.debug(f"MOVEMENT FAILURE: commanded={commanded_movement:.2f}, actual={actual_movement:.4f}")
        
        # Reduced penalty for erratic action changes (jittery behavior)
        if len(self.action_history) >= 2:
            prev_action = self.action_history[-2]
            action_change = np.linalg.norm(action - prev_action)
            if action_change > 4.0:  # Higher threshold
                reward -= 0.3  # Much reduced penalty
                logger.debug(f"ERRATIC BEHAVIOR: action_change={action_change:.2f}")
        
        # Reduced penalty for going out of bounds or near boundaries
        field_margin = 0.3  # Smaller margin
        if (pose_x < field_margin or pose_x > (16 - field_margin) or 
            pose_y < field_margin or pose_y > (8 - field_margin)):
            boundary_penalty = -1.0  # Much reduced
            reward += boundary_penalty
            logger.debug(f"BOUNDARY VIOLATION: pos=({pose_x:.2f}, {pose_y:.2f})")
        
        # Penalty for counterproductive actions - enhanced
        if not has_game_piece:
            # Penalize trying to score without a game piece
            if action[3] > 0.8 or action[4] > 20.0 or action[5] < -0.3:  # Higher thresholds
                scoring_attempt_penalty = -2.0  # Increased penalty
                reward += scoring_attempt_penalty
                logger.debug(f"FUTILE SCORING ATTEMPT: {scoring_attempt_penalty:.2f} - No game piece but trying to score")
            
            # Extra penalty for being near reef and trying to score without piece
            if distance_to_nearest_reef < 2.0 and (action[3] > 0.5 or action[4] > 10.0):
                futile_near_reef_penalty = -3.0
                reward += futile_near_reef_penalty
                logger.debug(f"FUTILE NEAR REEF: {futile_near_reef_penalty:.2f} - Near reef without piece and trying to score")
        
        # Penalty for excessive intake when already having game piece - reduced
        if has_game_piece and action[5] > 0.7:  # Higher threshold
            reward -= 0.3  # Much reduced penalty
            logger.debug("EXCESSIVE INTAKE: Already has game piece")
        
        # Smaller time penalty to encourage efficiency
        reward -= 0.001  # Much reduced from 0.02
        
        # === DEBUG LOGGING FOR SIGNIFICANT PENALTIES ===
        total_penalty = stuck_penalty + oscillation_penalty + inefficiency_penalty
        if total_penalty < -0.5:  # Very low threshold to catch only severe issues
            logger.info(f"SIGNIFICANT PUNISHMENT: stuck={stuck_penalty:.2f}, "
                       f"oscillation={oscillation_penalty:.2f}, inefficiency={inefficiency_penalty:.2f}")
        
        # === EXPLORATION BONUS WHEN SEVERELY STUCK ===
        # Give a stronger exploration bonus when the agent has been stuck
        exploration_bonus = 0.0
        if self.consecutive_stuck_steps > 10:  # Earlier intervention
            # Encourage the agent to try different actions when stuck
            action_novelty = np.linalg.norm(action - np.mean(self.action_history[-5:], axis=0)) if len(self.action_history) >= 5 else 0.0
            if action_novelty > 0.5:  # Lower threshold for novelty
                exploration_bonus = min(0.8, 0.15 * action_novelty)  # Stronger bonus
                logger.debug(f"EXPLORATION BONUS: {exploration_bonus:.2f} for novel action when stuck")
        
        # Extra bonus for Q-learning when trying completely new actions
        if hasattr(self, 'q_learning_mode') and self.q_learning_mode:
            if self.consecutive_stuck_steps > 5:
                exploration_bonus += 0.2  # Additional Q-learning exploration bonus
        
        reward += exploration_bonus
        
        return reward
    
    def get_game_info(self) -> Dict:
        """
        Get game-specific information for analysis.
        
        Returns:
            Dictionary with game state information
        """
        if not self.connected or not self.state_table:
            return {}
        
        try:
            return {
                'algae_count': self.state_table.getNumber("algae_count", 0),
                'coral_count': self.state_table.getNumber("coral_count", 0),
                'red_score': self.state_table.getNumber("red_score", 0),
                'blue_score': self.state_table.getNumber("blue_score", 0),
                'total_pieces_on_reef': self.state_table.getNumber("total_pieces_on_reef", 0),
                'robot_has_piece': self.state_table.getBoolean("has_game_piece", False),
                'closest_algae_distance': self.state_table.getNumber("closest_algae_distance", 999),
                'closest_coral_distance': self.state_table.getNumber("closest_coral_distance", 999),
                'distance_to_nearest_reef': self.state_table.getNumber("distance_to_nearest_reef", 999)
            }
        except Exception as e:
            logger.error(f"Error getting game info: {e}")
            return {}
    
    def _update_behavior_tracking(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        """
        Update tracking variables for stuck detection, oscillation, and efficiency monitoring.
        
        Args:
            state: Previous state
            action: Action taken
            next_state: Resulting state
        """
        # Track position for stuck detection
        current_pos = (next_state[0], next_state[1])
        self.position_history.append(current_pos)
        if len(self.position_history) > self.stuck_check_window:
            self.position_history.pop(0)
        
        # Track actions for oscillation detection
        self.action_history.append(action.copy())
        if len(self.action_history) > self.oscillation_window:
            self.action_history.pop(0)
        
        # Track energy usage
        energy_this_step = np.sum(np.abs(action[:3]))  # Movement energy
        self.total_energy_used += energy_this_step
    
    def _detect_stuck_behavior(self, state: np.ndarray, next_state: np.ndarray) -> float:
        """
        Much more forgiving stuck detection with extensive debugging.
        
        Returns:
            Penalty value (negative) for stuck behavior
        """
        penalty = 0.0
        
        if len(self.position_history) < 5:  # Need more history before detecting stuck
            return penalty
        
        # Calculate movement in recent positions
        positions = np.array(self.position_history)
        
        # Check immediate movement (most recent step)
        immediate_movement = np.linalg.norm(positions[-1] - positions[-2])
        
        # Check average movement over window
        if len(positions) >= self.stuck_check_window:
            recent_distances = []
            for i in range(1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[i-1])
                recent_distances.append(dist)
            
            avg_movement = np.mean(recent_distances[-self.stuck_check_window:])
            max_movement = max(recent_distances[-self.stuck_check_window:])
            
            # Much more lenient stuck detection
            if avg_movement < self.stuck_threshold and max_movement < self.stuck_threshold * 3:
                self.consecutive_stuck_steps += 1
                
                # Very gradual penalty that prioritizes learning over punishment
                if self.consecutive_stuck_steps <= 15:
                    stuck_penalty = -0.1  # Almost no penalty initially
                elif self.consecutive_stuck_steps <= 30:
                    stuck_penalty = -0.3  # Light penalty
                elif self.consecutive_stuck_steps <= 50:
                    stuck_penalty = -0.6  # Moderate penalty
                else:
                    stuck_penalty = -1.0  # Capped penalty
                
                penalty += stuck_penalty
                
                # Debug logging every 10 stuck steps
                self._stuck_debug_counter += 1
                if self._stuck_debug_counter % 10 == 0:
                    logger.info(f"STUCK DEBUG: avg_movement={avg_movement:.4f}, max_movement={max_movement:.4f}")
                    logger.info(f"  consecutive_steps={self.consecutive_stuck_steps}, penalty={stuck_penalty:.2f}")
                    logger.info(f"  Recent positions (last 5): {[f'({p[0]:.3f},{p[1]:.3f})' for p in positions[-5:]]}")
                    logger.info(f"  Position history length: {len(positions)}")
            
            # Much more generous reset conditions
            elif immediate_movement > self.stuck_reset_threshold or avg_movement > self.stuck_threshold * 1.5:
                if self.consecutive_stuck_steps > 0:
                    logger.debug(f"UNSTUCK: immediate={immediate_movement:.4f}, avg={avg_movement:.4f}")
                    logger.debug(f"  Reducing stuck steps from {self.consecutive_stuck_steps} to {max(0, self.consecutive_stuck_steps - 3)}")
                self.consecutive_stuck_steps = max(0, self.consecutive_stuck_steps - 3)  # Even faster recovery
        
        # Much more lenient no-movement penalty
        position_change = np.linalg.norm(next_state[:2] - state[:2])
        commanded_movement = np.linalg.norm(self.current_action[:2]) if hasattr(self, 'current_action') and len(self.current_action) >= 2 else 0.0
        
        # Only penalize if there's a significant command but absolutely no movement
        if commanded_movement > 0.5 and position_change < 0.001:  # Very strict thresholds
            penalty -= 0.1  # Tiny penalty
            if self._stuck_debug_counter % 20 == 0:
                logger.debug(f"NO MOVEMENT: commanded={commanded_movement:.3f}, actual={position_change:.6f}")
        
        return penalty
    
    def _detect_oscillation(self) -> float:
        """
        Improved oscillation detection that's less prone to false positives.
        
        Returns:
            Penalty value (negative) for oscillation
        """
        penalty = 0.0
        
        if len(self.action_history) < 4:
            return penalty
        
        actions = np.array(self.action_history)
        
        # Check for action reversals (oscillation patterns) - more sophisticated
        if len(actions) >= 4:
            recent_actions = actions[-4:]
            
            # Only check drivetrain actions (first 3 components) for oscillation
            drive_actions = recent_actions[:, :3]
            
            # Check for repetitive patterns more carefully
            # Pattern 1: A-B-A-B oscillation
            similarity_02 = np.dot(drive_actions[0], drive_actions[2]) / (
                np.linalg.norm(drive_actions[0]) * np.linalg.norm(drive_actions[2]) + 1e-8)
            similarity_13 = np.dot(drive_actions[1], drive_actions[3]) / (
                np.linalg.norm(drive_actions[1]) * np.linalg.norm(drive_actions[3]) + 1e-8)
            
            # Pattern 2: A-(-A)-A-(-A) back-and-forth
            opposing_01 = np.dot(drive_actions[0], drive_actions[1])
            opposing_23 = np.dot(drive_actions[2], drive_actions[3])
            
            # Check for significant action magnitudes (avoid penalizing small adjustments)
            action_magnitudes = [np.linalg.norm(a) for a in drive_actions]
            significant_actions = all(mag > 0.2 for mag in action_magnitudes)
            
            if significant_actions:
                # Detect A-B-A-B pattern
                if (similarity_02 > self.oscillation_threshold and 
                    similarity_13 > self.oscillation_threshold and
                    similarity_02 + similarity_13 > 1.4):  # Both pairs must be similar
                    penalty += self.oscillation_penalty
                    logger.debug(f"A-B-A-B OSCILLATION: sim_02={similarity_02:.2f}, sim_13={similarity_13:.2f}")
                
                # Detect back-and-forth pattern (opposing actions)
                elif (opposing_01 < -0.6 and opposing_23 < -0.6):
                    penalty += self.oscillation_penalty * 0.7  # Slightly milder penalty
                    logger.debug(f"BACK-AND-FORTH: opp_01={opposing_01:.2f}, opp_23={opposing_23:.2f}")
        
        # Check for longer-term repetitive behavior
        if len(actions) >= 6:
            # Check if the last 6 actions show a repeating 3-action pattern
            last_6 = actions[-6:][:, :3]  # Only drivetrain actions
            pattern_similarity = np.dot(last_6[0], last_6[3]) + np.dot(last_6[1], last_6[4]) + np.dot(last_6[2], last_6[5])
            pattern_magnitude = np.linalg.norm(last_6[0]) + np.linalg.norm(last_6[1]) + np.linalg.norm(last_6[2])
            
            if pattern_magnitude > 0.5 and pattern_similarity / (pattern_magnitude + 1e-8) > 0.8:
                penalty += self.oscillation_penalty * 0.5
                logger.debug(f"REPEATING PATTERN DETECTED: similarity={pattern_similarity:.2f}")
        
        return penalty
        
        return penalty
    
    def _detect_inefficiency(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        Much more forgiving inefficiency detection.
        
        Returns:
            Penalty value (negative) for inefficiency
        """
        penalty = 0.0
        
        # Temporarily disable all inefficiency penalties to help learning
        return penalty
        
        # Much more forgiving energy waste detection (disabled for now)
        energy_used = np.sum(np.abs(action[:3]))
        position_progress = np.linalg.norm(next_state[:2] - state[:2])
        
        # Only penalize energy waste if it's excessive AND no progress
        if energy_used > 4.0 and position_progress < 0.001:  # Very high thresholds
            penalty -= energy_used * 0.05  # Very reduced penalty
            logger.debug(f"ENERGY WASTE: energy={energy_used:.2f}, progress={position_progress:.3f}")
        
        return penalty
    
    def reset_behavior_tracking(self):
        """
        Reset all behavior tracking variables. Call this at the start of each episode.
        """
        self.position_history.clear()
        self.action_history.clear()
        self.consecutive_stuck_steps = 0
        self.total_energy_used = 0.0
        self.steps_without_progress = 0
        self.last_meaningful_position = None
        logger.debug("Behavior tracking variables reset for new episode")
    
    def debug_robot_responsiveness(self):
        """
        Comprehensive debug method to check if the robot is responding to commands.
        """
        if not self.connected:
            logger.error("Not connected to robot")
            return False
        
        logger.info("=== ROBOT RESPONSIVENESS DEBUG ===")
        
        # Test 1: Check NetworkTables connection
        logger.info("Test 1: NetworkTables Connection")
        logger.info(f"  Instance connected: {self.inst.isConnected()}")
        logger.info(f"  Control table exists: {self.control_table is not None}")
        logger.info(f"  State table exists: {self.state_table is not None}")
        
        if self.control_table:
            logger.info(f"  Control table keys: {list(self.control_table.getKeys())}")
        if self.state_table:
            logger.info(f"  State table keys: {list(self.state_table.getKeys())}")
        
        # Test 2: Check initial position
        logger.info("\nTest 2: Initial Robot State")
        initial_state = self.get_state()
        initial_pos = (initial_state[0], initial_state[1])
        logger.info(f"  Initial position: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
        logger.info(f"  Initial velocity: ({initial_state[3]:.3f}, {initial_state[4]:.3f})")
        
        # Test 3: Send test command and check response
        logger.info("\nTest 3: Command Response Test")
        logger.info("  Sending forward command (1.0 m/s)...")
        
        # Send command
        self.send_action([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(0.5)  # Wait for command to take effect
        
        # Check if position changed
        mid_state = self.get_state()
        mid_pos = (mid_state[0], mid_state[1])
        position_change = np.linalg.norm(np.array(mid_pos) - np.array(initial_pos))
        
        logger.info(f"  After 0.5s: position=({mid_pos[0]:.3f}, {mid_pos[1]:.3f})")
        logger.info(f"  Position change: {position_change:.4f} meters")
        logger.info(f"  Velocity: ({mid_state[3]:.3f}, {mid_state[4]:.3f})")
        
        # Test 4: Continue for longer
        logger.info("\nTest 4: Extended Movement Test")
        time.sleep(1.0)  # Continue moving
        
        final_state = self.get_state()
        final_pos = (final_state[0], final_state[1])
        total_change = np.linalg.norm(np.array(final_pos) - np.array(initial_pos))
        
        logger.info(f"  After 1.5s total: position=({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        logger.info(f"  Total position change: {total_change:.4f} meters")
        logger.info(f"  Final velocity: ({final_state[3]:.3f}, {final_state[4]:.3f})")
        
        # Stop robot
        logger.info("\nTest 5: Stop Command")
        self.send_action([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        time.sleep(0.5)
        
        stop_state = self.get_state()
        logger.info(f"  After stop: velocity=({stop_state[3]:.3f}, {stop_state[4]:.3f})")
        
        # Analysis
        logger.info("\n=== ANALYSIS ===")
        if total_change < 0.01:
            logger.error("❌ ROBOT NOT MOVING: Commands may not be reaching robot or robot may be disabled")
            logger.error("   Check: 1) Robot code is running, 2) Robot is enabled, 3) NetworkTables keys match")
        elif total_change < 0.5:
            logger.warning("⚠️  LIMITED MOVEMENT: Robot moving but much slower than expected")
            logger.warning("   Check: 1) Action scaling in robot code, 2) Robot safety limits")
        else:
            logger.info("✅ ROBOT RESPONSIVE: Commands are being processed correctly")
        
        if abs(final_state[3]) > 0.1 or abs(final_state[4]) > 0.1:
            logger.warning("⚠️  ROBOT NOT STOPPING: Velocity should be near zero after stop command")
        
        return total_change > 0.01

    def detect_wall_collision(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> bool:
        """
        Detect if robot is stuck against a wall (high command but no movement).
        Includes memory to prevent bouncing behavior.
        
        Returns:
            True if wall collision detected
        """
        current_time = time.time()
        
        # Clean old wall collision memories
        self.recent_wall_collisions = [
            (wall_time, pos) for wall_time, pos in self.recent_wall_collisions
            if current_time - wall_time < self.wall_collision_memory_time
        ]
        
        # Check if robot commanded movement but didn't move
        commanded_vel = np.linalg.norm(action[:2])
        actual_movement = np.linalg.norm(next_state[:2] - state[:2])
        
        # Robot position
        x, y = next_state[0], next_state[1]
        
        # More conservative boundary detection to prevent bouncing
        boundary_margin = 1.2  # Larger margin to detect walls earlier
        near_wall = (x < boundary_margin or x > (16 - boundary_margin) or 
                    y < boundary_margin or y > (8 - boundary_margin))
        
        # Check recent collision memory - if we recently collided near this position, be more sensitive
        recent_collision_nearby = False
        for _, (prev_x, prev_y) in self.recent_wall_collisions:
            if np.linalg.norm([x - prev_x, y - prev_y]) < 2.0:  # Within 2m of previous collision
                recent_collision_nearby = True
                break
        
        # Detect collision with lower thresholds if recent collision nearby
        collision_detected = False
        if recent_collision_nearby:
            # More sensitive detection after recent collision
            if near_wall and commanded_vel > 0.3 and actual_movement < 0.02:
                collision_detected = True
        else:
            # Normal detection
            if near_wall and commanded_vel > 0.5 and actual_movement < 0.01:
                collision_detected = True
        
        if collision_detected:
            # Add to memory
            self.recent_wall_collisions.append((current_time, (x, y)))
            
            # Determine which wall we might be stuck against
            wall_info = []
            if x < boundary_margin:
                wall_info.append("left wall")
            if x > (16 - boundary_margin):
                wall_info.append("right wall")
            if y < boundary_margin:
                wall_info.append("bottom wall")
            if y > (8 - boundary_margin):
                wall_info.append("top wall")
            
            logger.debug(f"WALL COLLISION: Stuck against {', '.join(wall_info)} at ({x:.2f}, {y:.2f})")
            return True
        
        return False
    
    def get_wall_escape_action(self, state: np.ndarray) -> List[float]:
        """
        Get a gentle action to escape from wall collision.
        Uses conservative movements to prevent bouncing.
        
        Returns:
            Action vector to escape from wall
        """
        x, y = state[0], state[1]
        
        # Use gentler escape velocities to prevent bouncing
        escape_x = 0.0
        escape_y = 0.0
        
        # More conservative boundaries for escape detection
        boundary = 1.5
        
        if x < boundary:  # Near left wall, move right gently
            escape_x = 0.8  # Reduced from 1.5
        elif x > (16 - boundary):  # Near right wall, move left gently
            escape_x = -0.8  # Reduced from -1.5
        
        if y < boundary:  # Near bottom wall, move up gently
            escape_y = 0.8  # Reduced from 1.5
        elif y > (8 - boundary):  # Near top wall, move down gently
            escape_y = -0.8  # Reduced from -1.5
        
        # If in corner, use diagonal escape but gentler
        if abs(escape_x) > 0 and abs(escape_y) > 0:
            # Reduce both components to prevent overshoot
            escape_x *= 0.7
            escape_y *= 0.7
        
        # If still no clear direction, move toward center gently
        if escape_x == 0 and escape_y == 0:
            center_x, center_y = 8.0, 4.0
            dx = center_x - x
            dy = center_y - y
            
            # Normalize and scale down
            magnitude = np.linalg.norm([dx, dy])
            if magnitude > 0:
                escape_x = 0.6 * (dx / magnitude)  # Gentle movement toward center
                escape_y = 0.6 * (dy / magnitude)
        
        # Add small random component to avoid getting stuck in exact same pattern
        import random
        escape_x += random.uniform(-0.1, 0.1)
        escape_y += random.uniform(-0.1, 0.1)
        
        # Clamp to safe limits
        escape_x = max(-1.0, min(1.0, escape_x))
        escape_y = max(-1.0, min(1.0, escape_y))
        
        logger.debug(f"WALL ESCAPE: pos=({x:.2f},{y:.2f}) -> escape=({escape_x:.2f},{escape_y:.2f})")
        
        return [escape_x, escape_y, 0.0, 0.0, 0.0, 0.0]
    

def main():
    """Main function for testing the agent."""
    print("FRC Reinforcement Learning Agent")
    print("================================")
    
    # Create agent (change team number as needed)
    agent = BasicFRCAgent(team_number=0000)  # Set your team number here
    
    try:
        # Connect to robot
        if agent.connect():
            print("Connected successfully!")
            # Mode selection
            print("\nChoose mode:")
            print("1. Simple movement test (recommended for debugging)")
            print("2. Full random episode test")
            print("3. Train RL agent")
            choice = input("Enter choice (1, 2, or 3): ").strip()
            if choice == "1":
                print("Running simple movement test...")
                agent.test_simple_movement()
            elif choice == "2":
                input("Press Enter to start random test episode (or Ctrl+C to exit)...")
                agent.run_test_episode(duration=15.0, control_rate=20.0)
            elif choice == "3":
                episodes = int(input("Enter number of episodes to train: ").strip())
                for ep in range(episodes):
                    print(f"Episode {ep+1}/{episodes}")
                    agent.reset_behavior_tracking()
                    state = agent.get_state()
                    total_reward = 0.0
                    duration = 10.0
                    rate = 20.0
                    dt = 1.0 / rate
                    start_time = time.time()
                    while time.time() - start_time < duration:
                        action = agent.select_action(state)
                        agent.send_action(action)
                        time.sleep(dt)
                        next_state = agent.get_state()
                        reward = agent.calculate_reward(state, np.array(action), next_state)
                        total_reward += reward
                        agent.remember(state, action, reward, next_state, False)
                        agent.optimize_model()
                        state = next_state
                    print(f"Episode {ep+1} total reward: {total_reward:.2f}")
                print("Training completed.")
            else:
                print("Invalid choice, exiting.")
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    
    finally:
        # Clean disconnect
        agent.disconnect()
        print("Agent stopped.")


if __name__ == "__main__":
    main()
