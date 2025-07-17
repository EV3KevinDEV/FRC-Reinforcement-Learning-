"""
Template for Creating Your Own Custom Reward Functions

Copy this file and modify the calculate_reward method to implement your own reward logic.
All the action and observation functionality from BasicFRCAgent is available.
"""

import numpy as np
import logging
from custom_reward_agent import CustomRewardAgent

logger = logging.getLogger(__name__)


class YourRewardAgent(CustomRewardAgent):
    """
    Template for your custom reward function.
    
    INSTRUCTIONS:
    1. Copy this class and rename it
    2. Modify the __init__ method to set your parameters
    3. Implement your reward logic in calculate_reward
    4. Optionally customize get_reward_info for debugging
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # YOUR CUSTOM PARAMETERS HERE
        # Example parameters you might want to tune:
        self.movement_reward_weight = 1.0      # How much to reward movement
        self.piece_collection_reward = 10.0    # Reward for collecting game pieces
        self.scoring_reward = 20.0             # Reward for scoring
        self.efficiency_penalty = 0.1          # Penalty for wasteful actions
        self.exploration_bonus = 0.5           # Bonus for exploring new areas
        
        # Behavioral preferences
        self.prefer_coral_over_algae = True    # Whether to prefer coral pieces
        self.encourage_smooth_movement = True   # Penalty for jerky movements
        self.penalize_wall_hits = True         # Penalty for hitting walls
        
        logger.info("YourRewardAgent initialized with custom parameters")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        YOUR CUSTOM REWARD FUNCTION - MODIFY THIS!
        
        Available state information (indices may vary):
        - state[0:3]: Robot pose (x, y, rotation)
        - state[9]: Has game piece (0 or 1)
        - state[13]: Distance to closest algae
        - state[14:16]: Closest algae position (x, y)
        - state[16]: Distance to closest coral
        - state[17:19]: Closest coral position (x, y)
        - state[21]: Score difference
        - state[24]: Distance to nearest reef
        - state[25]: Intake alignment
        
        Available action information:
        - action[0:2]: Velocity commands (vx, vy)
        - action[2]: Angular velocity
        - action[3]: Shooter speed
        - action[4]: Shooter angle
        - action[5]: Intake power (-1 to 1)
        
        Args:
            state: Previous state vector
            action: Action that was taken
            next_state: Resulting state vector
            
        Returns:
            float: Reward value (positive = good, negative = bad)
        """
        
        # Initialize reward and components for debugging
        reward = 0.0
        components = {}
        
        # Extract basic state information
        pose_x, pose_y = next_state[0], next_state[1]
        prev_x, prev_y = state[0], state[1]
        has_game_piece = next_state[9] > 0.5 if len(next_state) > 9 else False
        prev_has_game_piece = state[9] > 0.5 if len(state) > 9 else False
        
        # === BASIC MOVEMENT REWARD ===
        # Reward the robot for moving around
        movement_distance = np.linalg.norm([pose_x - prev_x, pose_y - prev_y])
        movement_reward = movement_distance * self.movement_reward_weight
        reward += movement_reward
        components['movement'] = movement_reward
        
        # === GAME PIECE COLLECTION REWARD ===
        # Big reward for successfully collecting a game piece
        if has_game_piece and not prev_has_game_piece:
            collection_reward = self.piece_collection_reward
            reward += collection_reward
            components['piece_collection'] = collection_reward
            logger.info(f"CUSTOM REWARD: Game piece collected! +{collection_reward}")
        
        # === GAME PIECE PROXIMITY REWARD ===
        # Reward for getting closer to game pieces
        if len(next_state) > 16 and not has_game_piece:
            algae_dist = next_state[13]
            coral_dist = next_state[16]
            
            # Choose which piece to pursue based on preference
            if self.prefer_coral_over_algae:
                target_dist = coral_dist
                piece_type = "coral"
            else:
                target_dist = min(algae_dist, coral_dist)
                piece_type = "nearest"
            
            # Previous distance for comparison
            if len(state) > 16:
                prev_algae_dist = state[13]
                prev_coral_dist = state[16]
                prev_target_dist = prev_coral_dist if self.prefer_coral_over_algae else min(prev_algae_dist, prev_coral_dist)
                
                # Reward for getting closer to target piece
                if target_dist < prev_target_dist:
                    proximity_reward = (prev_target_dist - target_dist) * 2.0
                    reward += proximity_reward
                    components['piece_proximity'] = proximity_reward
        
        # === SCORING REWARD ===
        # Reward for successfully scoring
        if len(next_state) > 21 and len(state) > 21:
            score_diff = next_state[21]
            prev_score_diff = state[21]
            
            if score_diff > prev_score_diff:
                scoring_reward = (score_diff - prev_score_diff) * self.scoring_reward
                reward += scoring_reward
                components['scoring'] = scoring_reward
                logger.info(f"CUSTOM REWARD: Successful score! +{scoring_reward}")
        
        # === REEF PROXIMITY REWARD (when carrying piece) ===
        # Reward for approaching reef when carrying a game piece
        if has_game_piece and len(next_state) > 24:
            reef_dist = next_state[24]
            prev_reef_dist = state[24] if len(state) > 24 else reef_dist
            
            if reef_dist < prev_reef_dist:
                reef_approach_reward = (prev_reef_dist - reef_dist) * 1.5
                reward += reef_approach_reward
                components['reef_approach'] = reef_approach_reward
        
        # === EFFICIENCY PENALTY ===
        # Penalty for using too much energy
        action_magnitude = np.linalg.norm(action[:3])  # Movement actions only
        efficiency_penalty = -action_magnitude * self.efficiency_penalty
        reward += efficiency_penalty
        components['efficiency'] = efficiency_penalty
        
        # === SMOOTH MOVEMENT BONUS/PENALTY ===
        if self.encourage_smooth_movement and hasattr(self, 'action_history') and len(self.action_history) > 0:
            prev_action = self.action_history[-1]
            action_change = np.linalg.norm(action[:3] - prev_action[:3])
            
            if action_change > 2.0:  # Jerky movement
                smoothness_penalty = -0.5
                reward += smoothness_penalty
                components['smoothness'] = smoothness_penalty
        
        # === BOUNDARY PENALTY ===
        # Penalty for going out of bounds or hitting walls
        boundary_penalty = 0.0
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            boundary_penalty = -5.0
        elif pose_x < 0.5 or pose_x > 15.5 or pose_y < 0.5 or pose_y > 7.5:
            if self.penalize_wall_hits:
                boundary_penalty = -1.0
        
        reward += boundary_penalty
        components['boundary'] = boundary_penalty
        
        # === EXPLORATION BONUS ===
        # Simple exploration bonus for visiting different areas
        if movement_distance > 0.1:
            field_center_x, field_center_y = 8.0, 4.0
            distance_from_center = np.linalg.norm([pose_x - field_center_x, pose_y - field_center_y])
            
            # Bonus for exploring different areas
            if distance_from_center > 4.0:
                exploration_reward = self.exploration_bonus
                reward += exploration_reward
                components['exploration'] = exploration_reward
        
        # === CUSTOM PENALTIES (ADD YOUR OWN) ===
        # Example: Penalty for trying to score without a game piece
        if not has_game_piece and (action[3] > 0.5 or action[4] > 10.0):  # Shooter commands
            futile_action_penalty = -0.5
            reward += futile_action_penalty
            components['futile_action'] = futile_action_penalty
        
        # === STORE DEBUGGING INFO ===
        self.reward_components = components
        self.reward_components['total_reward'] = reward
        self.reward_components['source'] = 'your_custom_agent'
        
        self.custom_reward_history.append(reward)
        
        # Uncomment for detailed reward logging:
        # logger.debug(f"Custom reward: {reward:.3f} = {components}")
        
        return reward
    
    def get_reward_info(self) -> dict:
        """
        Custom reward information for debugging.
        You can modify this to show the information that matters to you.
        """
        base_info = super().get_reward_info()
        
        # Add your custom debugging information
        base_info.update({
            'custom_parameters': {
                'movement_weight': self.movement_reward_weight,
                'piece_collection_reward': self.piece_collection_reward,
                'scoring_reward': self.scoring_reward,
                'prefer_coral': self.prefer_coral_over_algae,
                'smooth_movement': self.encourage_smooth_movement
            },
            'reward_breakdown': self.reward_components,
            'recent_total_rewards': self.custom_reward_history[-5:] if self.custom_reward_history else []
        })
        
        return base_info


# EXAMPLE: Alternative reward function focusing only on scoring
class ScoringFocusedAgent(CustomRewardAgent):
    """Agent that only cares about scoring - ignores everything else."""
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        reward = -0.01  # Small time penalty
        
        # Only care about successful scoring
        if len(next_state) > 21 and len(state) > 21:
            score_increase = next_state[21] - state[21]
            if score_increase > 0:
                reward += score_increase * 100.0  # HUGE scoring reward
                logger.info(f"SCORING FOCUSED: +{score_increase * 100.0}")
        
        # Basic boundary penalty
        pose_x, pose_y = next_state[0], next_state[1]
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            reward -= 10.0
        
        self.reward_components = {'total_reward': reward, 'source': 'scoring_focused'}
        self.custom_reward_history.append(reward)
        return reward


# EXAMPLE: Agent that prioritizes exploration over everything
class ExplorationAgent(CustomRewardAgent):
    """Agent that just wants to explore the field."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_positions = set()
        self.exploration_grid_size = 1.0  # 1 meter grid
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        reward = 0.0
        
        # Grid-based exploration reward
        pose_x, pose_y = next_state[0], next_state[1]
        grid_x = int(pose_x / self.exploration_grid_size)
        grid_y = int(pose_y / self.exploration_grid_size)
        grid_pos = (grid_x, grid_y)
        
        if grid_pos not in self.visited_positions:
            self.visited_positions.add(grid_pos)
            reward += 5.0  # Big reward for new area
            logger.info(f"EXPLORATION: New area visited! Total areas: {len(self.visited_positions)}")
        
        # Movement reward
        prev_x, prev_y = state[0], state[1]
        movement = np.linalg.norm([pose_x - prev_x, pose_y - prev_y])
        reward += movement * 2.0  # Reward any movement
        
        # Boundary penalty
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            reward -= 10.0
        
        self.reward_components = {'total_reward': reward, 'source': 'exploration_agent'}
        self.custom_reward_history.append(reward)
        return reward


if __name__ == "__main__":
    print("Custom Reward Function Template")
    print("=" * 40)
    print("Available templates:")
    print("1. YourRewardAgent - Full template with all common reward components")
    print("2. ScoringFocusedAgent - Only cares about scoring")
    print("3. ExplorationAgent - Only cares about exploring")
    print()
    print("To use:")
    print("1. Copy the YourRewardAgent class")
    print("2. Rename it to something meaningful")
    print("3. Modify the calculate_reward method")
    print("4. Adjust the parameters in __init__")
    print("5. Use it like: agent = YourRewardAgent(); agent.connect_to_robot()")
