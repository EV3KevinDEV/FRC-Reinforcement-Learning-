"""
Custom Reward Agent - Modular FRC Agent with Customizable Reward Functions

This module provides a flexible agent that inherits all the action and observation
capabilities from BasicFRCAgent but allows you to define your own reward function.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional
import logging

# Add the Agent directory to the path to import BasicFRCAgent
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Agent'))

try:
    from ..Agent.basic_agent import BasicFRCAgent
except ImportError as e:
    logging.error(f"Could not import BasicFRCAgent: {e}")
    logging.error("Make sure the Agent/basic_agent.py file exists and is accessible")
    raise

logger = logging.getLogger(__name__)

class CustomRewardAgent(BasicFRCAgent):
    """
    FRC Agent that uses BasicFRCAgent for all action/observation functionality
    but allows custom reward function implementation.
    
    To use this agent:
    1. Inherit from CustomRewardAgent
    2. Override the calculate_reward method with your custom logic
    3. Optionally override get_reward_info for debugging
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the custom reward agent."""
        super().__init__(*args, **kwargs)
        
        # Custom reward tracking
        self.custom_reward_history = []
        self.reward_components = {}
        
        logger.info("CustomRewardAgent initialized - ready for custom reward functions")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        Default reward function - override this method in your custom agent.
        
        This is a placeholder that calls the original reward function.
        You should override this method with your custom reward logic.
        
        Args:
            state: Previous state vector
            action: Action taken
            next_state: Resulting state vector
            
        Returns:
            Reward value (float)
        """
        # Default to original reward function
        reward = super().calculate_reward(state, action, next_state)
        
        # Track reward components for analysis
        self.reward_components = {
            'total_reward': reward,
            'source': 'default_basic_agent'
        }
        
        self.custom_reward_history.append(reward)
        
        logger.debug(f"Default reward calculated: {reward:.3f}")
        return reward
    
    def get_reward_info(self) -> Dict:
        """
        Get information about the current reward calculation.
        Override this to provide custom reward debugging info.
        
        Returns:
            Dictionary with reward information
        """
        return {
            'reward_components': self.reward_components.copy(),
            'recent_rewards': self.custom_reward_history[-10:] if self.custom_reward_history else [],
            'average_reward': np.mean(self.custom_reward_history) if self.custom_reward_history else 0.0,
            'reward_std': np.std(self.custom_reward_history) if len(self.custom_reward_history) > 1 else 0.0
        }
    
    def reset_reward_tracking(self):
        """Reset reward tracking for new episode."""
        self.custom_reward_history.clear()
        self.reward_components.clear()
        logger.debug("Custom reward tracking reset")
    
    def reset_behavior_tracking(self):
        """Override to include custom reward reset."""
        super().reset_behavior_tracking()
        self.reset_reward_tracking()


class ExampleCustomAgent(CustomRewardAgent):
    """
    Example implementation showing how to create a custom reward function.
    This agent prioritizes different behaviors than the default agent.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Custom reward parameters
        self.movement_reward_scale = 1.0
        self.piece_proximity_scale = 2.0
        self.scoring_reward_scale = 10.0
        self.efficiency_penalty_scale = 0.1
        
        logger.info("ExampleCustomAgent initialized with custom reward parameters")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        Example custom reward function focusing on exploration and efficiency.
        
        This reward function prioritizes:
        1. Smooth movement patterns
        2. Exploration of the field
        3. Energy efficiency
        4. Game piece collection with bonus for speed
        """
        reward = 0.0
        components = {}
        
        # Extract state information
        pose_x, pose_y = next_state[0], next_state[1]
        prev_x, prev_y = state[0], state[1]
        has_game_piece = next_state[9] > 0.5 if len(next_state) > 9 else False
        prev_has_game_piece = state[9] > 0.5 if len(state) > 9 else False
        
        # === 1. MOVEMENT REWARD ===
        movement_distance = np.linalg.norm([pose_x - prev_x, pose_y - prev_y])
        movement_reward = movement_distance * self.movement_reward_scale
        components['movement'] = movement_reward
        reward += movement_reward
        
        # === 2. EXPLORATION BONUS ===
        # Reward for visiting new areas (simplified)
        exploration_bonus = 0.0
        if movement_distance > 0.1:  # Only if actually moving
            field_center_x, field_center_y = 8.0, 4.0
            distance_from_center = np.linalg.norm([pose_x - field_center_x, pose_y - field_center_y])
            
            # Bonus for being in different areas of the field
            if distance_from_center > 3.0:
                exploration_bonus = 0.5  # Reward for exploring edges
            elif 1.5 < distance_from_center < 3.0:
                exploration_bonus = 0.3  # Reward for mid-field exploration
        
        components['exploration'] = exploration_bonus
        reward += exploration_bonus
        
        # === 3. GAME PIECE INTERACTION ===
        if len(next_state) > 16:
            closest_algae_dist = next_state[13]
            closest_coral_dist = next_state[16]
            min_piece_distance = min(closest_algae_dist, closest_coral_dist)
            
            # Proximity reward
            if min_piece_distance < 3.0:
                proximity_reward = (3.0 - min_piece_distance) * self.piece_proximity_scale
                components['piece_proximity'] = proximity_reward
                reward += proximity_reward
            
            # Game piece collection bonus
            if has_game_piece and not prev_has_game_piece:
                collection_bonus = 20.0
                components['piece_collection'] = collection_bonus
                reward += collection_bonus
                logger.info("CUSTOM REWARD: Game piece collected! +20.0")
        
        # === 4. SCORING REWARD ===
        if len(next_state) > 21:
            score_diff = next_state[21]
            prev_score_diff = state[21] if len(state) > 21 else score_diff
            
            if score_diff > prev_score_diff:
                scoring_reward = (score_diff - prev_score_diff) * self.scoring_reward_scale
                components['scoring'] = scoring_reward
                reward += scoring_reward
                logger.info(f"CUSTOM REWARD: Successful score! +{scoring_reward:.1f}")
        
        # === 5. EFFICIENCY PENALTY ===
        action_magnitude = np.linalg.norm(action[:3])
        efficiency_penalty = -action_magnitude * self.efficiency_penalty_scale
        components['efficiency'] = efficiency_penalty
        reward += efficiency_penalty
        
        # === 6. BOUNDARY PENALTY ===
        boundary_penalty = 0.0
        if pose_x < 0.5 or pose_x > 15.5 or pose_y < 0.5 or pose_y > 7.5:
            boundary_penalty = -2.0
        components['boundary'] = boundary_penalty
        reward += boundary_penalty
        
        # Store components for debugging
        self.reward_components = components
        self.reward_components['total_reward'] = reward
        self.reward_components['source'] = 'example_custom_agent'
        
        self.custom_reward_history.append(reward)
        
        logger.debug(f"Custom reward: {reward:.3f} = {components}")
        return reward
    
    def get_reward_info(self) -> Dict:
        """Enhanced reward info with component breakdown."""
        base_info = super().get_reward_info()
        
        # Add custom information
        base_info.update({
            'reward_parameters': {
                'movement_scale': self.movement_reward_scale,
                'proximity_scale': self.piece_proximity_scale,
                'scoring_scale': self.scoring_reward_scale,
                'efficiency_penalty_scale': self.efficiency_penalty_scale
            },
            'component_breakdown': self.reward_components
        })
        
        return base_info


class SimpleRewardAgent(CustomRewardAgent):
    """
    Simple reward function that only cares about game piece collection and scoring.
    Ignores movement efficiency and complex behaviors.
    """
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        Very simple reward: +1 for collecting pieces, +10 for scoring, -0.01 per step.
        """
        reward = -0.01  # Small time penalty
        
        # Game piece collection
        has_game_piece = next_state[9] > 0.5 if len(next_state) > 9 else False
        prev_has_game_piece = state[9] > 0.5 if len(state) > 9 else False
        
        if has_game_piece and not prev_has_game_piece:
            reward += 1.0
            logger.debug("Simple reward: +1.0 for game piece")
        
        # Scoring
        if len(next_state) > 21 and len(state) > 21:
            score_diff = next_state[21]
            prev_score_diff = state[21]
            
            if score_diff > prev_score_diff:
                reward += 10.0
                logger.debug("Simple reward: +10.0 for scoring")
        
        # Boundary penalty
        pose_x, pose_y = next_state[0], next_state[1]
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            reward -= 1.0
        
        self.reward_components = {
            'total_reward': reward,
            'source': 'simple_reward_agent'
        }
        
        self.custom_reward_history.append(reward)
        return reward


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("Custom Reward Agent Module")
    print("=" * 40)
    print("Available agents:")
    print("1. CustomRewardAgent - Base class for custom rewards")
    print("2. ExampleCustomAgent - Example with exploration focus")
    print("3. SimpleRewardAgent - Minimal reward function")
    print()
    print("To use:")
    print("from custom_reward_agent import ExampleCustomAgent")
    print("agent = ExampleCustomAgent()")
    print("# Use like any BasicFRCAgent, but with custom rewards")
