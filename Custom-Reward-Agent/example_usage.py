"""
Example script showing how to use CustomRewardAgent with your own reward functions.

This script demonstrates:
1. How to create a custom reward function
2. How to use the agent for training
3. How to compare different reward functions
"""

import numpy as np
import time
import logging
from typing import Dict, List
from custom_reward_agent import CustomRewardAgent, ExampleCustomAgent, SimpleRewardAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MyCustomAgent(CustomRewardAgent):
    """
    Your custom agent - implement your own reward function here!
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Your custom parameters
        self.my_reward_scale = 1.0
        self.prioritize_coral = True  # Example: prefer coral over algae
        
        logger.info("MyCustomAgent initialized")
    
    def calculate_reward(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> float:
        """
        YOUR CUSTOM REWARD FUNCTION GOES HERE!
        
        Feel free to modify this completely based on what you want the robot to learn.
        """
        reward = 0.0
        components = {}
        
        # Extract useful state information
        pose_x, pose_y = next_state[0], next_state[1]
        prev_x, prev_y = state[0], state[1]
        has_game_piece = next_state[9] > 0.5 if len(next_state) > 9 else False
        prev_has_game_piece = state[9] > 0.5 if len(state) > 9 else False
        
        # Example: Basic movement reward
        movement = np.linalg.norm([pose_x - prev_x, pose_y - prev_y])
        movement_reward = movement * 0.1  # Small reward for any movement
        reward += movement_reward
        components['movement'] = movement_reward
        
        # Example: Game piece collection (big reward!)
        if has_game_piece and not prev_has_game_piece:
            piece_reward = 5.0
            reward += piece_reward
            components['piece_collection'] = piece_reward
            logger.info("MY CUSTOM REWARD: Collected game piece! +5.0")
        
        # Example: Coral preference (if you want to prioritize coral)
        if len(next_state) > 18 and self.prioritize_coral:
            coral_dist = next_state[16]
            algae_dist = next_state[13]
            
            if coral_dist < algae_dist and coral_dist < 2.0:
                coral_bonus = 0.5
                reward += coral_bonus
                components['coral_preference'] = coral_bonus
        
        # Example: Scoring reward
        if len(next_state) > 21 and len(state) > 21:
            score_increase = next_state[21] - state[21]
            if score_increase > 0:
                scoring_reward = score_increase * 10.0
                reward += scoring_reward
                components['scoring'] = scoring_reward
                logger.info(f"MY CUSTOM REWARD: Scored! +{scoring_reward:.1f}")
        
        # Example: Boundary penalty
        if pose_x < 0 or pose_x > 16 or pose_y < 0 or pose_y > 8:
            boundary_penalty = -1.0
            reward += boundary_penalty
            components['boundary'] = boundary_penalty
        
        # Store for debugging
        self.reward_components = components
        self.reward_components['total_reward'] = reward
        self.reward_components['source'] = 'my_custom_agent'
        
        self.custom_reward_history.append(reward)
        
        # Uncomment for detailed logging:
        # logger.debug(f"My custom reward: {reward:.3f} = {components}")
        
        return reward


def test_agent_with_custom_reward():
    """Test the custom reward agent functionality."""
    logger.info("Testing Custom Reward Agent")
    
    try:
        # Create your custom agent
        agent = MyCustomAgent()
        
        logger.info("✓ MyCustomAgent created successfully")
        
        # Test the reward function with dummy data
        dummy_state = np.random.rand(30)  # Simulate state vector
        dummy_action = np.random.rand(6)  # Simulate action vector
        dummy_next_state = dummy_state + np.random.rand(30) * 0.1  # Small change
        
        reward = agent.calculate_reward(dummy_state, dummy_action, dummy_next_state)
        logger.info(f"✓ Custom reward function works: {reward:.3f}")
        
        # Test reward info
        reward_info = agent.get_reward_info()
        logger.info(f"✓ Reward components: {reward_info['reward_components']}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Error testing custom agent: {e}")
        return False


def compare_reward_functions():
    """Compare different reward functions on the same state transitions."""
    logger.info("Comparing different reward functions")
    
    # Create different agents
    agents = {
        'Custom': MyCustomAgent(),
        'Example': ExampleCustomAgent(),
        'Simple': SimpleRewardAgent()
    }
    
    # Generate some test state transitions
    num_tests = 5
    results = {name: [] for name in agents.keys()}
    
    for test in range(num_tests):
        # Create dummy state transition
        state = np.random.rand(30)
        action = np.random.rand(6)
        next_state = state + np.random.normal(0, 0.1, 30)
        
        # Make it look more realistic
        next_state[0:2] = state[0:2] + action[0:2] * 0.02  # Position change
        next_state[9] = 1 if np.random.random() < 0.1 else 0  # Sometimes has piece
        
        logger.info(f"\nTest {test + 1}:")
        
        # Get rewards from each agent
        for name, agent in agents.items():
            reward = agent.calculate_reward(state, action, next_state)
            results[name].append(reward)
            logger.info(f"  {name} Agent: {reward:.3f}")
    
    # Summary
    logger.info("\nReward Function Comparison Summary:")
    for name, rewards in results.items():
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        logger.info(f"  {name}: {avg_reward:.3f} ± {std_reward:.3f}")


def run_training_example():
    """Example of how to use the custom agent for training."""
    logger.info("Running training example with custom reward")
    
    try:
        # Create your custom agent
        agent = MyCustomAgent()
        
        # Note: This requires actual robot connection
        # agent.connect_to_robot()
        
        logger.info("To run actual training:")
        logger.info("1. Connect to robot: agent.connect_to_robot()")
        logger.info("2. Run training: agent.run_training_session(num_episodes=50)")
        
        # Simulate training loop structure
        logger.info("Simulated training structure:")
        for episode in range(3):  # Just show the structure
            logger.info(f"  Episode {episode + 1}:")
            logger.info("    - Reset environment")
            logger.info("    - Run episode with custom rewards")
            logger.info("    - Log custom reward info")
            
            # Show reward info structure
            dummy_rewards = np.random.rand(10)
            agent.custom_reward_history.extend(dummy_rewards)
            info = agent.get_reward_info()
            logger.info(f"    - Average reward: {info['average_reward']:.3f}")
        
    except Exception as e:
        logger.error(f"Error in training example: {e}")


if __name__ == "__main__":
    print("Custom Reward Agent Examples")
    print("=" * 50)
    
    # Test the custom agent
    if test_agent_with_custom_reward():
        print("\n✓ Custom agent test passed!")
    else:
        print("\n✗ Custom agent test failed!")
        exit(1)
    
    # Compare reward functions
    print("\n" + "=" * 50)
    compare_reward_functions()
    
    # Show training example  
    print("\n" + "=" * 50)
    run_training_example() 
    
    print("\n" + "=" * 50)
    print("Custom Reward Agent Examples Complete!")
    print("\nTo create your own agent:")
    print("1. Copy MyCustomAgent class")
    print("2. Modify the calculate_reward method")
    print("3. Use agent.connect_to_robot() and agent.run_training_session()")
    # PPO support removed, so omit instructions for switch_to_ppo
