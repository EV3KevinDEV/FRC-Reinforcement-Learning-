#!/usr/bin/env python3
"""
Test script demonstrating the new RL agent with environment reset functionality.

This script shows how to:
1. Connect to the robot
2. Run training episodes with automatic environment reset
3. Monitor the training progress

Usage:
    python test_new_agent_with_reset.py
"""

import time
import logging
import numpy as np
from new_agent import HeuristicAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_training_episode(agent, episode_num, max_steps=1000):
    """
    Run a single training episode with the agent.
    
    Args:
        agent: The HeuristicAgent instance
        episode_num: Current episode number
        max_steps: Maximum steps per episode
        
    Returns:
        tuple: (total_reward, steps_taken, episode_info)
    """
    logger.info(f"Starting Episode {episode_num}")
    
    # Reset the agent (this will also reset the robot environment)
    agent.reset()
    
    # Get initial state
    state = agent.get_state()
    if state is None:
        logger.error("Failed to get initial state")
        return 0.0, 0, {"error": "No initial state"}
    
    total_reward = 0.0
    steps_taken = 0
    coral_collected = 0
    coral_scored = 0
    
    # Track initial game state
    initial_coral_count = state[agent.state_indices['coral_count']]
    
    for step in range(max_steps):
        # Get action from agent
        action = agent.get_action(state)
        
        # Execute action
        agent.execute_action(action)
        
        # Small delay to allow robot to respond
        time.sleep(0.05)  # 50ms delay
        
        # Get next state
        next_state = agent.get_state()
        if next_state is None:
            logger.warning(f"Failed to get state at step {step}")
            break
        
        # Calculate reward
        reward = agent.calculate_reward(state, action, next_state)
        total_reward += reward
        
        # Track coral collection and scoring
        current_coral_count = next_state[agent.state_indices['coral_count']]
        has_piece = next_state[agent.state_indices['has_piece']] > 0.5
        
        # Update state
        state = next_state
        steps_taken += 1
        
        # Check for episode termination conditions
        done = False
        
        # Episode ends if we've been running for a long time
        if step >= max_steps - 1:
            done = True
            logger.info(f"Episode {episode_num} completed: max steps reached")
        
        # Episode ends if robot gets stuck for too long
        if hasattr(agent, 'desperation_mode') and agent.desperation_mode:
            if agent.desperation_counter > 100:  # Very stuck
                done = True
                logger.info(f"Episode {episode_num} ended: robot stuck")
        
        # Optional: End episode early if performance is very good or very bad
        if total_reward < -100:  # Very poor performance
            done = True
            logger.info(f"Episode {episode_num} ended early: poor performance")
        
        if done:
            break
    
    # Calculate episode statistics
    coral_change = initial_coral_count - current_coral_count
    episode_info = {
        "total_reward": total_reward,
        "steps_taken": steps_taken,
        "coral_collected_estimate": max(0, coral_change),
        "final_coral_count": current_coral_count,
        "has_piece_at_end": has_piece
    }
    
    logger.info(f"Episode {episode_num} Summary:")
    logger.info(f"  Total Reward: {total_reward:.2f}")
    logger.info(f"  Steps: {steps_taken}")
    logger.info(f"  Coral Change: {coral_change}")
    logger.info(f"  Final Coral Count: {current_coral_count}")
    
    return total_reward, steps_taken, episode_info

def main():
    """Main training loop with automatic environment reset."""
    
    logger.info("Starting RL Training with Automatic Environment Reset")
    
    # Create agent
    agent = HeuristicAgent(team_number=0, server_ip="localhost")
    
    # Connect to robot
    logger.info("Connecting to robot...")
    if not agent.connect():
        logger.error("Failed to connect to robot")
        return
    
    # Wait a moment for connection to stabilize
    time.sleep(1.0)
    
    # Training parameters
    num_episodes = 10  # Number of episodes to run
    episode_results = []
    
    try:
        for episode in range(1, num_episodes + 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPISODE {episode}/{num_episodes}")
            logger.info(f"{'='*50}")
            
            # Run the episode
            reward, steps, info = run_training_episode(agent, episode)
            
            # Store results
            episode_results.append({
                'episode': episode,
                'reward': reward,
                'steps': steps,
                **info
            })
            
            # Print running statistics
            if len(episode_results) >= 3:
                recent_rewards = [r['reward'] for r in episode_results[-3:]]
                avg_recent_reward = np.mean(recent_rewards)
                logger.info(f"Average reward (last 3 episodes): {avg_recent_reward:.2f}")
            
            # Brief pause between episodes
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
    
    finally:
        # Disconnect from robot
        logger.info("Disconnecting from robot...")
        agent.disconnect()
        
        # Print final summary
        if episode_results:
            logger.info(f"\n{'='*50}")
            logger.info("TRAINING SUMMARY")
            logger.info(f"{'='*50}")
            
            rewards = [r['reward'] for r in episode_results]
            steps = [r['steps'] for r in episode_results]
            
            logger.info(f"Episodes completed: {len(episode_results)}")
            logger.info(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
            logger.info(f"Best reward: {np.max(rewards):.2f}")
            logger.info(f"Average steps: {np.mean(steps):.1f}")
            
            # Show episode-by-episode results
            logger.info("\nEpisode Results:")
            for result in episode_results:
                logger.info(f"  Episode {result['episode']}: "
                          f"Reward={result['reward']:.2f}, "
                          f"Steps={result['steps']}, "
                          f"Coral={result.get('coral_collected_estimate', 0)}")

if __name__ == "__main__":
    main()
