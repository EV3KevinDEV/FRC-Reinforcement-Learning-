#!/usr/bin/env python3
"""
Simple test of the environment reset functionality.

This script connects to the robot and triggers a few environment resets
to verify the functionality is working properly.
"""

import time
import logging
from new_agent import HeuristicAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_environment_reset():
    """Test the environment reset functionality."""
    
    logger.info("Testing Environment Reset Functionality")
    logger.info("=" * 50)
    
    # Create agent
    agent = HeuristicAgent(team_number=0, server_ip="localhost")
    
    # Connect to robot
    logger.info("Connecting to robot...")
    if not agent.connect():
        logger.error("Failed to connect to robot")
        return False
    
    logger.info("Successfully connected to robot")
    time.sleep(1.0)  # Wait for connection to stabilize
    
    try:
        # Test multiple resets
        for i in range(3):
            logger.info(f"\nTest Reset #{i+1}")
            logger.info("-" * 30)
            
            # Get state before reset
            state_before = agent.get_state()
            if state_before is not None:
                coral_before = state_before[agent.state_indices['coral_count']]
                robot_x_before = state_before[agent.state_indices['x']]
                robot_y_before = state_before[agent.state_indices['y']]
                logger.info(f"Before reset - Coral: {coral_before}, Robot pos: ({robot_x_before:.2f}, {robot_y_before:.2f})")
            
            # Trigger reset
            logger.info("Triggering environment reset...")
            agent.reset()
            
            # Wait a moment for changes to take effect
            time.sleep(0.5)
            
            # Get state after reset
            state_after = agent.get_state()
            if state_after is not None:
                coral_after = state_after[agent.state_indices['coral_count']]
                robot_x_after = state_after[agent.state_indices['x']]
                robot_y_after = state_after[agent.state_indices['y']]
                logger.info(f"After reset  - Coral: {coral_after}, Robot pos: ({robot_x_after:.2f}, {robot_y_after:.2f})")
                
                # Verify reset worked
                if abs(robot_x_after - 3.0) < 0.1 and abs(robot_y_after - 3.0) < 0.1:
                    logger.info("✓ Robot position reset correctly")
                else:
                    logger.warning(f"✗ Robot position not reset correctly (expected ~3,3)")
                
                if coral_after == 15:
                    logger.info("✓ Coral count reset correctly")
                else:
                    logger.warning(f"✗ Coral count not reset correctly (expected 15, got {coral_after})")
            else:
                logger.error("Failed to get state after reset")
            
            # Wait between tests
            if i < 2:  # Don't wait after the last test
                time.sleep(2.0)
    
    except Exception as e:
        logger.error(f"Test error: {e}")
        return False
    
    finally:
        # Disconnect
        logger.info("\nDisconnecting from robot...")
        agent.disconnect()
    
    logger.info("\nEnvironment reset test completed!")
    return True

if __name__ == "__main__":
    test_environment_reset()
