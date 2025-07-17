# RL Agent with Environment Reset

This document explains how to use the enhanced RL agent with automatic environment reset functionality.

## Overview

The `resetRLEnvironment()` method in `Robot.java` and the enhanced `reset()` method in `new_agent.py` work together to provide automatic environment reset at the start of each training episode.

## How It Works

### Robot Side (Java)

1. **Network Table Setup**: The robot creates both `RL_State` and `RL_Control` network tables
2. **Control Command Monitoring**: In `robotPeriodic()`, the robot checks for reset commands
3. **Environment Reset**: When requested, `resetRLEnvironment()` clears all game pieces and spawns new ones
4. **Acknowledgment**: The robot signals completion back to the agent

### Agent Side (Python)

1. **Reset Request**: When `agent.reset()` is called, it sends a reset command via NetworkTables
2. **Wait for Completion**: The agent waits for the robot to acknowledge the reset
3. **State Reset**: The agent resets its internal state for the new episode

## Key Features

### Environment Reset (`resetRLEnvironment()`)
- Clears all existing game pieces from the field
- Resets robot position to (3, 3) with zero rotation
- Spawns 15 new coral pieces in valid locations (avoiding reef exclusion zones)
- Uses field dimensions 16.5m × 8.2m
- Maintains 2.5m exclusion radius around reef centers

### Agent Reset (`agent.reset()`)
- Triggers robot environment reset via NetworkTables
- Waits up to 2 seconds for reset completion
- Resets agent's internal tracking variables:
  - Position and velocity history
  - Obstacle mapping
  - Pathfinding cache
  - Stuck detection counters
  - Episode timing

## Usage

### Basic Usage

```python
from new_agent import HeuristicAgent

# Create and connect agent
agent = HeuristicAgent(team_number=0, server_ip="localhost")
agent.connect()

# Start a new episode (automatically resets environment)
agent.reset()

# Run training loop
for step in range(1000):
    state = agent.get_state()
    action = agent.get_action(state)
    agent.execute_action(action)
    reward = agent.calculate_reward(state, action, next_state)
```

### Training Loop with Multiple Episodes

```python
# Run multiple episodes with automatic reset
for episode in range(10):
    print(f"Starting Episode {episode + 1}")
    
    # Reset for new episode (resets environment automatically)
    agent.reset()
    
    # Run episode
    total_reward = 0
    for step in range(1000):
        state = agent.get_state()
        action = agent.get_action(state)
        agent.execute_action(action)
        # ... training logic ...
        
    print(f"Episode {episode + 1} completed with reward: {total_reward}")
```

### Using the Test Script

Run the provided test script to see the reset functionality in action:

```bash
cd Agent
python test_new_agent_with_reset.py
```

This script will:
- Connect to the robot simulation
- Run 10 training episodes
- Automatically reset the environment between episodes
- Display progress and statistics

## NetworkTables Protocol

### Control Commands (Agent → Robot)
- `RL_Control/reset_environment`: Boolean flag to request environment reset
- `RL_Control/reset_completed`: Boolean flag from robot indicating reset completion

### State Information (Robot → Agent)
- All existing `RL_State` entries (pose, velocities, game pieces, etc.)
- Enhanced tracking for coral pieces and scoring

## Configuration

### Field Parameters
- **Field Size**: 16.5m × 8.2m (standard FRC field)
- **Reef Centers**: Red (13.0, 4.0), Blue (3.0, 4.0)
- **Exclusion Radius**: 2.5m around each reef center
- **Coral Count**: 15 pieces spawned per reset

### Reset Timeout
- **Default**: 2 seconds
- **Configurable**: Modify timeout in `agent.reset()` method

## Troubleshooting

### Reset Not Working
1. Check NetworkTables connection
2. Verify robot simulation is running
3. Check logs for timeout messages

### Environment Not Resetting Properly
1. Verify `checkRLControlCommands()` is called in `robotPeriodic()`
2. Check that `SimulatedArena.getInstance().resetFieldForAuto()` is working
3. Verify coral spawning logic

### Agent Not Recognizing Reset
1. Check `control_table` is properly initialized
2. Verify timeout duration is sufficient
3. Look for error messages in agent logs

## Example Output

```
2025-01-16 10:30:15 - new_agent - INFO - Starting Episode 1
2025-01-16 10:30:15 - new_agent - INFO - Robot environment reset completed successfully
2025-01-16 10:30:25 - new_agent - INFO - Episode 1 Summary:
2025-01-16 10:30:25 - new_agent - INFO -   Total Reward: 45.67
2025-01-16 10:30:25 - new_agent - INFO -   Steps: 856
2025-01-16 10:30:25 - new_agent - INFO -   Coral Change: 3
2025-01-16 10:30:25 - new_agent - INFO -   Final Coral Count: 12
```

## Benefits

1. **Consistent Training**: Each episode starts with the same number of game pieces
2. **Reproducible Results**: Environment state is controlled and predictable
3. **Efficient Learning**: No manual intervention needed between episodes
4. **Realistic Simulation**: Proper game piece distribution matching competition conditions
5. **Automated Training**: Can run overnight or for extended periods without supervision
