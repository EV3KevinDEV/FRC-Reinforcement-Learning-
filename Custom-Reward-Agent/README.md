# Custom Reward Agent

This folder contains a modular agent system that allows you to use all the action and observation capabilities from `BasicFRCAgent` while implementing your own custom reward functions.

## What's Included

### Core Files

- **`custom_reward_agent.py`** - Main module with base classes for custom rewards
- **`example_usage.py`** - Complete examples showing how to use custom rewards
- **`reward_template.py`** - Ready-to-use templates for creating your own reward functions

### How It Works

The `CustomRewardAgent` inherits from `BasicFRCAgent`, which means you get:
- ✅ All robot connection and NetworkTables functionality
- ✅ All state observation and action execution
- ✅ All PPO and DQN training capabilities
- ✅ All movement and behavior tracking
- ✅ Environment reset functionality

The **only thing you need to implement** is your custom `calculate_reward()` method!

## Quick Start

### Option 1: Use a Template

1. Copy `reward_template.py` to a new file (e.g., `my_agent.py`)
2. Rename the `YourRewardAgent` class to something meaningful
3. Modify the `calculate_reward` method with your logic
4. Use it like any other agent!

```python
from my_agent import MyCustomAgent

# Create and connect
agent = MyCustomAgent()
agent.connect_to_robot()

# Train with your custom rewards
agent.run_training_session(num_episodes=50)

# Or use PPO with your custom rewards
agent.switch_to_ppo(total_timesteps=100000)
```

### Option 2: Create From Scratch

```python
from custom_reward_agent import CustomRewardAgent
import numpy as np

class MyAgent(CustomRewardAgent):
    def calculate_reward(self, state, action, next_state):
        reward = 0.0
        
        # Your custom reward logic here!
        # Example: +10 for collecting game pieces
        has_piece = next_state[9] > 0.5
        prev_has_piece = state[9] > 0.5
        if has_piece and not prev_has_piece:
            reward += 10.0
        
        return reward

# Use it
agent = MyAgent()
```

## Available State Information

When implementing your reward function, you have access to the full state vector:

```python
def calculate_reward(self, state, action, next_state):
    # Robot position and orientation
    x, y, rotation = next_state[0:3]
    prev_x, prev_y = state[0:2]
    
    # Game state
    has_game_piece = next_state[9] > 0.5
    closest_algae_distance = next_state[13]
    closest_coral_distance = next_state[16]
    score_difference = next_state[21]
    distance_to_reef = next_state[24]
    intake_alignment = next_state[25]
    
    # Actions taken
    vel_x, vel_y, angular_vel = action[0:3]
    shooter_speed, shooter_angle = action[3:5]
    intake_power = action[5]
    
    # Your reward calculation here
    reward = 0.0
    # ...
    return reward
```

## Example Reward Functions

### 1. Simple Game Piece Focus
```python
def calculate_reward(self, state, action, next_state):
    reward = -0.01  # Small time penalty
    
    # +5 for collecting pieces
    if next_state[9] > 0.5 and state[9] <= 0.5:
        reward += 5.0
    
    # +20 for scoring
    if next_state[21] > state[21]:
        reward += 20.0
        
    return reward
```

### 2. Exploration Focused
```python
def calculate_reward(self, state, action, next_state):
    # Reward movement
    movement = np.linalg.norm(next_state[0:2] - state[0:2])
    reward = movement * 2.0
    
    # Bonus for visiting new areas
    x, y = next_state[0:2]
    # ... track visited positions and reward new areas
    
    return reward
```

### 3. Efficiency Focused
```python
def calculate_reward(self, state, action, next_state):
    # Penalty for wasted energy
    energy_used = np.linalg.norm(action[0:3])
    progress_made = np.linalg.norm(next_state[0:2] - state[0:2])
    
    if energy_used > 0:
        efficiency = progress_made / energy_used
        reward = efficiency * 1.0
    else:
        reward = 0.0
        
    return reward
```

## Pre-built Examples

The module includes several example agents:

### `ExampleCustomAgent`
- Focuses on exploration and efficiency
- Rewards smooth movement patterns
- Balances piece collection with energy use

### `SimpleRewardAgent`
- Minimal reward function
- Only cares about piece collection (+1) and scoring (+10)
- Good starting point for testing

### `ScoringFocusedAgent` (in template)
- Only rewards successful scoring
- Ignores everything else
- Good for testing scoring behavior

### `ExplorationAgent` (in template)
- Grid-based exploration rewards
- Encourages visiting new areas of the field
- Good for learning field layout

## Training with Custom Rewards

All training methods work with your custom rewards:

### DQN Training
```python
agent = MyCustomAgent()
agent.connect_to_robot()
agent.run_training_session(num_episodes=100)
```

### PPO Training
```python
agent = MyCustomAgent()
agent.connect_to_robot()
model = agent.switch_to_ppo(total_timesteps=50000)
```

### Quick Testing
```python
agent = MyCustomAgent()
agent.connect_to_robot()
agent.quick_training_demo(episodes=5)
```

## Debugging Your Rewards

Use the built-in reward tracking:

```python
# After some training
reward_info = agent.get_reward_info()
print(f"Average reward: {reward_info['average_reward']}")
print(f"Recent rewards: {reward_info['recent_rewards']}")
print(f"Components: {reward_info['reward_components']}")
```

## Comparing Reward Functions

Use `example_usage.py` to compare different reward functions:

```python
python example_usage.py
```

This will show you how different reward functions behave on the same state transitions.

## Tips for Good Reward Functions

1. **Start Simple** - Begin with basic rewards (piece collection, scoring) and add complexity gradually

2. **Balance Exploration vs Exploitation** - Include small movement rewards to encourage exploration

3. **Use Shaped Rewards** - Reward progress toward goals, not just final achievements

4. **Avoid Reward Hacking** - Make sure your robot can't get rewards through unintended behaviors

5. **Test Frequently** - Use the comparison tools to see how your rewards behave

6. **Debug with Components** - Break your reward into components for easier debugging

## Troubleshooting

### "Could not import BasicFRCAgent"
- Make sure the `Agent/basic_agent.py` file exists
- Check that you're running from the correct directory

### "PPO not available"
- Install Stable Baselines3: `pip install stable-baselines3[extra] gymnasium`

### Robot not responding
- Check NetworkTables connection
- Verify robot simulation is running
- Use `agent.connect_to_robot()` before training

## Next Steps

1. Copy `reward_template.py` to create your own agent
2. Implement your custom reward function
3. Test it with `agent.quick_training_demo()`
4. Train with your preferred method (DQN or PPO)
5. Compare results with different reward functions
6. Iterate and improve!

The beauty of this system is that you can focus purely on **what you want the robot to learn** (the reward function) while leaving all the **how to learn it** (the algorithms) to the existing proven code.
