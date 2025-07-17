"""
Quick test to verify the custom reward agent setup works.
"""

try:
    print("Testing custom reward agent imports...")
    
    # Test the import path
    import sys
    import os
    
    # Add the Agent directory to path
    agent_path = os.path.join(os.path.dirname(__file__), '..', 'Agent')
    print(f"Adding to path: {os.path.abspath(agent_path)}")
    sys.path.append(agent_path)
    
    # Try importing the custom reward agent
    from custom_reward_agent import CustomRewardAgent, ExampleCustomAgent, SimpleRewardAgent
    print("✓ Successfully imported custom reward agents")
    
    # Test creating an agent (without connecting)
    agent = CustomRewardAgent()
    print("✓ Successfully created CustomRewardAgent instance")
    
    # Test the reward function with dummy data
    import numpy as np
    dummy_state = np.random.rand(30)
    dummy_action = np.random.rand(6)
    dummy_next_state = dummy_state + 0.1
    
    reward = agent.calculate_reward(dummy_state, dummy_action, dummy_next_state)
    print(f"✓ Reward function works: {reward:.3f}")
    
    print("\n🎉 Custom Reward Agent setup is working correctly!")
    print("\nYou can now:")
    print("1. Copy reward_template.py to create your own agent")
    print("2. Modify the calculate_reward method")
    print("3. Use agent.connect_to_robot() when ready to train")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure the Agent/basic_agent.py file exists")
    
except Exception as e:
    print(f"❌ Error: {e}")
    
print("\nTest complete!")
