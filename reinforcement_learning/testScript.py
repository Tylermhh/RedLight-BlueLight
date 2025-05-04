import numpy as np
from RedLightBlueLightEnv import RedLightGreenLightEnv

# Create environment
env = RedLightGreenLightEnv(num_players=5, max_steps=100)

# Reset environment
observation, info = env.reset()

# Run a simple random agent
total_reward = 0
for _ in range(100):
    # Simple strategy: alternate between red and green with longer green periods
    if observation['light'] == 1 and np.random.random() < 0.2:  # 20% chance to switch from green to red
        action = 0  # Switch to red
    elif observation['light'] == 0 and np.random.random() < 0.4:  # 40% chance to switch from red to green
        action = 1  # Switch to green
    else:
        action = observation['light']  # Keep current light
    
    # Execute action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Render environment (optional)
    env.render()
    
    if terminated or truncated:
        break

print(f"Total reward: {total_reward}")