import gymnasium as gym 
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")
env = Monitor(env)  # Wrap with Monitor
env = DummyVecEnv([lambda: env])  # Wrap the environment in DummyVecEnv

# Define and train the DQN model
model = DQN(
    "MlpPolicy",  # Uses a simple feedforward network
    env,
    learning_rate=0.001,
    buffer_size=50000,
    learning_starts=1000,  # Increased learning start to allow more exploration
    target_update_interval=500,  # Update target network less frequently
    policy_kwargs={"net_arch": [64, 64]},  # Slightly deeper network
    verbose=1
)

# Train the model for longer
print("Starting model training...")
model.learn(total_timesteps=50000)
print("Model training complete!")

# Evaluate the trained model
print("Evaluating model performance...")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean Reward: {mean_reward:.2f}, Std Reward: {std_reward:.2f}")

# Test the model with visualization
print("\nStarting visual test of the trained model...")
obs = env.reset()

# Track performance metrics
episode_count = 0
total_episodes = 5
max_steps_per_episode = 500

while episode_count < total_episodes:
    done = False
    truncated = False
    total_reward = 0
    step_count = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        step_result = env.step(action)
        
        # Handle different possible return formats
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected step result format: {len(step_result)} values")

        total_reward += reward
        step_count += 1

        # Render the environment
        env.render()

        # Optional: break if too many steps to prevent infinite loop
        if step_count >= max_steps_per_episode:
            break

    # Print episode results
    print(f"Episode {episode_count + 1}: Steps = {step_count}, Total Reward = {total_reward}")
    
    # Reset for next episode
    obs = env.reset()
    episode_count += 1

# Close the environment
env.close()

print("\nModel testing complete!")







'''
# pip install gym==0.25.2 tensorflow keras-rl2 pygame==2.6.0 numpy==1.26.0
# go to environment's Lib/site-packages/rl/util.py and replace model_from_config with model_from_json 
# go to environment's Lib/site-packages/rl/agents/dqn.py and replace "if list(model.output.shape) != list((None, self.nb_actions)):" with "if list(model.outputs[0].shape) != list((None, self.nb_actions)):"
'''

'''Tensorflow and keras version'''

# import random
# import numpy as np
# import gym

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Input
# from tensorflow.keras.optimizers import Adam

# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory

# env = gym.make("CartPole-v1")

# states = env.observation_space.shape[0]
# actions = env.action_space.n

# model = Sequential([
#     Flatten(input_shape=(1,states)),  # Explicitly define input shape
#     Dense(24, activation="relu"),
#     Dense(24, activation="relu"),
#     Dense(actions, activation="linear")
# ])

# # Force the model to initialize by calling it with dummy data
# model(np.zeros((1, states)))

# print(model.summary())



# agent = DQNAgent(
#     model=model,
#     memory=SequentialMemory(limit=50000, window_length=1),
#     policy=BoltzmannQPolicy(),
#     nb_actions=actions,
#     nb_steps_warmup=10,
#     target_model_update=1e-2
# )

# optimizer = Adam(learning_rate=0.001)

# agent.compile(optimizer=optimizer, metrics=["mae"])
# agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

# results = agent.test(env, nb_episodes=10, visualize=True)
# print(np.mean(results.history["episode_reward"]))

# env.close()


'''Random actions. To see if env works'''
# episodes = 10

# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = random.choice([0, 1])
#         n_state, reward, done, info = env.step(action)
#         score += reward
#         env.render()

#     print(f"Episode: {episode}  Score: {score}")

# env.close()