import gymnasium as gym
from gymnasium.spaces.utils import flatten, flatten_space

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

from RedLightBlueLightEnv import RedLightBlueLightEnv


class FlattenObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = flatten_space(env.observation_space)
        
    def observation(self, obs):
        return flatten(self.env.observation_space, obs)


# Create and wrap environment
def make_env():
    env = RedLightBlueLightEnv(num_players=5, max_steps=200)
    return FlattenObservation(env)

# Create vectorized environment
env = make_vec_env(make_env, n_envs=4)
env = VecMonitor(env)  # Add monitoring wrapper for better logging

# Create evaluation environment
eval_env = make_vec_env(make_env, n_envs=1)
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path="./models/",
    log_path="./logs/", 
    eval_freq=5000,
    deterministic=True, 
    render=False,
    n_eval_episodes=10
)

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path="./checkpoints/",
    name_prefix="redlight_greenlight_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

######################################
####  Create and train the agent  ####
######################################

# Set up custom logger
new_logger = configure("./logs/", ["stdout", "csv", "tensorboard"])

# initialize agent with balanced hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    ent_coef=0.02,         # Increased entropy for more exploration
    clip_range=0.2,
    gae_lambda=0.95,
    policy_kwargs=dict(
        net_arch=[dict(pi=[128, 64], vf=[128, 64])]  # Complex network
    ),
    tensorboard_log="./tensorboard/"
)

# Set the new logger
model.set_logger(new_logger)

# Train agent
model.learn(
    total_timesteps=750000,
    callback=[eval_callback, checkpoint_callback]
)

# Save the trained model
model.save("redlight_bluelight_referee")

print("Training Complete!")



##################################
####  Test the trained agent  ####
##################################
print("Testing trained model...")

model = PPO.load("redlight_bluelight_referee")
env = make_env()

obs, info = env.reset()
done = False
total_reward = 0
episode_steps = 0
total_red_light = 0
total_green_light = 0
consecutive_same_action = 0
prev_action = None

while not done:
    action, _ = model.predict(obs, deterministic=True)
    
    # Track consecutive same actions
    if prev_action is not None and action == prev_action:
        consecutive_same_action += 1
    else:
        consecutive_same_action = 0
    prev_action = action
    
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    episode_steps += 1
    
    # Count light usage
    if action == 0:  # Red light
        total_red_light += 1
    else:  # Green light
        total_green_light += 1
        
    # Print consecutive same actions when it switches
    if consecutive_same_action == 0 and episode_steps > 1:
        light_name = "RED" if action == 0 else "GREEN"
        print(f"Switched to {light_name} LIGHT after {consecutive_same_action} steps")
    
    env.render()

print(f"Total reward: {total_reward}")
print(f"Red light usage: {total_red_light} steps ({total_red_light/episode_steps*100:.1f}%)")
print(f"Green light usage: {total_green_light} steps ({total_green_light/episode_steps*100:.1f}%)")

# Add a function to visualize the policy
def evaluate_policy_behavior(model, env, episodes=10):
    """Evaluate how the model uses red and green lights across multiple episodes"""
    all_red_counts = []
    all_green_counts = []
    all_rewards = []
    all_caught = []
    all_goals = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        red_count = 0
        green_count = 0
        episode_reward = 0
        
        # Keep track of light switches
        light_switches = 0
        prev_action = None
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            
            # Count light switches
            if prev_action is not None and action != prev_action:
                light_switches += 1
            prev_action = action
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            if action == 0:  # Red light
                red_count += 1
            else:  # Green light
                green_count += 1
        
        all_red_counts.append(red_count)
        all_green_counts.append(green_count)
        all_rewards.append(episode_reward)
        all_caught.append(info['players_caught'])
        all_goals.append(info['players_at_goal'])
        
        print(f"Episode {episode+1}: Red={red_count}, Green={green_count}, "
              f"Switches={light_switches}, Caught={info['players_caught']}, "
              f"Goals={info['players_at_goal']}, Reward={episode_reward:.2f}")
    
    avg_red = sum(all_red_counts) / len(all_red_counts)
    avg_green = sum(all_green_counts) / len(all_green_counts)
    avg_reward = sum(all_rewards) / len(all_rewards)
    avg_caught = sum(all_caught) / len(all_caught)
    avg_goals = sum(all_goals) / len(all_goals)
    
    print(f"\nAverage over {episodes} episodes:")
    print(f"Red light: {avg_red:.2f} steps ({avg_red/(avg_red+avg_green)*100:.1f}%)")
    print(f"Green light: {avg_green:.2f} steps ({avg_green/(avg_red+avg_green)*100:.1f}%)")
    print(f"Players caught: {avg_caught:.2f}")
    print(f"Players at goal: {avg_goals:.2f}")
    print(f"Average reward: {avg_reward:.2f}")

# Run the policy behavior evaluation
print("\nEvaluating policy behavior across multiple episodes:")
evaluate_policy_behavior(model, env, episodes=5)