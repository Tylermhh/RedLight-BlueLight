import gymnasium as gym
import numpy as np
from gymnasium import spaces

class RedLightBlueLightEnv(gym.Env):
    """
    Red Light Green Light environment where the agent acts as the referee.
    
    State:
        - Current light status (red/green)
        - Player positions (array of positions from 0 to 100)
        - Player velocities (array of current speeds)
        - Time elapsed in current episode
    
    Actions:
        - 0: Show Red Light
        - 1: Show Green Light
    
    Rewards:
        - Positive reward for catching players moving during red light
        - Small negative reward for each timestep (encourages efficiency)
        - Positive reward when players reach the goal (encourages fair play)
        - Large negative reward if all players are eliminated
    """
    
    def __init__(self, num_players=5, max_steps=500):
        super(RedLightBlueLightEnv, self).__init__()
        
        # Environment parameters
        self.num_players = num_players
        self.max_steps = max_steps
        self.goal_position = 100.0
        self.player_speed_range = (1.0, 4.0)  # Min/max player speeds
        self.red_light_reaction_time = 0.5  # Time it takes players to stop
        self.player_movement_randomness = 0.2  # Randomness in player movement
        self.player_risk_levels = np.zeros(self.num_players, dtype=np.float32)  # Player behavior to take risks
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Red or Green light
        
        # Observation space: light status, player positions, player velocities, time elapsed
        self.observation_space = spaces.Dict({
            'light': spaces.Discrete(2),
            'player_positions': spaces.Box(low=0, high=self.goal_position, shape=(num_players,), dtype=np.float32),
            'player_velocities': spaces.Box(low=0, high=10.0, shape=(num_players,), dtype=np.float32),
            'time_elapsed': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'players_active': spaces.Box(low=0, high=1, shape=(num_players,), dtype=np.int8)
        })
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset environment state
        self.current_step = 0
        self.current_light = 1  # Start with green light
        self.time_since_light_change = 0
        self.consecutive_same_action = 0
        self.green_light_duration = 0
        
        # Initialize players
        self.player_positions = np.zeros(self.num_players, dtype=np.float32)
        self.player_speeds = np.random.uniform(
            self.player_speed_range[0], 
            self.player_speed_range[1], 
            size=self.num_players
        ).astype(np.float32)
        self.player_velocities = np.zeros(self.num_players, dtype=np.float32)
        self.players_active = np.ones(self.num_players, dtype=np.int8)
        
        # Player behavior characteristics
        self.player_risk_levels = np.random.uniform(0, 1, size=self.num_players)
        
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        """Execute one time step in the environment."""
        prev_light = self.current_light
        self.current_light = action

        # Track consecutive same actions
        if prev_light == self.current_light:
            self.consecutive_same_action += 1
        else:
            self.consecutive_same_action = 0
            
        # Track green light duration for potential penalties
        if self.current_light == 1:  # Green light
            self.green_light_duration += 1
        else:
            self.green_light_duration = 0

        if prev_light != self.current_light:
            self.time_since_light_change = 0
        else:
            self.time_since_light_change += 1

        # Update player behaviors and positions
        reward = self._update_players()

        # STRONGER PENALTY for staying green too long - this is key to fixing the issue
        if self.green_light_duration > 15:
            reward -= 0.5 * (self.green_light_duration - 15)

        # Penalize sticking with the same light too long
        if self.consecutive_same_action > 10:
            reward -= 0.3 * (self.consecutive_same_action - 10)

        # Reward for switching lights (increased from original)
        if prev_light != self.current_light:
            reward += 0.5  # Stronger reward for switching lights

        # Time penalty to encourage efficiency
        reward -= 0.01

        # Track progress
        self.current_step += 1
        terminated = False
        truncated = False

        # Identify player statuses
        players_at_goal = (self.player_positions >= self.goal_position)
        active_players = self.players_active == 1
        eliminated_players = ~active_players & ~players_at_goal

        num_active = np.sum(active_players)
        num_goal = np.sum(players_at_goal)
        num_eliminated = np.sum(eliminated_players)

        # Reward players reaching the goal, but decreased slightly
        reward += num_goal * 1.5  # Reduced from 2.0

        # Add a small reward for having caught SOME players (balanced gameplay)
        if num_eliminated > 0 and num_eliminated < self.num_players:
            reward += 0.5 * num_eliminated

        # Penalty if all players eliminated or all win
        if num_goal + num_eliminated == self.num_players:
            if num_goal == self.num_players:
                reward -= 5.0  # Increased penalty for too easy game
            elif num_eliminated == self.num_players:
                reward -= 8.0  # Harsh—caught everyone
            else:
                # Scale reward based on how balanced the outcome was
                balance_factor = min(num_goal, num_eliminated) / max(1, max(num_goal, num_eliminated))
                reward += 5.0 * balance_factor
            terminated = True

        # Clip overachieving players from continuing
        self.players_active[players_at_goal] = 0

        if self.current_step >= self.max_steps:
            truncated = True

        observation = self._get_observation()
        info = {
            'players_caught': num_eliminated,
            'players_at_goal': num_goal,
            'active_players': num_active,
            'green_light_duration': self.green_light_duration,
            'consecutive_same_action': self.consecutive_same_action
        }

        return observation, reward, terminated, truncated, info

    
    def _update_players(self):
        """Update player positions and check for violations."""
        reward = 0
        
        for i in range(self.num_players):
            if not self.players_active[i]:
                continue
                
            # Update player velocity based on light
            if self.current_light == 1:  # Green light
                # Player accelerates gradually on green light
                target_velocity = self.player_speeds[i]
                self.player_velocities[i] = min(
                    target_velocity,
                    self.player_velocities[i] + 0.5 * self.player_speeds[i]
                )
            else:  # Red light
                # Check if player is caught moving during red light
                if (self.time_since_light_change > self.red_light_reaction_time and 
                    self.player_velocities[i] > 0.5):
                    # Player caught moving on red light!
                    if np.random.random() < 0.8:  # 80% chance to catch violators
                        self.players_active[i] = 0
                        reward += 1.5  # Increased reward for catching a player
                
                # Player tries to stop on red light
                self.player_velocities[i] = max(
                    0, 
                    self.player_velocities[i] - (1.0 - self.player_risk_levels[i])
                )
            
            # Add some randomness to player movement
            if np.random.random() < self.player_movement_randomness:
                # Sometimes players make mistakes
                if self.current_light == 0 and self.players_active[i]:  # Red light
                    if np.random.random() < self.player_risk_levels[i]:
                        # Risky players might move a little during red light
                        self.player_velocities[i] = max(0, min(1.0, self.player_velocities[i] + 0.3))
            
            # Update positions
            self.player_positions[i] += self.player_velocities[i]
            
            # Ensure position is within bounds
            self.player_positions[i] = min(self.goal_position, max(0, self.player_positions[i]))
        
        return reward
    
    def _get_observation(self):
        """Return the current observation."""
        return {
            'light': self.current_light,
            'player_positions': self.player_positions,
            'player_velocities': self.player_velocities,
            'time_elapsed': np.array([self.current_step / self.max_steps], dtype=np.float32),
            'players_active': self.players_active
        }
    
    def render(self):
        """Render the environment (text-based for simplicity)."""
        light_status = "🟢 GREEN LIGHT" if self.current_light == 1 else "🔴 RED LIGHT"
        
        # Print game status
        print(f"\nStep: {self.current_step}/{self.max_steps} | {light_status}")
        print("=" * 50)
        
        # Render the game field as a simple ASCII representation
        field = ["_"] * 50
        
        # Add players to field
        for i, (pos, active) in enumerate(zip(self.player_positions, self.players_active)):
            if active:
                idx = min(49, int(pos / self.goal_position * 49))
                field[idx] = "P"
            else:
                continue  # Skip eliminated players
        
        # Add goal line
        field[49] = "|"
        
        print("START " + "".join(field) + " GOAL")
        
        # Print player statuses
        print("\nPlayers:")
        for i in range(self.num_players):
            status = "Active" if self.players_active[i] == 1 else "Eliminated"
            if self.player_positions[i] >= self.goal_position:
                status = "Reached Goal"
            print(f"Player {i+1}: Position {self.player_positions[i]:.1f}/100.0 | Speed: {self.player_velocities[i]:.1f} | {status}")