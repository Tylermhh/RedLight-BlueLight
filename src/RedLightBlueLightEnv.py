import pygame
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class RedLightBlueLightEnv(gym.Env):
    """
    Environment for RL refree agent to train

    State Variables:
        - current light status (red/green)
        - Player positions (array of positions)
        - Player velocities (array of current speeds)
        - Time elapsed in current episode

    Actions:
        - 0: Show Red Light
        - 1: Show Green Light

    Rewards:
        - Positive reward for catching violators (players moving during red light)
        - Small negative reward for each timestep (encourages to catch people faster)
        Options:
            1. Fair and easy refree (just wants to help people play)
                - Positive reward when players reach the goal (to encourage fair play. might be conflicting with first one)
                - Large negative reward if all players are eliminated (to not make the game too hard to play)
            2. Terminator
                - Negative reward whenever a player reach the goal (to encourage to eliminate more)
                - Large positive reward if all players are eliminated (to try to elimate everyone)
    """

    def __init__(self, num_players=5, max_steps=500):
        pass

    def reset(self, seed=None, options=None):
        """Reset env to starting state"""
        pass

    def step(self, action):
        """What one timestep should execute in environment"""
        pass

    def update_players(self):
        """
        Updates player positions and velocities and checks for violators
        To be called in step function to update the environmment based on action taken
        """
        pass

    def _get_observations(self):
        pass

    def render():
        """To render the environment and its state during training or testing"""
        pass

    def close(self):
        """Close anything was used for rendering when done"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()