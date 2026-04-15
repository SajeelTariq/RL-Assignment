# Eureka BEST Reward Function (Iteration 2
# Score: -156.67 ± 31.67

import numpy as np

def compute_reward(obs, action, next_obs, terminated, truncated, info) -> float:
    # Stronger position-based shaping to guide the agent towards the pad
    distance_reward = -0.2 * (np.abs(next_obs[0]) + 2 * np.abs(next_obs[1]))

    # Reward for slowing down as it approaches the pad
    velocity_reward = -0.1 * (np.abs(next_obs[2]) + np.abs(next_obs[3]))

    # Reward for staying upright
    angle_reward = -0.2 * np.abs(next_obs[4])

    # Reward for making leg contact with the ground
    leg_contact_reward = 1.0 * (next_obs[6] + next_obs[7])

    # Penalty for wasting fuel
    fuel_penalty = -0.05 if action != 0 else 0

    # Penalty for deviation from the center
    center_deviation_penalty = -0.1 * np.abs(next_obs[0])

    # Landing bonus
    landing_bonus = 280 if terminated and next_obs[6] == 1 and next_obs[7] == 1 and np.abs(next_obs[2]) < 0.1 and np.abs(next_obs[3]) < 0.1 else 0

    # Crash penalty
    crash_penalty = -180 if terminated and not (next_obs[6] == 1 and next_obs[7] == 1) else 0

    # Combine all rewards
    reward = distance_reward + velocity_reward + angle_reward + leg_contact_reward + fuel_penalty + center_deviation_penalty + landing_bonus + crash_penalty

    return reward