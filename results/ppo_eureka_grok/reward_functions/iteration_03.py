# Eureka Iteration 3
# Evaluation score (original reward): -370.14 ± 259.59
# Feedback: The agent is performing very poorly — likely crashing repeatedly or drifting off-screen. The reward signal may not be guiding the agent towards the pad at all. Consider stronger position-based shaping

import numpy as np

def compute_reward(obs, action, next_obs, terminated, truncated, info) -> float:
    distance_to_pad = np.sqrt(next_obs[0]**2 + next_obs[1]**2)
    distance_reward = -0.5 * distance_to_pad

    velocity_reward = -0.2 * (np.abs(next_obs[2]) + np.abs(next_obs[3]))
    if distance_to_pad < 0.5:
        velocity_reward = -0.5 * (np.abs(next_obs[2]) + np.abs(next_obs[3]))

    angle_reward = -0.3 * np.abs(next_obs[4])

    leg_contact_reward = 2.0 * (next_obs[6] + next_obs[7])

    fuel_penalty = -0.1 if action != 0 else 0

    center_deviation_penalty = -0.2 * np.abs(next_obs[0])

    landing_bonus = 300 if terminated and next_obs[6] == 1 and next_obs[7] == 1 and np.abs(next_obs[2]) < 0.1 and np.abs(next_obs[3]) < 0.1 else 0

    crash_penalty = -200 if terminated and not (next_obs[6] == 1 and next_obs[7] == 1) else 0

    reward = distance_reward + velocity_reward + angle_reward + leg_contact_reward + fuel_penalty + center_deviation_penalty + landing_bonus + crash_penalty

    return reward