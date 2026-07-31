# =====================================================
# Experiment 15:
# PPO and TRPO for Humanoid Robot Walking and Balance
# =====================================================


import gymnasium as gym

from stable_baselines3 import PPO

from sb3_contrib import TRPO

from stable_baselines3.common.evaluation import evaluate_policy



# ==========================================
# Create Humanoid Environment
# ==========================================


env_name = "Humanoid-v5"


env = gym.make(
    env_name
)


print("Environment Created:")
print(env_name)



# ==========================================
# PPO Algorithm
# ==========================================


print("\nTraining PPO Agent...")


ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    verbose=1
)


ppo_model.learn(
    total_timesteps=50000
)


ppo_model.save(
    "humanoid_ppo_model"
)


print(
    "PPO Training Completed"
)



# Evaluate PPO


ppo_reward, ppo_std = evaluate_policy(
    ppo_model,
    env,
    n_eval_episodes=5
)


print("\nPPO Performance")

print(
    "Average Reward:",
    ppo_reward
)



# ==========================================
# TRPO Algorithm
# ==========================================


print("\nTraining TRPO Agent...")


trpo_model = TRPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    gamma=0.99,
    verbose=1
)


trpo_model.learn(
    total_timesteps=50000
)


trpo_model.save(
    "humanoid_trpo_model"
)


print(
    "TRPO Training Completed"
)



# Evaluate TRPO


trpo_reward, trpo_std = evaluate_policy(
    trpo_model,
    env,
    n_eval_episodes=5
)


print("\nTRPO Performance")

print(
    "Average Reward:",
    trpo_reward
)



# ==========================================
# Compare Results
# ==========================================


print("\n==========================")

print("Algorithm Comparison")

print("==========================")

print(
    "PPO Reward:",
    ppo_reward
)


print(
    "TRPO Reward:",
    trpo_reward
)


if ppo_reward > trpo_reward:

    print(
        "PPO achieved better walking stability"
    )

else:

    print(
        "TRPO achieved better walking stability"
    )



# ==========================================
# Test PPO Agent
# ==========================================


state,info = env.reset()


for step in range(1000):


    action, _ = ppo_model.predict(
        state
    )


    state,reward,terminated,truncated,info = env.step(
        action
    )


    if terminated or truncated:

        state,info = env.reset()



env.close()


print(
    "\nHumanoid walking simulation completed!"
)