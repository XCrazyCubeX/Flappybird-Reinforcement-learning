import os
import torch
from stable_baselines3 import A2C
from env import FlappyBird
from multiprocessing import Process

# Set up directories
models_dir = f"models/PPO"
log_dir = f"logs/PPO"
best_model = f"models/PPO/3.zip"

#  Forloop to count number of files in dir
count = 0
for path in os.listdir(models_dir):
    if os.path.isfile(os.path.join(models_dir, path)):
        count += 1


def train_model(best_model):
    # Check if CUDA is available
    device = "cuda"

    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create Snake environment
    env = FlappyBird()
    env.reset()
    # Define hyperparameters
    total_steps = 10000
    # Training loop just leave it be
    for i in range(1, 10000000000):
        # Initialize PPO model
        #model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)

        # Initialize A2C model
        model = A2C.load(f"{models_dir}/{count}", env, verbose=1, device=device, tensorboard_log=log_dir)
        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO")
        # Save the model at intervals
        model.save(f"{models_dir}/{i}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    num_processes = 8  # Set the number of processes you want to run concurrently

    # Create and start processes
    processes = []
    for i in range(num_processes):
        process = Process(target=train_model, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()


# Check if CUDA is available
# use_cuda = torch.cuda.is_available()
# device = "cuda"
#
# # Create directories if they don't exist
# os.makedirs(models_dir, exist_ok=True)
# os.makedirs(log_dir, exist_ok=True)
#
# # Create Snake environment
# env = SnakeEnv()
# env.reset()
# # Define hyperparameters
# total_steps = 20000
# # Training loop just leave it be
# for i in range(1, 100000000):
#     # Initialize PPO model
#     model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)
#     #model = PPO.load(f"{models_dir}/{count}", env, verbose=1, device=device, tensorboard_log=log_dir)
#     # Train the model
#     model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO3")
#     # Save the model at intervals
#     model.save(f"{models_dir}/{i}")
#
# # Close the environment
# env.close()
