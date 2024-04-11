import os
import torch
from stable_baselines3 import A2C
from env import FlappyBird
from multiprocessing import Process

# Set up directories
models_dir = f"models/PPO"
log_dir = f"logs/PPO"
best_model = f"models/PPO/26.zip"

#  Forloop to count number of files in dir

# FB Finn: You can make this easier by calling len(os.listdir(models_dir)). len() returns the number of items in a list, and os.listdir() returns a list of the files in a directory.
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
    total_steps = 100
    # Training loop just leave it be
    for i in range(1, 10000000000):
        # Initialize PPO model
        #model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device=device)

        # Initialize A2C model
        model = A2C.load(f"{models_dir}/{count}", env, verbose=1, device=device, tensorboard_log=log_dir)
        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO")
        # Save the model at intervals
        # FB Finn: You are saving the model at models_dir/i. You define i in the for loop, meaning that every time you save the model, it will be saved as models_dir/1, models_dir/2, models_dir/3, etc. This means that when you restart the training, you will overwrite the previous model. You should start the for loop from count instead of 1.
        model.save(f"{models_dir}/{i}")

    # Close the environment
    env.close()


if __name__ == "__main__":
    num_processes = 8  # Set the number of processes you want to run concurrently

    # Create and start processes
    processes = []
    for i in range(num_processes):
        # FB Finn: You should add an extra parameter to the train_model function, which is the process number. This way, you can save the models with different names preventing overwriting.
        process = Process(target=train_model, args=(i,))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

# FB Finn: This is part of your previous code, which you have commented out. Shouldn't you remove it?
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
