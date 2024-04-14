import os
import torch

from stable_baselines3 import PPO
from env import FlappyBird
from multiprocessing import Process

# Set up directories
models_dir = f"models/PPO"
log_dir = f"logs/PPO"

# Use cuda as device for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(process_num):

    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create Snake environment
    env = FlappyBird()
    env.reset()

    # Define hyperparameters
    total_steps = 50000

    # define how many models already exist
    # This will make the program keep going with newest model
    # NOT BEST
    count = 1

    # Create log dir for each process
    log_dir_process = os.path.join(log_dir, f"process_{process_num}")

    best_process = 4
    best_model = 1

    # Training loop just leave it be
    while True:

        # create new model if no models exist
        # if count == 0
        if count == 0:
            model = PPO("MlpPolicy",
                        env, verbose=1,
                        tensorboard_log=log_dir_process,
                        device=device,
                        )

        # else use best model
        else:
            model = PPO.load(f"{models_dir}/process_{best_process}_model_{best_model}",
                             env,
                             verbose=1,
                             device=device,
                             tensorboard_log=log_dir_process,
                             )
            count += 1

        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO")
        # Save the model at intervals
        model.save(f"{models_dir}/process_{process_num}_model_{count}")

        best_process = process_num
        best_model = count

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