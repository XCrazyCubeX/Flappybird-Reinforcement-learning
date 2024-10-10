import os
import torch
from stable_baselines3 import PPO
from env import FlappyBird
from multiprocessing import Process, Manager

# Set up directories
models_dir = f"models/PPO3"
log_dir = f"logs/PPO3"

# Use cuda as device for faster training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(process_num, best_process_dict, models_dir, log_dir):

    # Create directories if they don't exist
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create the FlappyBird environment
    env = FlappyBird()
    env.reset()

    # Number of steps to train before saving
    total_steps = 50000

    # Start from the latest model count
    count = 0

    # Log directory for the current process
    log_dir_process = os.path.join(log_dir, f"process_{process_num}")

    # Infinite loop to keep training the model
    while True:

        # Load the best model if available, otherwise create a new model
        best_process = best_process_dict["process"]
        best_model = best_process_dict["model"]

        if count == 0:
            # Start with a new model if this is the first time
            model = PPO("MlpPolicy",
                        env, verbose=1,
                        tensorboard_log=log_dir_process,
                        device=device)
        else:
            # Load the best model to continue training
            model_path = f"{models_dir}/process_{best_process}_model_{best_model}"
            model = PPO.load(model_path,
                             env,
                             verbose=1,
                             device=device,
                             tensorboard_log=log_dir_process)
            print(f"Loading model from {model_path}")

        # Increment the model count
        count += 1

        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO3")

        # Save the model after training
        model_save_path = f"{models_dir}/process_{process_num}_model_{count}"
        model.save(model_save_path)

        # Update the best process and model in the shared dictionary
        best_process_dict["process"] = process_num
        best_process_dict["model"] = count

        # Close the environment
        env.close()


if __name__ == "__main__":

    # Number of processes to run
    num_processes = 4

    # Manager to share best process and model between processes
    with Manager() as manager:

        # Shared dictionary to store the best process and model
        best_process_dict = manager.dict()
        best_process_dict["process"] = int(input("Please select best performing process (default 0): ") or 0)
        best_process_dict["model"] = int(input("Please select best performing model (default 0): ") or 0)

        # Create and start the processes
        processes = []
        for i in range(num_processes):
            process = Process(target=train_model, args=(i, best_process_dict, models_dir, log_dir))
            processes.append(process)
            process.start()

        # Wait for all processes to finish
        for process in processes:
            process.join()
