import os
import torch
from stable_baselines3 import PPO
from env import FlappyBird
from multiprocessing import Process

# Set up directories
models_dir = f"models/PPO3_NEXT_GEN"
log_dir = f"logs/PPO3_NEXT_GEN"

# Use cuda as device for faster training
# current graphics card used for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(process_num, best_process, best_model):

    # Create directories if they don't exist
    # Directory models_dir
    # Directory log_dir
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create Snake environment
    # env is the environment class inside the env.py
    # env.reset() initializes environment
    env = FlappyBird()
    env.reset()

    # Define how many steps the model trains for
    # after completed this many steps
    # the model will save
    total_steps = 50000

    # Define how many models already exist
    # This will make the program keep going with newest model
    count = 1

    # Create log dir for each process
    log_dir_process = os.path.join(log_dir, f"process_{process_num}")

    # Training loop just leave it be
    # Makes sure the training goes on continuously
    while True:

        # create new model if no models exist`
        # if count == 0
        if count == 0:
            model = PPO("MlpPolicy",
                        env, verbose=1,
                        tensorboard_log=log_dir_process,
                        device=device,
                        )

        # Loads in model that is saved to models_dir
        # process_0_model_0.zip
        else:
            model = PPO.load(f"{models_dir}/process_{best_process}_model_{best_model}",
                             env,
                             verbose=1,
                             device=device,
                             tensorboard_log=log_dir_process,

                             )

            print(f"Loading model {models_dir}/process_{best_process}_model_{best_model}")

        count += 1

        # Make it save the model
        # When you shut down the program
        # maybe saving on higher graph peaks

        # if pygame.QUIT:
        #     model.save(f"{models_dir}/process_{process_num}_model_{count}")

        # Train the model
        model.learn(total_timesteps=total_steps, reset_num_timesteps=False, tb_log_name="PPO3")

        # Save the model at intervals
        # Intervals are total steps
        model.save(f"{models_dir}/process_{process_num}_model_{count}")

        # Makes sure the best model and process
        # Are updated to the most recent
        best_process = process_num
        best_model = count

        # Close the environment
        # Not necessary
        env.close()


if __name__ == "__main__":

    # define which process and model are the best performing
    # see tensorboard graph to see which performs best
    best_process = (input("Please select best performing process: "))
    best_model = (input("please select best performing model: "))

    # Number of processes you want to run
    # Be careful, higher numbers means, -
    # Higher GPU load
    num_processes = 4

    # Create and start processes
    processes = []
    for i in range(num_processes):
        process = Process(target=train_model, args=(i, best_process, best_model))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()