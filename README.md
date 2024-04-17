# Flappy-bird-python
A basic Flappy Bird game made in Python with reinforcement learning

I took the assets from https://github.com/zhaolingzhi/FlapPyBird-master | Credits to him :D

## Current state 

![](https://github.com/LeonMarqs/Flappy-bird-python/blob/master/Screenshot_1.png)

## Install  dependencies
You can install all dependencies with requirements.txt
This file includes every library used for this program
These dependencies are used:

os
torch
pygame
openAI Gymnasium
multiprocessing
numpy
random
time
collections



## Train models
training a model is done inside the agent.py. 
Run agent.py and the terminal will ask you for the best process and model.


If no models exist it will start training a new model.

The models are found inside the models folder.
like this:

process_0_model_0

## play game using the trained agent

Playing the game is done inside loadModel.py

This works the same way.
Terminal will ask for best process and model,
after input the model will start playing the game with the things it has learned from training


## Contribution
It's a simple model, so I'd be very grateful if you could help me to improve the agent



