# all libraries used for the environment
# imports

import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
import random
import time
from collections import deque

####################
# Global Variables
####################

# frame name
pygame.display.set_caption('Flappy Bird')

# Screen width and height

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Velocity
SPEED = 20
GRAVITY = 2.5
GAME_SPEED = 20

# Ground
GROUND_WIDTH = SCREEN_WIDTH
GROUND_HEIGHT = 100

# Pipes
PIPE_WIDTH = 80
PIPE_HEIGHT = 500
PIPE_GAP = 150

# Observation Shape
N_CHANNELS = 3
HEIGHT = SCREEN_HEIGHT
WIDTH = SCREEN_WIDTH
pygame.init()

# Background image
BACKGROUND = pygame.image.load('assets/sprites/background-night.png')
BACKGROUND = pygame.transform.scale(BACKGROUND, (SCREEN_WIDTH, SCREEN_HEIGHT))
BEGIN_IMAGE = pygame.image.load('assets/sprites/message.png').convert_alpha()

wing = 'assets/audio/wing.wav'
hit = 'assets/audio/hit.wav'

initial_epsilon = 0.9  # Initial exploration rate
min_epsilon = 0.3  # Minimum exploration rate
epsilon_decay = 0.995  # Exploration rate decay factor

##########################
# Flappy bird Environment
# includes:
# the flappy bird game
# observations
# rewards
# and reset
##########################


class FlappyBird(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlappyBird, self).__init__()
        self.action_space = spaces.Discrete(2)
        # FB Finn: Still doubting if saving the previous actions will help in the observation space. I think it might be better to remove it.
        self.prev_actions = deque(maxlen=10)
        self.start_time = time.time()  # Initialize start time

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird()

        self.pipe_group = pygame.sprite.Group()

        self.pipe_gap = None
        self.pipe_lower_y = None
        self.pipe_top_y = None
        self.pipes_passed = 0  # Track the number of pipes passed

        self.ground_group = pygame.sprite.Group()
        self.epsilon = initial_epsilon

        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(6 + len(self.prev_actions),), dtype=np.int32)

        # FB Finn: Self.reset() I believe is called by default when the environment is created. You can remove this line.
        self.reset()

        screen.blit(BACKGROUND, (0, 0))
        screen.blit(BEGIN_IMAGE, (120, 150))

    ##################################################
    # Get observation
    # add more observations !!
    # Add bird velocity, Add distances to pipes,
    # Add gap between pipes, Add position of the bird,
    # Add previous actions
    ##################################################
    def _get_observation(self):

        # Current bird position
        # bird_y
        # bird_x
        bird_x = self.bird.rect[0]
        bird_y = self.bird.rect[1]
        bird_velocity = self.bird.speed

        # Current distance to pipe
        # Using the first pipe of the group
        # Calculating with distance to pipe - bird_x
        min_distance_to_pipe = float("inf")
        for pipe in self.pipe_group:
            distance_to_pipe = pipe.rect[0] - bird_x
            if 0 < distance_to_pipe < min_distance_to_pipe:
                min_distance_to_pipe = distance_to_pipe

        # If no pipes available..
        if min_distance_to_pipe == float("inf"):
            min_distance_to_pipe = SCREEN_WIDTH

        # FB Finn: I think this comment can be better haha
        # Calculate gap between pipes
        # I really don't know how l0l,
        # so this is created by chatGPT,
        # I do know, calculate vertical distance between upper and lower pipe

        if self.pipe_group:
            upper_pipe = self.pipe_group.sprites()[1]
            lower_pipe = self.pipe_group.sprites()[0]
            self.pipe_top_y = upper_pipe.rect[1] + upper_pipe.rect[3]
            self.pipe_lower_y = lower_pipe.rect[1]

        ceiling = 0
        floor = SCREEN_HEIGHT - GROUND_HEIGHT
        # observation
        # combining all observations in one
        # :D
        # FB Finn: The velocity, floor and ceiling are constant values. You should remove them from the observation space. The observation space should only contain the variables that change over time. I'm not sure if bird_x changes over time, but if it does not change, you should remove it as well.
        observation = np.array([min_distance_to_pipe, self.pipe_lower_y, self.pipe_top_y, bird_x, bird_y, bird_velocity,
                                floor, ceiling])

        return observation.shape

    ############################################
    # Step function
    # return observation, reward, done, info
    # Step equals every action happened in game
    # Frame by frame
    ############################################
    def step(self, action):
        obs = self._get_observation()
        reward = self.reward_value()
        self.render()
        # FB Finn: These two variables should be local.
        self.terminated = False
        self.truncated = False
        info = {}
        # Store every action
        # Using deque to create a list with a max of 10
        # Deque inside in __init__
        # FB Finn: If you decide to remove the previous actions from the observation space, you should remove this line as well.
        self.prev_actions.append(action)

        self.bird.begin()
        self.bird.update()
        self.ground_group.update()
        self.pipe_group.update()

        # FB Finn: Do definitely NOT implement the exploration vs exploitation logic here. This should be done in the agent.py file. When you implement this logic here, the agent thinks that it chooses an action, but then this action is overridden by a random action. This is not how it should work. The agent should choose an action, and then the environment should execute this action. The environment should not change the action chosen by the agent. The agent can however be set to choose a random action with a certain probability, but this should be done in the agent.py file.
        if np.random.rand() < self.epsilon:
            action = self.action_space.sample()  # Random action
        else:
            self.epsilon = max(self.epsilon * epsilon_decay, min_epsilon)

        # FB Finn: Add a comment here to explain what the if statements are doing
        if action == 1:
            reward -= 100
            self.bird.bump()
            pygame.mixer.music.load(wing)
            pygame.mixer.music.play()
        # FB Finn: It does not make sense to always give a penalty for both not flapping and flapping. You should choose to give a penalty for one of the actions.
        if action == 0:
            reward -= 50
        # Keep adding pipes on screen
        # Don't understand this logic.. YET
        if self._is_off_screen(self.pipe_group.sprites()[0]):
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            self.pipe_group.remove(self.pipe_group.sprites()[0])
            pipes = get_random_pipes(SCREEN_WIDTH * 2)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        if self._is_off_screen(self.ground_group.sprites()[0]):
            self.ground_group.remove(self.ground_group.sprites()[0])
            new_ground = Ground(self.ground_group.sprites()[-1].rect.right)
            self.ground_group.add(new_ground)

        return obs, reward, self.truncated, self.terminated, info

    ####################################################
    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything
    ####################################################

    def reset(self, seed=None, options=None):
        # FB Finn: These two variables should be local.
        self.truncated = False
        self.terminated = False

        self.prev_actions.clear()
        self.bird = Bird()
        self.pipe_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()

        for i in range(2):
            ground = Ground(i * GROUND_WIDTH)
            self.ground_group.add(ground)

        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        self.render()  # Render after adding pipes and ground

        # FB Finn: This variable should be local.
        self.info = {}  # Moved outside the loop
        return self._get_observation(), self.info

    ##################################
    # Render in game objects
    # User interface User experience
    ##################################

    def render(self, mode='human'):
        self.screen.fill((255, 255, 255))
        self.screen.blit(BACKGROUND, (0, 0))

        # Begin screen only when 0 actions has taken yet
        # if pev_actions == 0
        if len(self.prev_actions) == 0:
            self.screen.blit(BEGIN_IMAGE, (120, 150))

        self.screen.blit(self.bird.image, self.bird.rect.topleft)  # Blit bird image directly onto the screen
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        pygame.display.update()
        self.clock.tick(100)

    #########################################
    # Defining rewards
    # Using function inside the step function
    # keeps checking for rewards
    #########################################

    def reward_value(self):
        reward = 0

        current_time = time.time()
        time_elapsed = current_time - self.start_time

        # collision with pipes
        if pygame.sprite.spritecollideany(self.bird, self.pipe_group, pygame.sprite.collide_mask):
            reward -= 1500
            self.terminated = True
            self.truncated = True

        # collision with ceiling and ground
        # top
        if self.bird.rect.top <= 0:
            reward -= 5000
            self.terminated = True
            self.truncated = True

        # bottom
        if self.bird.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            reward -= 5000
            self.terminated = True
            self.truncated = True

        # FB Finn: This is a good idea, maybe you can make the penalty relative to the distance to the ceiling and ground. For example, if the bird is very close to the ceiling, the penalty should be higher than if the bird is far away from the ceiling.
        # dangerous close to ceiling
        if self.bird.rect.bottom > SCREEN_HEIGHT - GROUND_HEIGHT - 50:
            reward -= 500

        # dangerous close to ground
        if self.bird.rect.top < 50:
            reward -= 500

        # reward or penalty based on distance (y) to gap
        if self.bird.rect.top < self.pipe_top_y:
            reward += 150 - (self.pipe_top_y - self.bird.rect.top)
        elif self.bird.rect.bottom > self.pipe_lower_y:
            reward += 150 - (self.bird.rect.bottom - self.pipe_lower_y)

        elif self.bird.rect.top > self.pipe_top_y and self.bird.rect.bottom < self.pipe_lower_y:
            reward += 300

        # reward every second passed
        reward += 5 * int(time_elapsed)
        self.start_time = current_time

        # passed pipe
        for pipe in self.pipe_group:
            if pipe.rect.right < self.bird.rect.left:
                self.pipes_passed += 1
                reward += 2000

        if self.terminated:
            self.reset()
        # FB Finn: Is the printing still necessary?
        print(reward)
        return reward

    def _is_off_screen(self, sprite):
        return sprite.rect[0] < -(sprite.rect[2])

##############################
# In game classes
# Bird(), Ground(), Pipes()
##############################
# FB Finn: I did not check these, because I assume you copied those from the internet (which of course is fine). I will check them if you want me to.
class Bird(pygame.sprite.Sprite):

    def __init__(self):
        pygame.sprite.Sprite.__init__(self)

        self.images = [pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-midflap.png').convert_alpha(),
                       pygame.image.load('assets/sprites/bluebird-downflap.png').convert_alpha()]

        self.speed = SPEED

        self.current_image = 0
        self.image = pygame.image.load('assets/sprites/bluebird-upflap.png').convert_alpha()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = SCREEN_WIDTH / 6
        self.rect[1] = SCREEN_HEIGHT / 2

    def update(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]
        self.speed += GRAVITY

        # Update bird's position based on speed
        self.rect[1] += self.speed

        # Ensure bird doesn't go below ground level
        if self.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT - GROUND_HEIGHT
            self.speed = 0  # Stop the bird from falling further

    def bump(self):
        self.speed = -SPEED

    def begin(self):
        self.current_image = (self.current_image + 1) % 3
        self.image = self.images[self.current_image]


class Pipe(pygame.sprite.Sprite):

    def __init__(self, inverted, xpos, ysize):
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.image.load('assets/sprites/pipe-green.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (PIPE_WIDTH, PIPE_HEIGHT))

        self.rect = self.image.get_rect()
        self.rect[0] = xpos

        if inverted:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect[1] = - (self.rect[3] - ysize)
        else:
            self.rect[1] = SCREEN_HEIGHT - ysize

        self.mask = pygame.mask.from_surface(self.image)

    def update(self):
        self.rect[0] -= GAME_SPEED


class Ground(pygame.sprite.Sprite):

    def __init__(self, xpos):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('assets/sprites/base.png').convert_alpha()
        self.image = pygame.transform.scale(self.image, (GROUND_WIDTH, GROUND_HEIGHT))

        self.mask = pygame.mask.from_surface(self.image)

        self.rect = self.image.get_rect()
        self.rect[0] = xpos
        self.rect[1] = SCREEN_HEIGHT - GROUND_HEIGHT

    def update(self):
        self.rect[0] -= GAME_SPEED


# Pipe randomization....
# Generating various heights for the pipe

def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


