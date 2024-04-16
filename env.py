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

# Velocity and physics
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

# Audio for wing and dead
# Leave commented when training
# Really annoying 12 flapping birds
# wing = 'assets/audio/wing.wav'
# hit = 'assets/audio/hit.wav'

##########################
# Flappy bird Environment
# includes:
# the flappy bird game,
# step,
# observations,
# rewards,
# and reset,
##########################


class FlappyBird(gym.Env):
    metadata = {'render.modes': ['human']}

    # initialize flappy bird game, make sure all self values exists
    # Here we also define our observation space and action space

    def __init__(self):
        super(FlappyBird, self).__init__()

        # define action space
        # 2 actions ==
        # flap [1]
        # not flap [0]
        self.action_space = spaces.Discrete(2)

        # previous actions
        self.prev_actions = deque(maxlen=10)

        # initialize time
        self.start_time = time.time()

        # Set screen, clock, bird and pipe
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipe_group = pygame.sprite.Group()

        # Bottom of top pipe
        # and
        # Top of bottom pipe
        self.pipe_lower_y = None
        self.pipe_top_y = None

        # vertical distance to lower pipe
        # and
        # horizontal distance to center of the gap
        self.v_distance_lower_y = None
        self.x_distance_to_gap = None

        # How many pipes have passed.
        # Start with 0..
        self.pipes_passed = 0  # Track the number of pipes passed

        self.ground_group = pygame.sprite.Group()

        # set observation space
        # shape would be 5 because there are 5 observations now
        # distance to ceiling
        # distance to floor,
        # vertical distance to lower pipe
        # horizontal distance to center of the gap
        # and
        # velocity
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(5,), dtype=np.int32)

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
        # bird_x
        bird_x = self.bird.rect[0]

        # Current distance to pipe
        # Using the first pipe of the group
        # Calculating with distance to pipe - bird_x
        min_distance_to_pipe = float("inf")
        for pipe in self.pipe_group:
            distance_to_pipe = pipe.rect[0] - bird_x
            if 0 < distance_to_pipe < min_distance_to_pipe:
                min_distance_to_pipe = distance_to_pipe

        # calculate top of bottom pipe
        # calculate bottom of top pipe
        if self.pipe_group:
            upper_pipe = self.pipe_group.sprites()[1]
            lower_pipe = self.pipe_group.sprites()[0]
            self.pipe_top_y = upper_pipe.rect[1] + upper_pipe.rect[3]
            self.pipe_lower_y = lower_pipe.rect[1]

        """ Observations """
        # velocity of the bird
        # == SPEED
        velocity = self.bird.speed
        distance_to_ceiling = self.bird.rect.top
        distance_to_floor = SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.rect.bottom
        # vertical distance to top of lower pipe
        self.v_distance_lower_y = self.pipe_lower_y - self.bird.rect.bottom

        # calculate y distance to center gap
        # first calculate center y of 2 pipes
        # then pipe y - bird_y
        lower_pipe_bottom_y = self.pipe_group.sprites()[0].rect.bottom
        upper_pipe_top_y = self.pipe_group.sprites()[1].rect.top
        pipe_gap_middle_y = (lower_pipe_bottom_y + upper_pipe_top_y) / 2

        # calculate x distance to center gap
        # first calculate center x of pipe
        # then pipe x - bird_x
        left_pipe_right_x = self.pipe_group.sprites()[0].rect.right
        right_pipe_left_x = self.pipe_group.sprites()[1].rect.left
        pipe_gap_middle_x = (left_pipe_right_x + right_pipe_left_x) / 2

        self.x_distance_to_gap = pipe_gap_middle_x - bird_x

        # observation
        # combining all observations in one array
        observation = np.array([self.v_distance_lower_y, self.x_distance_to_gap,
                                velocity, distance_to_floor, distance_to_ceiling])

        return observation

    ############################################
    # Step function
    # return observation, reward, done, info
    # Step equals every action happened in game
    # Frame by frame
    ############################################
    def step(self, action):
        self.render()
        obs = self._get_observation()

        # Get everything from the reward value
        # Even dead
        # If terminated and truncated == True
        # reset will happen
        reward, terminated, truncated = self.reward_value()
        info = {}

        # Store every action
        # Using deque to create a list with a max of 10
        # Deque inside in __init__
        self.prev_actions.append(action)

        # load in ground, pipes, and bird
        # mechanics etc,
        self.bird.begin()
        self.bird.update()
        self.ground_group.update()
        self.pipe_group.update()

        # If agent chooses 1 instead of 0
        # This to prevent spamming 1
        # also loads in the sound effect
        if action == 1:
            self.bird.bump()
            # pygame.mixer.music.load(wing)
            # pygame.mixer.music.play()

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

        return obs, reward, terminated, truncated, info

    ####################################################
    # Reset function
    # Makes sure the game starts with all observations
    # Resets everything
    ####################################################

    def reset(self, seed=None, options=None):

        # Set every variable to its default value
        # when reset is called everything will be reset
        # reset is called every truncated or terminated
        # or at start
        reward = 0
        truncated = False
        terminated = False

        self.info = {}
        self.prev_actions.clear()
        self.bird = Bird()
        self.pipe_group = pygame.sprite.Group()
        self.ground_group = pygame.sprite.Group()
        self.pipes_passed = 0

        for i in range(2):
            ground = Ground(i * GROUND_WIDTH)
            self.ground_group.add(ground)

        for i in range(2):
            pipes = get_random_pipes(SCREEN_WIDTH * i + 800)
            self.pipe_group.add(pipes[0])
            self.pipe_group.add(pipes[1])

        return self._get_observation(), self.info

    ##################################
    # Render in game objects
    # User interface User experience
    ##################################

    def render(self, mode='human'):

        # render background
        # not start screen
        self.screen.fill((255, 255, 255))
        self.screen.blit(BACKGROUND, (0, 0))

        # Load in begin screen
        # Begin screen only when 0 actions has taken yet
        # if pev_actions == 0
        if len(self.prev_actions) == 0:
            self.screen.blit(BEGIN_IMAGE, (120, 150))

        # Load in bird, pipes and ground
        self.screen.blit(self.bird.image, self.bird.rect.topleft)  # Blit bird image directly onto the screen
        self.pipe_group.draw(self.screen)
        self.ground_group.draw(self.screen)

        # set game ticks per second
        # 100 would be fine
        # 1000 for training
        pygame.display.update()
        self.clock.tick(1000)

    #########################################
    # Defining rewards
    # Using function inside the step function
    # keeps checking for rewards
    #########################################

    def reward_value(self):
        reward = 0
        terminated = False
        truncated = False

        # collision with pipes
        # dead
        # this will be punished with a reward of -1000
        if pygame.sprite.spritecollideany(self.bird, self.pipe_group, pygame.sprite.collide_mask):
            reward -= 1000
            terminated = True
            truncated = True

        # collision with top and bottom ( frame )
        # top
        # this also means dead
        # this will be punished with a reward of -1000
        if self.bird.rect.top <= 0:
            reward -= 1000
            terminated = True
            truncated = True

        # bottom
        # and this too means dead
        # this will be punished with a reward of -1000
        if self.bird.rect.bottom >= SCREEN_HEIGHT - GROUND_HEIGHT:
            reward -= 1000
            terminated = True
            truncated = True

        # reward or penalty based on distance (y) to gap
        # if bird goes a higher than top pipe
        # Reward -= distance
        if self.bird.rect.top < self.pipe_top_y:
            reward -= 30

        # if bird goes lower than lower pipe
        if self.bird.rect.bottom > self.pipe_lower_y:
            reward -= 30

        # if bird is between the gaps path
        # he will be rewarded with + 3
        if self.bird.rect.top > self.pipe_top_y and self.bird.rect.bottom < self.pipe_lower_y:
            reward += 3

        # if bird passed a pipe
        # the bird will be given a reward of + 1000
        for pipe in self.pipe_group:
            if pipe.rect.right < self.bird.rect.left:
                self.pipes_passed += 1
                reward += self.pipes_passed * 10 + 40

        return reward, terminated, truncated

    def _is_off_screen(self, sprite):
        return sprite.rect[0] < -(sprite.rect[2])


##############################
# In game classes
# Bird(), Ground(), Pipes()
# no use modifying this
# don't even look at it
##############################
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
# why is it here all the way in the back?
# don't ask me
def get_random_pipes(xpos):
    size = random.randint(100, 300)
    pipe = Pipe(False, xpos, size)
    pipe_inverted = Pipe(True, xpos, SCREEN_HEIGHT - size - PIPE_GAP)
    return pipe, pipe_inverted


