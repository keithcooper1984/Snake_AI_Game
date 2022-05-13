Snake game build to practice reinforcement learning.

Requirements
------------

Pygame
Pytorch
Numpy


File Sets
----------

# Shared by all #
arial.ttf: font file used in game
helper.py: Various helper functions

# human implimentation
Snake.py: Snake game playable by humans, controls are keyboard arrows. Built on pygame

# Original implimentation - shows danger to left, right, up and down, from snakes perspective#
Snake_ai.py: Snake game for AI to play. Built on pygame
model.py: Model to learn how to play snake. Uses Pytorch
agent.py: Controller for AI game and model

# 5*5 implimentation - shows danger in 5*5 box centered on snake head. No perspective#
Snake_ai_25.py: Snake game for AI to play. Built on pygame
model.py_25: Model to learn how to play snake. Uses Pytorch
agent.py_25: Controller for AI game and model




Credits
-------

Initial build based upon design of 'Python Engineer'

game: https://www.youtube.com/watch?v=--nsd2ZeYvs&t=1625s&ab_channel=PythonEngineer
ai: https://www.youtube.com/watch?v=L8ypSXwyBds&ab_channel=freeCodeCamp.org

github: https://github.com/python-engineer/snake-ai-pytorch


Future
------

Currently game model has 11 inputs:
* danger straight ahead
* danger left
* danger right

* direction left
* direction right
* direction up
* direction down

* food left
* food right
* food up
* food down

This limits its ability to learn. It gets stuck at about 50 points, turning into positions with no escape route.
Next stage is to try and expand its horizons, probably by giving it a 5*5 square view centered upon its head.

It might be possible to give it a view of the entire board (say 64 * 64) along with its current direction (for 4097 inputs) but this may be too many and I would rather not go in the direction of perfect information.

Instead, I would like to explore options akin to driving. You may know the position of roads / walls in advance but would only be able to see hazards a little way in advance. This is more scalable and considers object permanence, a subject of interest to me.
