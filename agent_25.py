import torch
import random
import numpy as np
from collections import deque
from Snake_ai import SnakeGameAI, Direction, Point
from model_25 import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.target = 80 # default = 80
        self.gamma = 0.9 # default = 0.9
        self.block = 20
        self.memory = deque(maxlen = MAX_MEMORY)
        self.model = Linear_QNet(16, 256, 3) # len(state) = 11, hidden_nodes = 256, outputs =3
        self.trainer = QTrainer(self.model, lr = LR, gamma = self.gamma)

    
    def get_state(self, game):
        head = game.snake[0]
        
        # point_1 = Point(head.x+(2*self.block), head.y+(2*self.block))
        # point_2 = Point(head.x+(2*self.block), head.y)
        # point_3 = Point(head.x+(2*self.block), head.y)
        # point_4 = Point(head.x+(2*self.block), head.y)
        # point_5 = Point(head.x+(2*self.block), head.y-(2*self.block))
        # point_6 = Point(head.x+self.block, head.y+(2*self.block))
        point_7 = Point(head.x+self.block, head.y+self.block)
        point_8 = Point(head.x+self.block, head.y)
        point_9 = Point(head.x+self.block, head.y-self.block)
        # point_10 = Point(head.x+self.block, head.y-(2*self.block))
        # point_11 = Point(head.x, head.y+(2*self.block))
        point_12 = Point(head.x, head.y+self.block)
        point_13 = Point(head.x, head.y-self.block)
        # point_14 = Point(head.x, head.y-(2*self.block))
        # point_15 = Point(head.x-self.block, head.y+(2*self.block))
        point_16 = Point(head.x-self.block, head.y+self.block)
        point_17 = Point(head.x-self.block, head.y)
        point_18 = Point(head.x-self.block, head.y-self.block)
        # point_19 = Point(head.x-self.block, head.y-(2*self.block))
        # point_20 = Point(head.x-(2*self.block), head.y+(2*self.block))
        # point_21 = Point(head.x-(2*self.block), head.y+self.block)
        # point_22 = Point(head.x-(2*self.block), head.y)
        # point_23 = Point(head.x-(2*self.block), head.y-self.block)
        # point_24 = Point(head.x-(2*self.block), head.y-(2*self.block))

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        danger_dict = { Direction.LEFT:  [  #point_1,point_2,point_3,point_4,point_5,
                                            #point_6,
                                            point_7,point_8,point_9, 
                                            #point_10,
                                            #point_11,
                                            point_12,point_13,
                                            #point_14,
                                            #point_15,
                                            point_16,point_17,point_18
                                            #,point_19,
                                            #point_20,point_21,point_22,point_23,point_24
                                            ],
                        Direction.DOWN:[    #point_20,point_15,point_11,point_6,point_1,point_21,
                                            point_16,point_12,point_7,
                                            #point_2,point_22,
                                            point_17,point_8,
                                            #point_3,point_23,
                                            point_18,point_13,point_9
                                            #,point_4,point_24,point_19,point_14,point_10,point_5
                                            ],
                        Direction.RIGHT:[   #point_24,point_23,point_22,point_21,point_20,point_19,
                                            point_18,point_17,point_16,
                                            #point_15,point_14,
                                            point_13,point_12,
                                            #point_11,point_10,
                                            point_9,point_8,point_7
                                            #,point_6,point_5,point_4,point_3,point_2,point_1
                                            ],
                        Direction.UP:[      #point_5,point_10,point_14,point_19,point_24,point_4,
                                            point_9,point_13,point_18,
                                            #point_23,point_3,
                                            point_8,point_17,
                                            #point_22,point_2,
                                            point_7,point_12,point_16
                                            #,point_21,point_1,point_6,point_11,point_15,point_20
                                            ]
                        }

        danger = [game.is_collision(x) for x in danger_dict[game.direction]]

        state = [
            #dangers
            *danger,

            # move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = self.target - self.n_games
        final_move = [0,0,0]

    
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = Agent()
    game = SnakeGameAI()

    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)

        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)


            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()