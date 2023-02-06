import os
import time
import torch
import random
import pickle
from snake import Direction


class QLearningAgent:
    def __init__(self, block_size, bounds, epsilon, discount, lr=0.1, is_training=True, load_qfunction_path=None):
        """ There should be an option to load already trained Q Learning function from the pickled file. You can change
        interface of this class if you want to."""
        self.block_size = block_size
        self.bounds = bounds
        self.is_training = is_training
        self.Q = self.load_qfunction(load_qfunction_path)
        self.obs = None
        self.action = None
        self.epsilon = epsilon
        self.discount = discount
        self.learning_rate = lr

    def act(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        if self.is_training:
            return self.act_train(game_state, reward, is_terminal)
        return self.act_test(game_state, reward, is_terminal)

    def act_train(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        """ Update Q-Learning function for the previous timestep based on the reward, and provide the action for the current timestep.
        Note that if snake died then it is an end of the episode and is_terminal is True. The Q-Learning update step is different."""
        new_obs = self.game_state_to_observation(game_state)
        if random.random() < self.epsilon:
            new_action = random.randint(0, 3)
        else:
            new_action = torch.argmax(self.Q[new_obs])

        if self.action:
            if not is_terminal:
                update = reward + self.discount * torch.max(self.Q[new_obs]) - self.Q[self.obs][self.action]
            else:
                update = reward - self.Q[self.obs][self.action]
            self.Q[self.obs][self.action] += self.learning_rate * update

        self.action = new_action
        self.obs = new_obs
        return Direction(int(new_action))

    @staticmethod
    def game_state_to_observation(game_state: dict) -> tuple:
        gs = game_state
        is_up = int(gs["food"][1] < gs["snake_body"][-1][1])
        is_right = int(gs["food"][0] > gs["snake_body"][-1][0])
        is_down = int(gs["food"][1] > gs["snake_body"][-1][1])
        is_left = int(gs["food"][0] < gs["snake_body"][-1][0])
        collision_left = int(gs["snake_body"][-1][0] == 0 and gs["snake_direction"] == Direction.LEFT)
        collision_right = int(gs["snake_body"][-1][0] == 270 and gs["snake_direction"] == Direction.RIGHT)
        collision_up = int(gs["snake_body"][-1][1] == 0 and gs["snake_direction"] == Direction.UP)
        collision_down = int(gs["snake_body"][-1][1] == 270 and gs["snake_direction"] == Direction.DOWN)
        is_food_in_front = int(QLearningAgent.is_food_in_front(game_state))
        return is_up, is_right, is_down, is_left, collision_left, collision_right, collision_up, collision_down, is_food_in_front, gs["snake_direction"].value

    @staticmethod
    def is_food_in_front(game_state: dict) -> bool:
        food = game_state["food"]
        head = game_state["snake_body"][-1]
        direc = game_state["snake_direction"]
        result = False
        if head[0] < food[0] and head[1] == food[1] and direc == Direction.LEFT:
            result = True
        elif head[0] > food[0] and head[1] == food[1] and direc == Direction.RIGHT:
            result = True
        elif head[1] < food[1] and head[0] == food[0] and direc == Direction.UP:
            result = True
        elif head[1] > food[1] and head[0] == food[0] and direc == Direction.DOWN:
            result = True
        return result

    def act_test(self, game_state: dict, reward: float, is_terminal: bool) -> Direction:
        new_obs = self.game_state_to_observation(game_state)
        new_action = torch.argmax(self.Q[new_obs])
        return Direction(int(new_action))


    def load_qfunction(self, path: str) -> None:
        if path is not None:
            with open("Qlearning/" + path, 'rb') as f:
                qfunction = pickle.load(f)
                return qfunction
        return torch.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4))

    def dump_qfunction(self) -> None:
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y_%m_%d_%H_%M_%S')
        name = f"{self.epsilon}_{self.discount}_{self.learning_rate}"
        with open(f"data/{current_time}_{name}.pickle", 'wb') as f:
            pickle.dump(self.Q, f)

