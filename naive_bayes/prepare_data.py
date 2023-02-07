import os
import pickle
import numpy as np
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def create_bayes_dataset(directory: str) -> np.ndarray:
    dataset = np.empty((0, 14))
    for filename in os.listdir(directory):
        with open(f"{directory}/{filename}", 'rb') as f:
            data_file = pickle.load(f)
            for i, game in enumerate(data_file["data"]):
                game_state, direction = game
                if i+1 < len(data_file["data"]):
                    next_state, _ = data_file["data"][i+1]
                    if not (next_state["snake_body"][-1] == (30, 60) and len(next_state["snake_body"]) == 3 and len(game_state["snake_body"]) != 3):
                        sample = state_to_sample(
                            game_state, direction).reshape(-1, 14)
                        dataset = np.append(dataset, sample, axis=0)
    return dataset


def state_to_sample(game_state: dict, direction=Direction.LEFT) -> np.ndarray:
    sample = np.zeros((14,))
    # is food left
    sample[0] = 1 if game_state["food"][0] < game_state["snake_body"][-1][0] else 0
    # is food right
    sample[1] = 1 if game_state["food"][0] > game_state["snake_body"][-1][0] else 0
    # is food up
    sample[2] = 1 if game_state["food"][1] < game_state["snake_body"][-1][1] else 0
    # is food down
    sample[3] = 1 if game_state["food"][1] > game_state["snake_body"][-1][1] else 0
    # is food in left line from head
    sample[4] = isFoodOnLeftLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in right line from head
    sample[5] = isFoodOnRightLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in up line from head
    sample[6] = isFoodOnUpLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in down line from head
    sample[7] = isFoodOnDownLine(
        game_state["snake_body"][-1], game_state["food"])
    # is collision left
    sample[8] = int(distanceToObstacleLeft(game_state["snake_body"]) == 0)
    # is collision right
    sample[9] = int(distanceToObstacleRight(game_state["snake_body"]) == 0)
    # is collision up
    sample[10] = int(distanceToObstacleUp(game_state["snake_body"]) == 0)
    # is collision down
    sample[11] = int(distanceToObstacleDown(game_state["snake_body"]) == 0)
    # is food in front
    sample[12] = int(is_food_in_front(game_state))
    # y - choosen direction
    sample[13] = direction.value
    return sample


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


def isFoodOnLeftLine(head, food):
    if head[1] == food[1] and food[0] < head[0]:
        return 1
    return 0


def isFoodOnRightLine(head, food):
    if head[1] == food[1] and food[0] > head[0]:
        return 1
    return 0


def isFoodOnUpLine(head, food):
    if head[0] == food[0] and food[1] < head[1]:
        return 1
    return 0


def isFoodOnDownLine(head, food):
    if head[0] == food[0] and food[1] > head[1]:
        return 1
    return 0


def distanceToObstacleLeft(snake_body: list):
    head_pos = snake_body[-1]
    for segment in snake_body[:-1]:
        if segment[1] == head_pos[1] and segment[0] < head_pos[0]:
            return (head_pos[0] - segment[0] - 30) // 30
    return head_pos[0] // 30


def distanceToObstacleRight(snake_body: list) -> int:
    head_pos = snake_body[-1]
    for segment in snake_body[:-1]:
        if segment[1] == head_pos[1] and segment[0] > head_pos[0]:
            return (segment[0] - head_pos[0] - 30) // 30
    return (270 - head_pos[0]) // 30


def distanceToObstacleUp(snake_body):
    head_pos = snake_body[-1]
    for segment in snake_body[:-1]:
        if segment[0] == head_pos[0] and segment[1] < head_pos[1]:
            return (head_pos[1] - segment[1] - 30) // 30
    return head_pos[1] // 30


def distanceToObstacleDown(snake_body):
    head_pos = snake_body[-1]
    for segment in snake_body[:-1]:
        if segment[0] == head_pos[0] and segment[1] > head_pos[1]:
            return (segment[1] - head_pos[1] - 30) // 30
    return (head_pos[1] + 30) // 30
