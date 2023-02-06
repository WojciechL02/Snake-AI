import os
import pickle
import numpy as np
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


def train_test_datasets(dataset: np.ndarray, train_size: float):
    np.random.seed(69)
    np.random.shuffle(dataset)
    split_idx = int(train_size * len(dataset))
    train = dataset[:split_idx]
    test = dataset[split_idx:]
    return train, test


def create_dataset(directory: str) -> np.ndarray:
    dataset = np.empty((0, 16))
    for filename in os.listdir(directory):
        with open(f"{directory}/{filename}", 'rb') as f:
            data_file = pickle.load(f)
            for i, game in enumerate(data_file["data"]):
                game_state, direction = game
                if i+1 < len(data_file["data"]):
                    next_state, _ = data_file["data"][i+1]
                    if not (next_state["snake_body"][-1] == (30, 60) and len(next_state["snake_body"]) == 3 and len(game_state["snake_body"]) != 3):
                        sample = game_state_to_data_sample(
                            game_state, direction).reshape(-1, 16)
                        dataset = np.append(dataset, sample, axis=0)
    return dataset


def game_state_to_data_sample(game_state: dict, direction=Direction.LEFT) -> np.ndarray:
    sample = np.zeros((16,))
    # is food left
    sample[0] = 1 if game_state["food"][0] < game_state["snake_body"][-1][0] else 0
    # is food right
    sample[1] = 1 if game_state["food"][0] > game_state["snake_body"][-1][0] else 0
    # is food up
    sample[2] = 1 if game_state["food"][1] < game_state["snake_body"][-1][1] else 0
    # is food down
    sample[3] = 1 if game_state["food"][1] > game_state["snake_body"][-1][1] else 0
    # snake length
    sample[4] = len(game_state["snake_body"])
    # actual direction
    sample[5] = game_state["snake_direction"].value
    # distance to wall in front
    sample[6] = distanceToWallFront(
        game_state["snake_body"][-1], game_state["snake_direction"])
    # is food in left line from head
    sample[7] = isFoodOnLeftLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in right line from head
    sample[8] = isFoodOnRightLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in up line from head
    sample[9] = isFoodOnUpLine(
        game_state["snake_body"][-1], game_state["food"])
    # is food in down line from head
    sample[10] = isFoodOnDownLine(
        game_state["snake_body"][-1], game_state["food"])
    # distance to obstacle UP
    sample[11] = distanceToObstacleUp(game_state["snake_body"])
    # distance to obstacle DOWN
    sample[12] = distanceToObstacleDown(game_state["snake_body"])
    # distance to obstacle LEFT
    sample[13] = distanceToObstacleLeft(game_state["snake_body"])
    # distance to obstacle RIGHT
    sample[14] = distanceToObstacleRight(game_state["snake_body"])
    # y - choosen direction
    sample[15] = direction.value
    return sample


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


def isWallInFront(head, direction):
    if direction == Direction.UP:
        if head[1] == 0:
            return 1
    elif direction == Direction.DOWN:
        if head[1] == 270:
            return 1
    elif direction == Direction.LEFT:
        if head[0] == 0:
            return 1
    elif direction == Direction.RIGHT:
        if head[0] == 270:
            return 1
    return 0


def distanceToWallFront(head, direction):
    if direction == Direction.UP:
        return head[1] // 30
    elif direction == Direction.DOWN:
        return (270 - head[1]) // 30
    elif direction == Direction.LEFT:
        return head[0] // 30
    return (270 - head[0]) // 30


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
