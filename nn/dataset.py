import os
import pickle
import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset
from enum import Enum


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class BCDataset(Dataset):
    len_data_sample = 19
    norm_vector = np.array([1., 1., 1., 1., 100., 1., 1., 1., 1., 1., 1.,1.,1.,10.,10.,10.,10.,1.,1.])

    def __init__(self, root_dir):
        self.root_dir = root_dir

        xy = self.create_dataset()
        self.x = from_numpy(xy[:,:-1]).float()
        self.y = from_numpy(xy[:,-1]).long()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def create_dataset(self) -> np.ndarray:
        dataset = np.empty((0, BCDataset.len_data_sample))
        for filename in os.listdir(self.root_dir):
            with open(f"{self.root_dir}/{filename}", 'rb') as f:
                data_file = pickle.load(f)
                for i, game in enumerate(data_file["data"]):
                    game_state, direction = game
                    if i+1 < len(data_file["data"]):
                        next_state, _ = data_file["data"][i+1]
                        if not (next_state["snake_body"][-1] == (30, 60) and len(next_state["snake_body"]) == 3 and len(game_state["snake_body"]) != 3):
                            sample = BCDataset.game_state_to_data_sample(
                                game_state, direction).reshape(-1, BCDataset.len_data_sample)
                            dataset = np.append(dataset, sample, axis=0)
        dataset_normed = dataset / self.norm_vector
        return dataset_normed

    @staticmethod
    def game_state_to_data_sample(game_state: dict, direction=Direction.LEFT) -> np.ndarray:
        sample = np.zeros((BCDataset.len_data_sample,))
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
        # is direction UP
        sample[5] = 1 if game_state["snake_direction"].value == 0 else 0
        # is direction DOWN
        sample[6] = 1 if game_state["snake_direction"].value == 2 else 0
        # is direction LEFT
        sample[7] = 1 if game_state["snake_direction"].value == 3 else 0
        # is direction RIGHT
        sample[8] = 1 if game_state["snake_direction"].value == 1 else 0

        # density relative to current snake head orientation
        d1,d2,d3,d4 = BCDataset.calc_rec_density(game_state["snake_body"], game_state["snake_direction"])
        sample[9] = d1
        sample[10] = d2
        sample[11] = d3
        sample[12] = d4

        # calc dist to every obstacle in every direction
        d_up,d_right,d_down, d_left = BCDataset.calc_dist_to_obst_in_every_direction(game_state["snake_body"])
        sample[13] = d_up
        sample[14] = d_right
        sample[15] = d_down
        sample[16] = d_left

        sample[17] = BCDataset.is_food_in_body(game_state["snake_body"], game_state["food"])

        sample[18] = direction.value
        return sample

    @staticmethod
    def calc_rec_density(body: list, cur_dir: Direction) -> tuple:
        head_x, head_y = body[-1]
        q1,q2,q3,q4 = 0,0,0,0 # default rectangle orientation
        for part_x, part_y in body[:-1]:
            if (part_x > head_x and part_y > head_y):
                q1+=1
            elif (part_x > head_x and part_y < head_y):
                q2+=1
            elif (part_x < head_x and part_y > head_y):
                q3+=1
            elif (part_x < head_x and part_y < head_y):
                q4+=1

        sq1 = ((270 - head_x)*(270-head_y)) // 900 + 1
        sq2 = ((270 - head_x)*head_y) // 900 + 1
        sq3 = (head_x*(270-head_y))//900 +1
        sq4 = (head_x*head_y)//900 +1

        q1,q2,q3,q4 = q1/sq1,q2/sq2,q3/sq3,q4/sq4
        if cur_dir == Direction.UP:
            q1,q2,q3,q4 = q4,q3,q2,q1
        elif cur_dir == Direction.LEFT:
            q1,q2,q3,q4 = q3,q1,q4,q2
        elif cur_dir == Direction.RIGHT:
            q1,q2,q3,q4 = q2,q4,q1,q3
        return (q1,q2,q3,q4)

    def calc_dist_to_obst_in_every_direction(body : list):
        head_x,head_y = body[-1]
        d_up = head_y//30 + 1
        d_left = head_x//30 + 1
        d_down = (270-head_y)//30 +1
        d_right = (270 - head_x)//30 +1

        for part_x,part_y in body[:-1]:
            if(head_y == part_y):
                if head_x>part_x:
                    d_left = min(d_left, (head_x-part_x)//30)
                else:
                    d_right = min(d_right, (part_x-head_x)//30)
            if(head_x == part_x):
                if head_y>part_y:
                    d_up = min(d_up, (head_y-part_y)//30)
                else:
                    d_down = min(d_down, (part_y-head_y)//30)

        return d_up,d_right, d_down,d_left

    def is_food_in_body(body, food_pos):
        food_x,food_y = food_pos
        for part_x, part_y in body:
            if part_x == food_x and part_y == food_y:
                return 1
        return 0


if __name__ == "__main__":

    print(BCDataset.norm_vector)
    print(BCDataset.norm_vector[:-1])

    # dataset = BCDataset("data")
    # game_state = {"food": (0, 0),
    #                   "snake_body": [(0,0),(30,0),(30,30)],  # The last element is snake's head
    #                   "snake_direction": Direction.DOWN}
    # a = BCDataset.game_state_to_data_sample(game_state)
    # print(a)
