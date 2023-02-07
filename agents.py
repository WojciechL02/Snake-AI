import copy
import pygame
import os
import time
import pickle
import torch
from torch import from_numpy
from snake import Direction

from decision_tree import prepare_data
from decision_tree.model import ID3

from naive_bayes.model import NaiveBayesClassifier
from naive_bayes.prepare_data import state_to_sample

from nn.model import MLP
from nn.dataset import BCDataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


class HumanAgent:
    def __init__(self, block_size, bounds):
        self.block_size = block_size
        self.bounds = bounds
        self.data = []

    def act(self, game_state) -> Direction:
        keys = pygame.key.get_pressed()
        action = game_state["snake_direction"]
        if keys[pygame.K_LEFT]:
            action = Direction.LEFT
        elif keys[pygame.K_RIGHT]:
            action = Direction.RIGHT
        elif keys[pygame.K_UP]:
            action = Direction.UP
        elif keys[pygame.K_DOWN]:
            action = Direction.DOWN

        self.data.append((copy.deepcopy(game_state), action))
        return action

    def dump_data(self):
        os.makedirs("data", exist_ok=True)
        current_time = time.strftime('%Y-%m-%d_%H:%M:%S')
        with open(f"data/{current_time}.pickle", 'wb') as f:
            pickle.dump({"block_size": self.block_size,
                         "bounds": self.bounds,
                         "data": self.data[:-10]}, f)


class MLPAgent:
    def __init__(self, path_to_trained: str):
        self._path_to_trained = path_to_trained
        self.n_attrs = 0
        self.lr = 0
        self.hidden = 0
        self.neurons = 0
        self._extract_model_info_from_path_name()
        self._device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.mlp = MLP(self.n_attrs, 4, self.hidden, self.neurons)
        self.mlp.load_state_dict(torch.load("nn/" + path_to_trained))
        self.mlp.to(self._device)
        self.mlp.eval()

    def _extract_model_info_from_path_name(self):
        model_info = self._path_to_trained.split("_")
        self.n_attrs = int(model_info[0])
        self.lr = float(model_info[1])
        self.hidden = int(model_info[2])
        self.neurons = int(model_info[3])

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        with torch.no_grad():
            data_sample = BCDataset.game_state_to_data_sample(game_state)
            temp_ds = from_numpy(data_sample[:-1]).float().reshape(1, -1)
            maxx = from_numpy(BCDataset.norm_vector[:-1])
            temp_ds /= maxx
            output = self.mlp(temp_ds.to(self._device))
            decision = output.argmax(dim=1, keepdim=True).item()
            return Direction(decision)

    def dump_data(self):
        pass


class DecisionTreeAgent:
    def __init__(self, dataset):
        self.dataset = dataset
        self.id3 = ID3()
        self.id3.fit(self.dataset)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = prepare_data.game_state_to_data_sample(game_state)
        decision = self.id3.decide_sample(data_sample)
        return Direction(decision)

    def dump_data(self):
        pass


class NaiveBayesAgent:
    def __init__(self, dataset):
        self.dataset = dataset
        self.nb = NaiveBayesClassifier(n_classes=4, n_attr=13, max_attr_val=4, smooth=5)
        self.nb.fit(dataset)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = state_to_sample(game_state)
        decision = self.nb.predict_sample(data_sample[:-1])
        return Direction(decision)

    def dump_data(self):
        pass


#=============== FROM SKLEARN ===============#

class RandomForestAgent:
    def __init__(self, dataset):
        self.dataset = dataset
        self.rf = RandomForestClassifier(criterion="entropy", random_state=42)
        y = self.dataset[:, -1]
        X = self.dataset[:, :-1]
        self.rf.fit(X, y)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = prepare_data.game_state_to_data_sample(game_state)
        decision = self.rf.predict(data_sample[:-1].reshape(1, -1))
        return Direction(decision)

    def dump_data(self):
        pass


class SVMAgent:
    def __init__(self, dataset):
        self.dataset = dataset
        self.svm = SVC(decision_function_shape='ovo', random_state=42)
        y = self.dataset[:, -1]
        X = self.dataset[:, :-1]
        self.svm.fit(X, y)

    def act(self, game_state) -> Direction:
        """ Calculate data sample attributes from game_state and run the trained model to predict snake's action/direction"""
        data_sample = prepare_data.game_state_to_data_sample(game_state)
        decision = self.svm.predict(data_sample[:-1].reshape(1, -1))
        return Direction(decision)

    def dump_data(self):
        pass

