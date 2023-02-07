import numpy as np
from prepare_data import create_dataset, train_test_datasets
from sklearn.metrics import accuracy_score, recall_score, precision_score


class Node:
    def __init__(self, attribute=None, label=None) -> None:
        self.attribute = attribute
        self.label = label
        self.children = {}

    def add_child(self, value: float):
        self.children[int(value)] = Node()

    def __str__(self) -> str:
        return str(self.attribute)


class ID3:
    def __init__(self, max_depth=None) -> None:
        self.root = Node()
        self.max_depth = max_depth
        self.attributes = None
        self.classes = None
        self.dataset = None

    def entropy(self, dataset: np.ndarray) -> float:
        return -1 * sum([f * np.log(f) for f in np.unique(dataset[:, -1], return_counts=True)[1]])

    def splitEntropy(self, attribute: int, dataset: np.ndarray) -> float:
        total = 0
        for j in np.unique(dataset[:, attribute]):
            splitData = dataset[np.where(dataset[:, attribute] == j)]
            splitEntropy = self.entropy(splitData)
            total += len(splitData) / len(dataset) * splitEntropy
        return total

    def infGain(self, attribute: int, dataset: np.ndarray) -> float:
        return self.entropy(dataset) - self.splitEntropy(attribute, dataset)

    def fit(self, dataset: np.ndarray) -> None:
        self.attributes = [i for i in range(len(dataset[0]) - 1)]
        self.classes = np.unique(dataset[:, -1])
        self.dataset = dataset
        self.fit_recurr(self.root, self.attributes, self.dataset, depth=0)

    def fit_recurr(self, node: Node, attributes: list, dataset: np.ndarray, depth=None) -> None:
        classes, counts = np.unique(dataset[:, -1], return_counts=True)

        if len(classes) == 1:
            node.label = classes[0]
            return

        node.label = classes[np.argmax(counts)]

        depth_check = (self.max_depth is not None and depth == self.max_depth)
        if depth_check or len(attributes) == 0:
            return

        d = attributes[np.argmax(self.infGain(attr, dataset)
                                 for attr in attributes)]
        node.attribute = d
        for j in np.unique(dataset[:, d]):
            node.add_child(j)
            new_attributes = [attr for attr in attributes if attr != d]
            split_dataset = dataset[np.where(dataset[:, d] == j)]
            next_depth = None if depth is None else depth+1
            self.fit_recurr(node.children[j], new_attributes, split_dataset, next_depth)

    def decide_sample(self, sample: np.ndarray):
        node = self.root
        while (node.attribute is not None) and (sample[node.attribute] in node.children.keys()):
            node = node.children[sample[node.attribute]]
        return node.label

    def predict(self, inputs: np.ndarray):
        return np.apply_along_axis(self.decide_sample, 1, inputs)


def show_scores(model_name, train_y, train_y_pred, test_y, test_y_pred):
    train_acc = round(accuracy_score(train_y, train_y_pred), 4)
    train_recall = round(recall_score(train_y, train_y_pred, average='macro'), 4)
    train_prec = round(precision_score(train_y, train_y_pred, average='macro'), 4)
    test_acc = round(accuracy_score(test_y, test_y_pred), 4)
    test_recall = round(recall_score(test_y, test_y_pred, average='macro'), 4)
    test_prec = round(precision_score(test_y, test_y_pred, average='macro'), 4)
    print(f"{model_name}:")
    print(f"TRAIN: acc={train_acc}, recall={train_recall}, prec={train_prec}")
    print(f"TEST: acc={test_acc}, recall={test_recall}, prec={test_prec}\n")


def main():
    dataset = create_dataset(directory="data")
    train, test = train_test_datasets(dataset, train_size=0.8)
    y = test[:, -1]

    id3 = ID3()

    # st = time.process_time()
    id3.fit(train)
    # et = time.process_time()
    # print("time: ", et-st, "s\n")

    show_scores("ID3", train[:, -1], id3.predict(train), y, id3.predict(test))


if __name__ == "__main__":
    main()
