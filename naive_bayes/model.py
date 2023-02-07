import numpy as np


class NaiveBayesClassifier:
    def __init__(self, n_classes: int, n_attr: int, max_attr_val: int, smooth:int=0) -> None:
        self.n_classes = n_classes
        self.n_attr = n_attr
        self.smooth = smooth
        self.num_val_attr = None
        self.class_count = np.zeros((n_classes))
        self.attr_class_count = np.zeros((n_classes, n_attr, max_attr_val + 1))
        self.prob_class = np.zeros((n_classes), dtype=float)
        self.prob_attr_class = np.zeros((n_classes, n_attr, max_attr_val + 1), dtype=float)

    def fit(self, dataset: np.ndarray) -> None:
        self.num_val_attr = np.array([len(np.unique(dataset[:, x])) for x in range(self.n_attr)])
        for sample in dataset:
            sample = sample.astype(int)
            self.class_count[sample[-1]] += 1
            for j in range(self.n_attr):
                self.attr_class_count[sample[-1], j, sample[j]] += 1

        for d in range(self.n_classes):
            self.prob_class[d] = self.class_count[d] / len(dataset)
            for j in range(self.n_attr):
                for v in np.unique(dataset[:, j].astype(int)):
                    smoothing = self.smooth * self.num_val_attr[j]
                    self.prob_attr_class[d, j, v] = (self.attr_class_count[d, j, v] + self.smooth) / (self.class_count[d] + smoothing)

    def predict_sample(self, sample: np.ndarray) -> np.ndarray:
        sample = sample.astype(int)
        norm_const = 0
        result_prob = np.zeros(self.n_classes, dtype=float)
        p = np.zeros(self.n_classes)
        for d in range(self.n_classes):
            p[d] = self.prob_class[d]
            for j in range(self.n_attr):
                smoothing = self.smooth * self.num_val_attr[j]
                prob = self.prob_attr_class[d, j, sample[j]] if sample[j] in range(self.prob_attr_class.shape[2]) else (self.smooth / self.class_count[d] + smoothing)
                p[d] *= prob
            norm_const += p[d]
        for d in range(self.n_classes):
            result_prob[d] = p[d] / norm_const
        return np.argmax(result_prob)

    def predict(self, dataset:np.ndarray) -> np.ndarray:
        result = np.zeros(dataset.shape[0])
        for i, sample in enumerate(dataset):
            pred_class = self.predict_sample(sample)
            result[i] = pred_class
        return result

