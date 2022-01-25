import unittest
import numpy as np
from src.data.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset_split(self):
        X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([1, 1, 1, 1, 1, 1, -1, -1, 0, 0])
        test_size = 0.4
        X_train, X_test, y_train, y_test = Dataset.split_dataset(X, y, test_size=test_size)
        unique_labels_y = np.unique(y, return_counts=True)
        unique_labels_train = np.unique(y_train, return_counts=True)
        unique_labels_test = np.unique(y_test, return_counts=True)
        stratified_train_counts = np.round(unique_labels_y[1] * (1-test_size)).astype(int)
        stratified_test_counts = np.round(unique_labels_y[1] * test_size).astype(int)
        assert np.all(stratified_train_counts == unique_labels_train[1])
        assert np.all(stratified_test_counts == unique_labels_test[1])


if __name__ == '__main__':
    unittest.main()
