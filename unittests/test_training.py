import unittest
import pandas as pd
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from training.train import trainNN

class TestTraining(unittest.TestCase):
    def test_train(self):
        # assume you have some prepared data
        X_train = pd.DataFrame({
            'x1': [1.2, 0.3, 1.44, 0.55],
            'x2': [1.55, 1.9, 0.4, 0],
            'x3': [0.4, 3, 2,3, 4,5],
            'x4': [0,4, 1.1, 0.3, 0.1],
        })
        X_test, y_test = None, None
        y_train = pd.Series([0,1,0.3])
        tr = trainNN(X_train,X_test,y_train,y_test)

        tr.fit(X_train, y_train)
        self.assertIsNotNone(tr.model.tree_)


if __name__ == '__main__':
    unittest.main()