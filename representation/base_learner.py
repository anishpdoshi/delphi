import os
import numpy as np
from pprint import pprint

class BaseLearner(object):
    def __init__(self, interface):
        self.interface = interface

    def save(self, location):
        pass

    def load(self, location):
        pass

    def train(self, examples):
        pass

    def run(self, examples):
        pass

    def get_inconsistent_examples(self, examples):
        inputs = [[ex.constant_value() for ex in example[0]] for example in examples]
        preds = self.run(inputs)
        outputs = [example[1].constant_value() for example in examples]
        return [
            example
            for example, pred, output in zip(examples, preds, outputs)
            if pred != output
        ]

    def evaluate(self, examples, metrics, print_results=True):
        inputs = [example[0] for example in examples]
        outputs = np.array([example[1] for example in examples])
        preds = np.array(self.run(inputs))
        
        if preds.dtype == np.bool:
            preds = preds.astype(int)
            outputs = outputs.astype(int)

        results = {}
        for metric in metrics:
            results[metric.__name__] = metric(preds, outputs)
        if print_results:
            print('\n-----Evaluation results----')
            pprint(results)
            print('---------------------------\n')
        return results

    def to_smt2(self):
        pass