import pickle
import random
import numpy as np
from river import tree
import pysmt
from pysmt.shortcuts import Symbol, And, Int, Ite, BV, Real, LE, LT, Bool, to_smtlib
from pysmt.typing import INT
from sklearn import metrics

from base_learner import BaseLearner
from utils import parse_interface, clean_examples, generate_variables
from logics_data import load_logics_data

# Try to fit a decision tree to the data, only binary for now
class DecisionTreeLearner(BaseLearner):
    def __init__(self, interface, variable_names=None, is_binary=True):
        super().__init__(interface)
        
        interface_types = [inp_symbol.get_type() for inp_symbol in interface[0]]
        supports_reals = any([itype.is_real_type() for itype in interface_types])
        
        self.variable_names = variable_names or generate_variables(len(interface[0]))
        self.classifier = tree.HoeffdingAdaptiveTreeClassifier(
            # splitter=splitter(),
            grace_period=1,
            leaf_prediction="mc",
            drift_window_threshold=20,
            binary_split=True,
            merit_preprune=False,
        )

    def get_paths(self, location):
        state_path = location + '_dt.pkl'
        return state_path

    def save(self, location):
        state_path = self.get_paths(location)
        pickle.dump(self.classifier, open(state_path, 'wb+'))

    def load(self, location):
        # Load the symb state and StandardScaler
        state_path = self.get_paths(location)
        self.classifier = pickle.load(open(state_path,'rb'))

    def train(self, examples, update_pretrained=False, train_args={}):
        examples = clean_examples(examples)
        for x, y in examples:
            formatted = {var_name: feat for var_name, feat in zip(self.variable_names, x)}
            self.classifier.learn_one(formatted, bool(y))
        print(f'Trained on {len(examples)} examples.')
        if update_pretrained:
            self.save(update_pretrained)
    
    def run(self, inputs):
        outputs = []
        for x in inputs:
            formatted = {var_name: feat for var_name, feat in zip(self.variable_names, x)}
            outputs.append(self.classifier.predict_one(formatted))
        return outputs

    def to_smt2(self):
        df = self.classifier.to_dataframe()
        if df is None:
            return "false"
        df['node_idx'] = df.index
        assert df is not None

        interface_types = [inp_symbol.get_type() for inp_symbol in self.interface[0]]
        supports_reals = any([itype.is_real_type() for itype in interface_types])
        
        def create_node_recur(row_dict):
            if row_dict['is_leaf']:
                return Bool(row_dict.get(True, 0) >= row_dict.get(False, 0))

            row_leq, row_gt = df[df['parent'] == row_dict['node_idx']].to_dict(orient='records')
            feat_symbol = self.interface[0][self.variable_names.index(row_dict['feature'])]
            
            if not supports_reals:
                threshold_val = int(row_dict['threshold'])
                feat_type = feat_symbol.get_type()
                if feat_type.is_bv_type():
                    threshold_val = BV(threshold_val, width=feat_type.width)
                else:
                    threshold_val = Int(threshold_val)
            else:
                threshold_val = Real(row_dict['threshold'])

            return Ite(
                (feat_symbol <= threshold_val),
                create_node_recur(row_leq),
                create_node_recur(row_gt)
            )

        root_dict = df.to_dict(orient='records')[0]
        formula = create_node_recur(root_dict)
        formula_smt = to_smtlib(formula, daggify=False)
        return formula_smt

if __name__ == '__main__':
    # learner = DecisionTreeLearner(
    #     interface=parse_interface('(Int,Int,Int,Int) -> Bool')
    # )
    # examples = []
    # for i in range(400):
    #     p0 = random.randint(1, 50)
    #     p1 = random.randint(1, 50)
    #     p2 = random.randint(1, 50)
    #     p3 = random.randint(1, 50)
    #     examples.append(([p0, p1, p2, p3], (p0 + p1 - p2 - p3) >= 0))
    
    # learner.train(examples)
    # print(learner.to_smt2())
    
    # learner.evaluate(examples, [metrics.accuracy_score])


    # Logics
    variable_names='length width height noseLength radius tailLength endRadius'.split(' ')
    learner = DecisionTreeLearner(
        interface=parse_interface('(Int,Int,Int,Int,Int,Int,Int) --> Bool', variable_names),
        variable_names=variable_names
    )

    X, y = load_logics_data()
    examples = list(zip(X.tolist(), (y >= 45).tolist()))

    learner.load('logics_100it_68acc')
    learner.evaluate(examples, [metrics.accuracy_score])
    print('---------SMT----------')
    print(learner.to_smt2())