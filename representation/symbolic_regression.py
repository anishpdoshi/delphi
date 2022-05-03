import json
import pickle
import gplearn
import pysmt
from collections import namedtuple

import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from logics_data import load_logics_data
from base_learner import BaseLearner
from utils import parse_interface, generate_variables, clean_examples
from symb_to_smt import parse_symb_program, output_smt, CONST, CONSTNEG

SHOULD_WRAP = True
Node = namedtuple('Node', ['value', 'children'])

def is_eq(a, b):
    return np.array(a == b).astype('int')

def lt(a, b):
    return np.array(a < b).astype('int')

def gt(a, b):
    return np.array(a > b).astype('int')

def ite(a, b, c):
    return np.where(a, b, c)

def mod(a, b):
    a_arr = np.array(a)
    b_arr = np.rint(b).astype('int')
    b_arr = np.where(b_arr == 0, 1, b_arr)
    return a_arr % b_arr

def make_int_caster(func, arity=2):
    if arity == 2:
        def safe(a, b):
            return func(np.array(a).astype('int'), np.array(b).astype('int'))
    elif arity == 1:
        def safe(a):
            return func(np.array(a).astype('int'))
    else:
        raise ValueError('unsupported arity')
    return safe


FUNCTIONS = {
    'bvnot': make_function(make_int_caster(np.invert, arity=1), name='bvnot', arity=1, wrap=SHOULD_WRAP),
    'bvand': make_function(make_int_caster(np.bitwise_and), name='bvand', arity=2, wrap=SHOULD_WRAP),
    'bvor': make_function(make_int_caster(np.bitwise_or), name='bvor', arity=2, wrap=SHOULD_WRAP),
    'bvshl': make_function(make_int_caster(np.left_shift), name='bvshl', arity=2, wrap=SHOULD_WRAP),
    'bvshr': make_function(make_int_caster(np.right_shift), name='bvshr', arity=2, wrap=SHOULD_WRAP),

    'ite': make_function(ite, name='ite', arity=3, wrap=SHOULD_WRAP),
    'is_eq': make_function(is_eq, name='is_eq', arity=2, wrap=SHOULD_WRAP),
    'lt': make_function(lt, name='lt', arity=2, wrap=SHOULD_WRAP),
    'gt': make_function(lt, name='gt', arity=2, wrap=SHOULD_WRAP),

    'mod': make_function(mod, name='mod', arity=2)
}

def get_rewrite_dict(itype):
    if itype.is_bv_type():
        return {
            'bvnot': 'bvneg',
            'bvshr': 'bvashr',
            'lt': 'bvslt',
            'gt': 'bvsgt',
            'add': 'bvadd',
            'sub': 'bvsub',
            'mul': 'bvmul',
            'is_eq': '=',
        }
    else:
        return {
            'lt': '<',
            'gt': '>',
            'add': '+',
            'sub': '-',
            'mul': '*',
            'is_eq': '=',
        }

def constants_for(int_range):
    CONSTANTS = []
    for i in int_range:
        ci = lambda a: np.full_like(a, 2)
        name = f'{CONST if i >= 0 else CONSTNEG}{abs(i)}'
        CONSTANTS.append(make_function(ci, name, 1))
    return CONSTANTS

def functions_for(interface):
    subset = None
    interface_types = [inp_symbol.get_type() for inp_symbol in interface[0]]
    if all([itype.is_bv_type() for itype in interface_types]):
        subset = ['bvnot', 'bvor', 'bvshl', 'add', 'sub', 'bvshr']
        # subset = ['sub', 'add']
    elif any([itype.is_real_type() for itype in interface_types]):
        subset = ['ite', 'is_eq', 'lt', 'gt', 'add', 'sub', 'mul', 'div', 'mod']
    elif all([itype.is_int_type() for itype in interface_types]):
        subset = ['ite', 'is_eq', 'lt', 'gt', 'add', 'sub', 'mul', 'mod'],
    else:
        raise ValueError(f'what functions to use for {interface}')
    
    functions_for_interface = [FUNCTIONS.get(key, key) for key in subset]
    if not any([itype.is_real_type() for itype in interface_types]):
        functions_for_interface.extend(constants_for(range(9, 11)))
    
    return functions_for_interface

class SymbolicLearner(BaseLearner):
    def __init__(self, interface, use_scaling=False):
        super().__init__(interface)
        
        interface_types = [inp_symbol.get_type() for inp_symbol in interface[0]]
        supports_reals = any([itype.is_real_type() for itype in interface_types])
        const_range = (-2.0, 2.0) if supports_reals else None
        self.function_set = tuple(functions_for(interface))

        self.use_scaling = use_scaling
        self.estimator = SymbolicRegressor(
            const_range=const_range,
            population_size=1000,
            generations=5,
            tournament_size=100,
            # parsimony_coefficient='auto',
            init_depth=(2,3),
            function_set=self.function_set,
            metric='mean absolute error',
            n_jobs=-1,
            verbose=True,
            feature_names = generate_variables(len(interface[0]))
        )
        self.standard_scaler = None

    def get_paths(self, location):
        state_path = location + '_symb_regr_state.pkl'
        scaler_path = location + '_scaler.pkl'
        return state_path, scaler_path

    def save(self, location):
        state_path, scaler_path = self.get_paths(location)
        pickle.dump(self.estimator, open(state_path, 'wb+'))
        pickle.dump(self.standard_scaler, open(scaler_path, 'wb+'))

    def load(self, location):
        # Load the symb state and StandardScaler
        state_path, scaler_path = self.get_paths(location)

        self.estimator = pickle.load(open(state_path,'rb'))
        self.standard_scaler = pickle.load(open(scaler_path,'rb'))
        self.estimator.warm_start = True

    def train(self, examples, update_pretrained=False, train_args={}):
        examples = clean_examples(examples)
        X = [example[0] for example in examples]
        y = [example[1] for example in examples]
        X, y = np.array(X), np.array(y)
        
        if self.use_scaling:
            # Refit a standard scaler if not pretrained
            if not self.standard_scaler:
                self.standard_scaler = StandardScaler()
                self.standard_scaler.fit(X)

            X = self.standard_scaler.transform(X)

        self.estimator.fit(X, y)
        
        if update_pretrained:
            self.save(update_pretrained)
    
    def run(self, inputs):
        X = np.array(inputs)
        if self.use_scaling:
            if self.standard_scaler:
                X = self.standard_scaler.transform(X)
            else:
                print('WARN no standard scaler initialized')
        
        return self.estimator._program.execute(X).tolist()

    def to_smt2(self):
        var_names = [f'p{i}' for i in range(len(self.interface[0]))]
        parsed = parse_symb_program(
            str(self.estimator._program),
            self.function_set,
            var_names
        )
        return output_smt(
            parsed,
            get_rewrite_dict(self.interface[1].get_type()),
            var_names,
            self.interface[1].get_type()
        )


if __name__ == '__main__':
    
    interface = parse_interface('((_ BitVec 32),(_ BitVec 32)) --> (_ BitVec 32)')
    learner = SymbolicLearner(interface)
    examples = []
    import random
    for i in range(100):
        p0 = random.randint(1, 50)
        p1 = random.randint(1, 50)
        examples.append(([p0, p1], (p0 << p1) + (p1 << p0)))
    
    learner.train(examples)
    print(learner.to_smt2())
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    learner.evaluate(examples, [mean_absolute_error, mean_squared_error])

    # X, y = load_logics_data()
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # print(f'Rescaled with mean {scaler.mean_} and var {scaler.var_}')
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # est = SymbolicRegressor(
    #     const_range=None,
    #     population_size=1000,
    #     generations=10,
    #     tournament_size=20,
    #     function_set=('add', 'sub', 'mul', EQ_CHECK, ITE, LT, GT) + tuple(CONSTANTS),
    #     metric='mse',
    #     n_jobs=-1,
    #     verbose=True
    # )
    # est.fit(X_train, y_train)
    # y_pred = est.predict(X_test)
    # y_train_pred = est.predict(X_train)
    
    # print()
    # print(f'Train mean squared error: {mean_squared_error(y_train_pred, y_train)}')
    # print(f'Pred mean squared error: {mean_squared_error(y_pred, y_test)}')
    # print()
    # print(f'Train mean absolute error: {mean_absolute_error(y_train_pred, y_train)}')
    # print(f'Pred mean absolute error: {mean_absolute_error(y_pred, y_test)}')
    # print()
    # print(f'Train mape error: {mean_absolute_percentage_error(y_train_pred, y_train)}')
    # print(f'Pred mape error: {mean_absolute_percentage_error(y_pred, y_test)}')
    # print()
    # print(f'Train max error: {max_error(y_train_pred, y_train)}')
    # print(f'Pred max error: {max_error(y_pred, y_test)}')
    # print()
    # print('PROGRAM:---------------------------------')
    # print(est._program)
