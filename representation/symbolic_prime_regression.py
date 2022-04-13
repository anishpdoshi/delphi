import z3
import random
import textwrap
import subprocess

import numpy as np
import pandas as pd

# pip install gplearn
import gplearn
from gplearn.functions import make_function
from gplearn.genetic import SymbolicClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def is_prime(n):
    if n & 1 == 0:
        return False
    d= 3
    while d * d <= n:
        if n % d == 0:
            return False
        d= d + 2
    return True

PRIMES = []
COMPOSITES = []
for i in range(2, 10000):
    (PRIMES if is_prime(i) else COMPOSITES).append(i)

def create_training_data(num_samples, low=1, high=10000):
    X = []
    y = []
    
    X.extend([[num] for num in random.sample(PRIMES, num_samples // 2)])
    y.extend([1 for i in range(num_samples // 2)])

    X.extend([[num] for num in random.sample(COMPOSITES, num_samples // 2)])
    y.extend([0 for i in range(num_samples // 2)])

    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    
    return np.array(X), np.array(y)

def mod(a, b):
    a_arr = np.array(a)
    b_arr = np.rint(b).astype('int')
    b_arr = np.where(b_arr == 0, 1, b_arr)
    return a_arr % b_arr

def eq_check(a, b):
    return np.array(a == b).astype('int')

def lt(a, b):
    return np.array(a < b).astype('int')

def gt(a, b):
    return np.array(a > b).astype('int')

def ite(a, b, c):
    return np.where(a, b, c)

def c0(a): return np.zeros_like(a)
def c1(a): return np.ones_like(a)
def c2(a): return np.full_like(a, 2)
def c3(a): return np.full_like(a, 3)
def c4(a): return np.full_like(a, 4)
def c5(a): return np.full_like(a, 5)


MOD = make_function(mod, name='mod', arity=2, wrap=False)
EQ_CHECK = make_function(eq_check, name='eq_check', arity=2, wrap=False)
ITE = make_function(ite, name='ite', arity=3, wrap=False)

LT = make_function(lt, name='<', arity=2, wrap=False)
GT = make_function(gt, name='>', arity=2, wrap=False)

C0 = make_function(c0, 'CONST_0', 1, wrap=False)
C1 = make_function(c1, 'CONST_1', 1, wrap=False)
C2 = make_function(c2, 'CONST_2', 1, wrap=False)
C3 = make_function(c3, 'CONST_3', 1, wrap=False)
C4 = make_function(c4, 'CONST_4', 1, wrap=False)
C5 = make_function(c5, 'CONST_5', 1, wrap=False)

def sygus_add_to_smt_recur(elements):
    if len(elements) == 1:
        return elements[0]
    return f'(+ {elements[0]} {sygus_add_to_smt_recur(elements[1:])})'

def sygus_train(X_train, y_train, optim=True):
    print('\n\nNOW TRYING SyGuS SYNTH')
    
    sygus_fmted = """
    ;; The background theory is Ints
    (set-logic ALL)

    ;; Name and signature of the function to be synthesized
    (synth-fun is_prime ((x Int)) Bool

        ;; Declare the non-terminals that would be used in the grammar
        ((B Bool) (I Int))

        ;; Define the grammar for allowed implementations of max2
        ((B Bool (true false (= I I)))
        (I Int (x 0 1 2 3 4 5 (mod I I) (ite B I I))))
    )

    ;; Define the semantic constraints on the function
    (declare-var x Int)"""

    if optim:
        scores = []
        for i in range(len(y_train)):
            num = X_train[i][0]
            label = y_train[i]
            
            if label == 1:
                scores.append(f'(ite (is_prime {num}) 1 0)')
            else:
                scores.append(f'(ite (is_prime {num}) 0 1)')
        print(scores)
        objective_value = sygus_add_to_smt_recur(scores)
        print(objective_value)
        
        sygus_fmted += '\n\t' + '(declare-var score Int)'
        sygus_fmted += '\n\t' + f'(constraint (= score {objective_value}))'
        sygus_fmted += '\n\t' + '(optimize-synth ((! score :max)))'
        print(textwrap.dedent(sygus_fmted))
    else:
        for i in range(len(y_train)):
            sygus_fmted += '\n\t' + f'(constraint (= (is_prime {X_train[i][0]}) {"true" if y_train[i] == 1 else "false"}))'
        sygus_fmted += '(check-synth)'

    with open('prime_synth_test.sl', 'w+') as f:
        f.write(textwrap.dedent(sygus_fmted))

    output = subprocess.run(['cvc5', 'prime_synth_test.sl'], check=True)
    print(output)
    return output

if __name__ == '__main__':
    X, y = create_training_data(500)
    print(X[:10])
    print(y[:10])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    sygus_train(X_train, y_train)
    # est = SymbolicClassifier(
    #     const_range=None,
    #     population_size=2000,
    #     generations=40,
    #     tournament_size=20,
    #     function_set=(MOD, EQ_CHECK, ITE,C0, C1, C2, C3, C4, C5),
    #     metric='log loss',
    #     n_jobs=-1
    # )
    # est.fit(X_train, y_train)
    # y_pred = est.predict(X_test)
    # y_train_pred = est.predict(X_train)
    
    # print()
    # print(f'Train accuracy_score: {accuracy_score(y_train, y_train_pred)}')
    # print(f'Test accuracy_score: {accuracy_score(y_test, y_pred)}')
    # print()
    # print(f'Train precision_score: {precision_score(y_train, y_train_pred)}')
    # print(f'Test precision_score: {precision_score(y_test, y_pred)}')
    # print()
    # print(f'Train recall_score: {recall_score(y_train, y_train_pred)}')
    # print(f'Test recall_score: {recall_score(y_test, y_pred)}')
    # print()
    # print('PROGRAM:---------------------------------')
    # print(est._program)

    
