import os
import json
import glob
import shutil
import subprocess
import time

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_blue(text):
    print(f'{bcolors.OKBLUE}{text}{bcolors.ENDC}')

def main():
    """
    const_range=const_range,
    population_size=1000,
    generations=6,
    tournament_size=100,
    # parsimony_coefficient='auto',
    init_depth=(2,4),
    subset = ['bvand','bvnot', 'bvshl', 'bvshr', 'sub']

    """
    if os.path.exists('results'):
        print('results exists already')
        return

    os.mkdir('results')
    shutil.copy(
        os.path.abspath('../../representation/symbolic_regression.py'),
        os.path.abspath('results/symbolic_regression_content.py')
    )
    data = []
    for smtfile in glob.glob("*.smt2"):
        print_blue(f'-----RUNNING: {smtfile}')
        spec = [
            (['--symbolic-bitblast'], 'no oracle repr'),
            (['--symbolic-bitblast', '--oracle-repr', 'symb_regr', '--oracle-repr-freq', '5'], 'symb_regr representation'),
            (['--symbolic-bitblast', '--oracle-repr', 'sygus'], 'sygus representation')
        ]
        data = { 'file': smtfile }
        for argslist, key in spec:
            print_blue(f'RUNNING {key}')
            t_zero = time.perf_counter()
            
            result = subprocess.run(["delphi"] + argslist + [smtfile], capture_output=True, text=True)
            t_total = time.perf_counter() - t_zero
            print_blue(f'Finished in {t_total} sec')
            data[key] = t_total
            data[f'{key}_out'] = result.stdout

        with open('results/infos.json', 'w+') as f:
            json.dump(data, f)

if __name__ == '__main__':
    main()