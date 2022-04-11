import pandas as pd

# pip install gplearn
import gplearn
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import max_error, mean_squared_error


if __name__ == '__main__':
    df = pd.read_csv('logics_samples.csv')
    X = df.iloc[:,1:8].values
    y = df['dragForce'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(f'Rescaled with mean {scaler.mean_} and var {scaler.var_}')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    est = SymbolicRegressor(
        const_range=(-2.0, 2.0),
        population_size=2000,
        generations=20,
        tournament_size=20,
        function_set=('add', 'sub', 'mul', 'div', 'max', 'min', 'abs', 'neg'),
        metric='mse',
        n_jobs=-1
    )
    est.fit(X_train, y_train)
    y_pred = est.predict(X_test)
    y_train_pred = est.predict(X_train)
    
    print()
    print(f'Train mean squared error: {mean_squared_error(y_train_pred, y_train)}')
    print(f'Pred mean squared error: {mean_squared_error(y_pred, y_test)}')
    print()
    print(f'Train max error: {max_error(y_train_pred, y_train)}')
    print(f'Pred max error: {max_error(y_pred, y_test)}')
    print()
    print('PROGRAM:---------------------------------')
    print(est._program)
