import pandas as pd

# pip install gplearn
import gplearn
from gplearn.genetic import SymbolicRegressor
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    df = pd.read_csv('logics_samples.csv')
    X = df.iloc[:,1:8].values
    p = StandardScaler()
    X_scaled = p.fit_transform(X)


    # y = df['dragForce'].values

    # est = SymbolicRegressor()
    # print(est.fit(X_scaled, y))
    # y_pred = est.predict(X_scaled)

    # print(est._program)