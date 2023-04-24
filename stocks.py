import yfinance as yf
import time
from datetime import date,timedelta
from dask_ml.wrappers import Incremental
import pickle
from sklearn.linear_model import SGDRegressor
import dask.dataframe as dd



df = dd.from_pandas(yf.download("BTC-USD", interval="5m", start=date.today() - timedelta(59)),npartitions=10)
X = df[['Low']].values
y = df['High']
model = SGDRegressor()
model = Incremental(model, scoring='accuracy')
model.partial_fit(X, y)
pickle.dump(model, open('model.sav', 'wb'))

while True:
    model = pickle.load(open('model.sav','rb'))
    df = dd.from_pandas(yf.download("BTC-USD", interval="5m",start = date.today()).tail(1),npartitions=1)
    X = df[['Low']].values
    y= df['High']
    model.partial_fit(X, y)
    pickle.dump(model,open('model.sav','wb'))
    time.sleep(300)