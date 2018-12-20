import csv
import numpy as np 
from sklearn.svm import SVR 
import matplotlib.pyplot as plt 

days = []
prices = []

def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            days.append(int(row[8]))
            prices.append(float(row[3]))
    return

def predict_price(days, prices, x):
    days = np.reshape(days,(len(days), 1))       # Converting to matrix of n X 1

    svr_lin = SVR(kernel= 'linear', C=1e3)
    svr_poly = SVR(kernel= 'poly', C=1e3, degree = 2)
    svr_rbf = SVR(kernel= 'rbf', C=1e3, gamma=0.1)
    svr_lin.fit(days, prices)
    svr_poly.fit(days, prices)
    svr_rbf.fit(days, prices)

    plt.scatter(days, prices, color='black', label='Data')
    plt.plot(days, svr_rbf.predict(days), color='red', label='RBF model')
    plt.plot(days, svr_lin.predict(days), color='green', label='Linear model')
    #plt.plot(days, svr_poly.predict(days), color='blue', label='Polynomial model')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('aapl_2017.csv')        # Calling get_data method by passing the csv file to it
#print("Days- ", days)
#print("Prices- ", prices)

predicted_price = predict_price(days, prices, 29)

print(predicted_price)