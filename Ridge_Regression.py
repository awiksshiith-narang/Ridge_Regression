from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt

#Defining a function for improving the datasets for Polynomial regression and regularizing the model by Ridge regularization:
def get_preds_ridge( x1, x2, alpha ):
    model = Pipeline( [ ( 'poly_feats', PolynomialFeatures( degree = 16 ) ), ( 'ridge', Ridge( alpha = alpha ) ) ] )
    model.fit( x1, x2 )
    return model.predict( x1 )

#Gathering data:
m = 100
x_1 = 5 * np.random.rand( m, 1 ) - 2
x_2 = ( (0.7) * x_1 ** 2 ) - ( 2 * x_1 ) + 3 + np.random.randn( m, 1 )

#Training the data using Polynomial Regression, regularizing the model using Ridge Regression and comparing with original output values and for different alphas:
plt.scatter( x_1, x_2 )
alphas = [ 0, 20, 200 ]
cs = [ 'r', 'g', 'b' ]
for alpha, c in zip( alphas, cs ):
    preds = get_preds_ridge( x_1, x_2, alpha )
    plt.plot( sorted( x_1[ :, 0 ] ), preds[ np.argsort( x_1[ :, 0 ] ) ], c, label = 'Alpha : {}'.format( alpha ) )
plt.legend()
plt.show()