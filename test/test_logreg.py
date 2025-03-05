"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
# (you will probably need to import more things here)

def test_prediction():
	log_model_test = logreg.LogisticRegressor(num_feats=4, learning_rate=0.0001, tol=0.01, max_iter=10000, batch_size=100)
	log_model_test.W = np.array([10,10,10,10,10])

	#given the weights of the features W and the value of the features for x_test_1, make a prediction for the value of y
	x_test_1 = np.array([1,1,1,1,1])
	prediction_true = log_model_test.make_prediction(x_test_1)

	#because the dot product is a large positive number, the prediction should approach 1 for the sigmoid function
	assert prediction_true == 1

	# given the weights of the features W are all 0 and the value of the features for x_test_equal, make a prediction for the value of y
	log_model_test.W = np.array([0, 0, 0, 0, 0])
	x_test_equal = np.array([1, 0, 1, 0, 1])
	prediction_equal = log_model_test.make_prediction(x_test_equal)


	#since the weights are all 0, there is equal percent chance of the x_test_equal being either 0 or 1. none of the features given influence its prediction
	assert prediction_equal == 0.5

def test_loss_function():
	log_model_test = logreg.LogisticRegressor(num_feats=4, learning_rate=0.001, tol=0.01, max_iter=1000,
											  batch_size=10)
	y_true = np.array([0,1,1,0])
	y_pred = np.array([0.203, 0.674,0.880,0.02])

	#calculate loss manually
	loss_check = log_loss(y_true, y_pred)
	loss_test = log_model_test.loss_function(y_true, y_pred)
	assert np.isclose(loss_check, loss_test, atol = 1e-6)

def test_gradient():
	log_model_test = logreg.LogisticRegressor(num_feats=3, learning_rate=0.001, tol=0.01, max_iter=5,
										 batch_size=10)
	y_true = np.array([0,1,1,1])
	x_test = np.array([[0.123, 0.998, 0.213, 1],
					   [0.889, 0.452, 0.889, 1],
					   [0.923, 0.111, 0.644, 1],
					   [0.655, 0.621, 0.412, 1]])

	test_grad = log_model_test.calculate_gradient(y_true, x_test)

	y_pred = log_model_test.make_prediction(x_test)
	check_grad = np.dot(x_test.T, (y_pred-y_true)) / len(y_true)
	assert np.array_equal(test_grad, check_grad)


def test_training():
	# For testing purposes, once you've added your code.
	# CAUTION: hyperparameters have not been optimized.

	X_train, X_val, y_train, y_val = utils.loadDataset(
		features=['Penicillin V Potassium 500 MG',
        'Computed tomography of chest and abdomen',
        'Plain chest X-ray (procedure)',
        'Low Density Lipoprotein Cholesterol',
        'Creatinine','GENDER','Penicillin V Potassium 500 MG',
		'Documentation of current medications',
		'AGE_DIAGNOSIS'
		],
		split_percent=0.8,
		split_seed=42
	)

	sc = StandardScaler()
	X_train = sc.fit_transform(X_train)
	X_val = sc.transform(X_val)
	num_features = X_train.shape[1]

	log_model = logreg.LogisticRegressor(num_feats= num_features, learning_rate=0.0001, tol=0.00001, max_iter=1000, batch_size=50)

	x_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])

	#calculate the prediction and the loss from the predictions for the test set before training
	y_pred_pretrain = log_model.make_prediction(x_val)
	loss_pretrain = log_model.loss_function(y_val, y_pred_pretrain)

	#train model
	log_model.train_model(X_train, y_train, X_val, y_val)

	#calculate prediction and loss from predictions for test set after training
	y_pred_train = log_model.make_prediction(x_val)
	loss_train = log_model.loss_function(y_val, y_pred_train)

	#print(loss_pretrain)
	#print(loss_train)
	#log_model.plot_loss_history()

	#the trained model should show less loss on average than the untrained model
	assert loss_pretrain > loss_train





