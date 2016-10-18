from __future__ import division, print_function 

import numpy as np 


def likelihood(params, v, vmin):
	"""
	Likelihood function for the speed.

	Arguments
	---------

	params: array_like

		A, k, vesc

	v: array_like

		speed of a star 

	vmin: float

		the minimum speed considered

	Returns
	-------

	likelihood: array_like

		likelihood of the speed under the model

	"""

	A, k, vesc, vmin = params 

	return A*(vesc - v)**k / ( (k+1.) * (vesc - vmin)**(k+1.) ) + \
			(1. - A) / 10000.


def likelihood_samples(params, vsamples, vmin):
	"""
	Likelihood function given samples of the 
	speed of each star.

	Arguments
	---------

	params: array_like

		A, k, vesc, vmin

	vsamples: array_like[nsamples, nstars]

		samples of the speed for each star

	vmin: float

		the minimum speed considered

	Returns
	-------

	likelihood: array_like

		the average likelihood of each set of samples
	"""

	return np.mean( likelihood(params,vsamples), axis=0 )







