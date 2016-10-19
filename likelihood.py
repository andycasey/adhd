from __future__ import division, print_function 

import numpy as np 
import emcee


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

	A, k, vesc = params

	powerlaw = np.zeros_like(v)
	powerlaw[v<vesc] = (k+1.)*(1.-A)*(vesc - v[v<vesc])**k / (vesc - vmin)**(k+1.)

	return powerlaw + A / (10000. - vesc)


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


def mock_data(A,k,vesc,size=2000,uncertainties=False):
	"""
	Draw samples from the model to fit.

	Arguments
	---------

	A: float

		outlier fraction

	k: float

		power law slope 

	vesc: float

		escape velocity

	size: int (=2000)

		number of fake observations to take

	Returns
	-------

	v: array_like[size]

		sample of fake observations

	"""

	v_model = vesc + (200-vesc)*np.random.power(k+1, size= int( (1-A)*size ) )
	v_outlier = np.random.uniform(low=vesc, high=800., size = int( A*size ) )
	v = np.hstack((v_model,v_outlier))
	np.random.shuffle(v)
	if not uncertainties return v
	else:
		v_err = np.clip( np.random.normal(loc=3.,scale=.5,size=size), 0., np.inf )
		v_scattered = np.array([np.float(np.random.normal(loc=v[i], scale=v_err[i], size=1)) for \
					  i in arange(v.shape[0])])
		return v_scattered












