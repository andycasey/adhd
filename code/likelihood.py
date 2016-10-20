from __future__ import division, print_function 

import numpy as np 
import emcee
import gus_utils as gu
import corner_plot as cp
import matplotlib.pyplot as plt

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

	if k<0. or k>20.:
		return -np.inf
	if vesc<vmin or vesc>2e4:
		return -np.inf
	if A<0. or A>1.:
		return -np.inf

	powerlaw = np.zeros_like(v)
	powerlaw[v<vesc] = (k+1.)*(1.-A)*(vesc - v[v<vesc])**k / (vesc - vmin)**(k+1.)

	return np.sum(np.log( powerlaw + A/(1000. - vmin) ))


def likelihood_samples(params, v, vmin):
	"""
	Likelihood function given samples of the 
	speed of each star.

	Arguments
	---------

	params: array_like

		A, k, vesc, vmin

	vsamples: array_like[nstars, nsamples]

		samples of the speed for each star

	vmin: float

		the minimum speed considered

	Returns
	-------

	likelihood: array_like

		the average likelihood of each set of samples
	"""

	A, k, vesc = params

	if k<0. or k>20.:
		return -np.inf
	if vesc<vmin or vesc>2e4:
		return -np.inf
	if A<0. or A>1.:
		return -np.inf

	powerlaw = np.zeros_like(v)
	powerlaw[v<vesc] = (k+1.)*(1.-A)*(vesc - v[v<vesc])**k / (vesc - vmin)**(k+1.)

	return np.sum(np.log( np.mean( powerlaw + A/(1000. - vmin), axis=1) ))


def mock_data(A,k,vesc,vmin,size=2000,uncertainties=False,n_samples=1000,error_scale=3.):
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

	v_model = vesc + (vmin-vesc)*np.random.power(k+1, size= int( (1-A)*size ) )
	v_outlier = np.random.uniform(low=vmin, high=1000., size = int( A*size ) )
	v = np.hstack((v_model,v_outlier))
	np.random.shuffle(v)
	if not uncertainties: return v
	else:
		v_err = np.clip( np.random.normal(loc=error_scale,scale=.25*error_scale,size=size), 0.01*error_scale, np.inf )
		v_observed = np.array([np.float(np.random.normal(loc=v[i], scale=v_err[i], size=1)) for \
					  i in np.arange(v.shape[0])])
		v_samples = np.array([np.random.normal(loc=v_observed[i], scale=v_err[i], size=n_samples) for \
					  i in np.arange(v.shape[0])])
		return v_samples

def mock_and_fit(n_steps=500,uncertainties=True,outfile="adhd_mcmc.dat",**kwargs):
	"""
	Generate mock data using randomly generated 
	parameters and then fit it using emcee.

	I'll do this stuff later...
	"""

	np.random.seed(0)
	A = np.random.uniform(low=0.,high=0.1)
	k = np.random.uniform(low=2.3,high=3.7)
	vesc = np.random.uniform(low=400,high=600)
	vmin = 0.6*vesc

	v = mock_data(A,k,vesc,vmin,uncertainties=uncertainties,**kwargs)

	p0 = emcee.utils.sample_ball([A,k,vesc], std=[0.0001,0.1,10.], size=30)
	if not uncertainties: sampler = emcee.EnsembleSampler(30,3,likelihood,args=[v,vmin],threads=4)
	else: sampler = emcee.EnsembleSampler(30,3,likelihood_samples,args=[v,vmin],threads=4)

	print("Running MCMC...")
	gu.write_to_file(sampler,outfile,p0,Nsteps=n_steps)
	print("...done!")

	return [A,k,vesc], v

def main():
	print("10 samples per star...")
	true,v_10 = mock_and_fit(uncertainties=True,outfile="adhd_mcmc_10.dat",n_samples=10)
	np.save("true",true)
	np.save("v_10",v_10)
	print("100 samples per star...")
	_,v_100 = mock_and_fit(uncertainties=True,outfile="adhd_mcmc_100.dat",n_samples=100)
	np.save("v_100",v_100)
	print("1000 samples per star...")
	_,v_1000 = mock_and_fit(uncertainties=True,outfile="adhd_mcmc_1000.dat",n_samples=1000)
	np.save("v_10000",v_1000)

if __name__=="__main__":
	main()

















