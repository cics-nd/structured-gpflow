# structured-gpflow #

Implements a variety of Gaussian process models exploiting the "structured" 
assumption that one has inputs that ae formed as a Cartesian product as well 
as a kernel that is separable so that one may decompose the associated kernel 
matrices as Kronecker products for a representation that is computationally 
efficient in terms of both time and memory.  
The models are built on top of [GPflow](https://github.com/GPflow/GPflow), and 
the computational backend is [TensorFlow](https://www.tensorflow.org).

## Installation ##

First, install GPflow.
(Note: this repo is designed to work with [this fork](https://github.com/sdatkinson/GPflow).)

Then, simply `python setup.py install` as usual.

## Models ##

* SGPR: Structured GP for regression
* SGPLVM: Structured Bayesian Gaussian process latent variable model
* SWGP: Structured Bayesian warped Gaussian processes

See [[Atkinson and Zabaras, 2018]](https://arxiv.org/abs/1805.08665) for more 
information.

## Questions ##

Contact [Steven Atkinson](mailto:steven@atkinson.mn) or 
[Nicholas Zabaras](mailto:nzabaras@nd.edu) with questions or comments.
