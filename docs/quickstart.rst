Quickstart
==========

Theanets is a tool for defining, computing and optimizing complex, parametric
functions that map multiple input variables to multiple output variables. The
primary goals of the package are to (1) be easy to use and (2) have readable,
well-tested code.

This page provides a quick overview of the package that should get you up and
running with a few simple examples. Once you understand the basic workflow, you
should be able to extend the examples to your own datasets!

Basic overview
--------------

Theanets provides several basic classes that are useful for automating various
tasks in machine learning. These include:

* :class:`theanets.Network` -- the base class for all neural networks
* :class:`theanets.Trainer` -- the base class for all network trainers
* :class:`theanets.SequenceDataset` -- a class for managing training and
  validation data

Most of the functionality of the package lives in the :class:`theanets.Network`
class. This class defines the topology of a neural network, meaning the number
and configuration of the units of a feedforward neural network.

Creating a network
------------------

TODO!
