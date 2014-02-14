=====
Setup
=====

The easiest way to install the package is to use ``pip``::

    pip install theanets

This will install the most recent released version of the source code.

Development
-----------

If you want to install the latest source, or if you decide you'd like to help
out with development of the :mod:`theanets` source code, you can find it at
https://github.com/lmjohns3/theano-nets. The source code also includes complete
code for the examples described in the documentation.

Dependencies
------------

The :mod:`theanets` package relies on several fantastic libraries to get things
done. All of these should be installed automatically for you if you install with
``pip``. The most important dependency, without which :mod:`theanets` would
simply not exist, is :mod:`theano` (which, in turn, depends on :mod:`numpy`).

We also depend on :mod:`climate`, a small library of command-line utilities.
This module depends in turn on :mod:`plac`.
