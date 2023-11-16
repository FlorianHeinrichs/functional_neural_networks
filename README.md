# Functional Neural Networks
Implementation of Functional Neural Networks in TensorFlow based on the paper:

    Heinrichs, F., Heim, M. & Weber, C. (2023). Functional Neural Networks:
    Shift invariant models for functional data with applications to EEG
    classification. Proceedings of the 40th International Conference on Machine
    Learning, 12866-12881.

[Link to paper](https://proceedings.mlr.press/v202/heinrichs23a.html)


## Modules

- The functional dense layer is defined in *dense.py*
- The functional convolutional layer is defined in *convolution.py*
- A function to create basis functions is provided in *basis.py* (currently only
 Legendre polynomials and the Fourier basis are supported)
- An example is provided in **example.py**

## Dependencies

The code was developed under NumPy 1.22.4, SciPy 1.8.1, TensorFlow 2.9.1, 
Python 3.8 but is expected to be compatible with most other versions. 