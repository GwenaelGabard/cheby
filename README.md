# cheby

This package provides:
* Basis of Chebyshev polynomials of the first kind
* Functions represented as Chebyshev series

It is intended to be used for the resolution of differential equations using spectral methods.
It is primarily a C++ library with a Python wrapper using [pybind11](https://pybind11.readthedocs.io/en/stable/).
It also relies on [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) for linear algebra.

## Installation

The simplest way is to use `pip`:

```bash
pip install cheby
```

The package can be installed directly from source:

```bash
git clone https://github.com/GwenaelGabard/cheby
cd cheby
git submodule update --init --recursive
pip install .
```
This will require a C++ compiler and `cmake`.

## Usage

The Python class `Basis1D` provides the following features:
* Evaluation of Chebyshev polynomials of the first kind and their derivatives
* Chebyshev points of the first and second kinds
* Differentiation matrix
* Matrix for Dirichlet recombination

The Python classes `RealFunction` and `ComplexFunction` provide representations of univariate functions as Chebyshev series.
THey provide the following features:
* Construction of the Chebyshev representation based on a Python function
* Evaluation of the function and its derivatives
* Addition, subtraction and multiplication
* Primitive and integrals (over the whole domain or over a subsegment)
* Roots and extrema
* Integer powers

See the Jupyter notebooks in the `examples` folder for examples of usage.

## Unit tests

Unit tests are written using [pytest](https://docs.pytest.org/en/latest/).
They can be run using

```bash
pytest tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
