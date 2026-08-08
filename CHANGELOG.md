# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

TBC...

## [1.0.0] - 2026-08-08

### Added

- Full documentation site (Jupyter Book): pages covering `Basis1D`,
  `RealFunction` and `ComplexFunction`, plus an API reference generated from
  the Python docstrings, published via ReadTheDocs.
- `Function::Inverse`, the coefficient-domain inverse of a Chebyshev series.
- Docstrings for every Python binding, with keyword-argument support
  (`nb::arg`) on all methods that take arguments.
- Dual licensing: MIT for the code and example notebooks, CC BY 4.0 for the
  documentation.
- `nbstripout` pre-commit hook to keep notebook metadata and outputs
  consistent in git.
- Substantial new test coverage, including the projection matrix, basis
  recombinations, monomial conversion, the power operator, constant
  functions, and `Function::Inverse`.

### Changed

- Replaced `pybind11` with `nanobind` for the Python bindings.
- Improved `Function::Primitive` and `Function::Derivative`.
- Switched Python formatting from `black` to `ruff format`.

### Fixed

- Numerous correctness bugs: root and extrema finding (colleague-matrix
  balancing and eigenvalue extraction), the monomial-conversion matrix and
  `Function::Monomials`, the projection matrix, `Function::Inverse` corner
  cases, and `Function` constructor edge cases.
- Missing `#include` in `function.hpp` that made the header order-dependent
  to compile correctly.

## [0.3.0] - 2023-11-06

_Covers tags v0.3.0 through v0.3.11 (2023-11-06 to 2026-02-27)._

### Added

- Neumann recombination matrix (`Basis::NeumannMatrix`), alongside the
  existing Dirichlet recombination.
- `Basis1D::Order`, the projection matrix, and the monomial-conversion
  matrix.
- `Function::MultiplyCoef` and a generalised `Function::ProductMatrix`.
- An additional `Function` constructor overload.
- Inline C++ documentation.
- Automated PyPI release workflow; package version is now derived from the
  git tag.

### Fixed

- Matrix-initialisation bug in the Dirichlet recombination matrix.

## [0.2.0] - 2023-10-16

### Added

- Initial `Basis1D`: Chebyshev points of the first and second kind,
  evaluation and derivatives of the Chebyshev polynomials, the
  differentiation matrix, and the Dirichlet recombination matrix.
- Initial `RealFunction`/`ComplexFunction`: construction from a Python
  callable via FFT, derivative and primitive, addition/subtraction/
  multiplication, integer powers, integrals, norms, roots, extrema, and
  real/imaginary/conjugate parts.
- Constant-function constructors.
- Python bindings via `pybind11`, packaged with `cibuildwheel`.
