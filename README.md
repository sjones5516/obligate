# Obligate

Obligate is a Python library for enforcing contracts.
See the documentation at <https://obligate.readthedocs.io/en/latest/>

## Installation

Use the package manager pip to install obligate.

```bash
pip install obligate
```

## Usage

```python
from obligate.contracts import precondition

@precondition(
    lambda x: x >= 0,
    lambda x: ValueError(f"Expected value greater than 0. Got {x}."),
)
def sqrt(x: float) -> float:
    return x ** 0.5
```

## Development

Install the development and documentation dependencies with Poetry:

```bash
poetry install --with dev,docs
```

Run the unit tests, including doctests:

```bash
poetry run pytest
```

Build the HTML documentation:

```bash
cd docs
poetry run make html
```

The generated documentation is written to `docs/build/html/`. On Windows,
run `poetry run make.bat html` from the `docs` directory instead.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
