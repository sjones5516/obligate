# Obligate

Obligate is a Python library for enforcing contracts.

## Installation

Use the package manager pip to install obligate.

```bash
pip install obligate
```

## Usage

```python
import obligate

@obligate.BoolValidator.pre(
    lambda x : x >= 0,
    lambda x: ValueError(f"Expected value greater than 0. Got {x}."),
)
def sqrt(x: float) -> float:
    return x ** 0.5
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.
