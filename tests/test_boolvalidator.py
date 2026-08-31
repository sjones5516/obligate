import pytest

from src.obligate import BoolValidator


@pytest.fixture
def precondition_sqrt():
    def sqrt(x: float) -> float:
        return x**0.5

    return BoolValidator.pre(
        lambda x: x >= 0,
        lambda x: ValueError(f"Expected value greater than 0. Got {x}."),
    )(sqrt)


@pytest.fixture
def postcondition_buggy_absolute():
    def buggy_absolute(x: int) -> int:
        return x

    return BoolValidator.post(
        lambda r: r >= 0,  # type: ignore
        lambda r: ValueError(f"Expected value greater than 0. Got {r}"),
    )(buggy_absolute)


class TestBoolValidator:
    def test_precondition_passes(self, precondition_sqrt):
        precondition_sqrt(1.0)

    def test_precondition_fails(self, precondition_sqrt):
        with pytest.raises(ValueError) as exc:
            precondition_sqrt(-1.0)

            exc.match("Expected value greater than 0.")

    def test_postcondition_passes(self, postcondition_buggy_absolute):
        postcondition_buggy_absolute(1)

    def test_postcondition_fails(self, postcondition_buggy_absolute):
        with pytest.raises(ValueError) as exc:
            postcondition_buggy_absolute(-1.0)

            exc.match("Expected value greater than 0.")
