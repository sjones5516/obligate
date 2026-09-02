import pytest

from obligate.validator import precondition


def test_precondition_allows_valid_arguments():
    @precondition(lambda value: value >= 0, lambda value: ValueError(value))
    def identity(value):
        return value

    assert identity(3) == 3


def test_precondition_raises_error_for_invalid_arguments():
    @precondition(
        lambda value: value >= 0,
        lambda value: ValueError(f"Expected a non-negative value, got {value}"),
    )
    def identity(value):
        return value

    with pytest.raises(ValueError, match="Expected a non-negative value, got -1"):
        identity(-1)


def test_precondition_forwards_keyword_arguments():
    calls = []

    @precondition(
        lambda *, value: value > 0,
        lambda *, value: ValueError(value),
    )
    def record(*, value):
        calls.append(value)
        return value

    assert record(value=2) == 2
    assert calls == [2]
