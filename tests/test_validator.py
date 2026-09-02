import pytest

from obligate.validator import postcondition, precondition


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


def test_postcondition_passes_arguments_and_response_to_callbacks():
    calls = []

    @postcondition(
        lambda value, *, response: calls.append((value, response))
        or response == value + 1,
        lambda value, *, response: ValueError((value, response)),
    )
    def increment(value):
        return value + 1

    assert increment(2) == 3
    assert calls == [(2, 3)]


def test_postcondition_raises_dynamic_error_with_response():
    @postcondition(
        lambda *, value, response: response >= value,
        lambda *, value, response: ValueError(
            f"Expected {response} to be at least {value}"
        ),
    )
    def decrement(*, value):
        return value - 1

    with pytest.raises(ValueError, match="Expected 1 to be at least 2"):
        decrement(value=2)
