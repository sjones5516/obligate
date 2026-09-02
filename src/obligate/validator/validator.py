from typing import Callable
import functools


def precondition[**P, R](
    condition_callback: Callable[P, bool], error_callback: Callable[P, Exception]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Guard a function with a precondition, raising a dynamically built exception.

    ``error`` is called with the *same* arguments that were passed to the
    wrapped function, letting the exception message embed the actual
    values that failed validation.

    Examples:
        >>> from obligate.validator import precondition
        >>> @precondition(
        ...     lambda value: value >= 0,
        ...     lambda value: ValueError(f"Expected a non-negative value, got {value}"),
        ... )
        ... def square_root(value: float) -> float:
        ...     return value ** 0.5
        >>> square_root(9)
        3.0
        >>> square_root(-1)
        Traceback (most recent call last):
        ...
        ValueError: Expected a non-negative value, got -1

    Args:
        condition: Returns ``True`` if the arguments are valid.
        error: Called as ``error(*args, **kwargs)`` when ``condition``
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not condition_callback(*args, **kwargs):
                raise error_callback(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def postcondition[**P, R](
    condition_callback: Callable[..., bool], error_callback: Callable[..., Exception]
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Guard a function with a postcondition, raising a dynamically built exception.

    Both callbacks receive the arguments passed to the wrapped function and
    the returned value as the ``response`` keyword argument.

    Examples:
        >>> from obligate.validator import postcondition
        >>> @postcondition(
        ...     lambda value, *, response: response >= value,
        ...     lambda value, *, response: ValueError(
        ...         f"Expected {response} to be at least {value}"
        ...     ),
        ... )
        ... def increment(value: int) -> int:
        ...     return value + 1
        >>> increment(9)
        10
        >>> increment(-1)
        0

        A callback can reject the returned value:

        >>> @postcondition(
        ...     lambda value, *, response: response >= 0,
        ...     lambda value, *, response: ValueError(f"Got {response}"),
        ... )
        ... def invalid(value: int) -> int:
        ...     return -value
        >>> invalid(1)
        Traceback (most recent call last):
        ...
        ValueError: Got -1

    Args:
        condition_callback: Returns ``True`` when the returned value is valid.
        error_callback: Called with the original arguments and ``response``
        when ``condition_callback`` returns ``False``.
    """

    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            response = func(*args, **kwargs)
            if not condition_callback(*args, response=response, **kwargs):
                raise error_callback(*args, response=response, **kwargs)
            return response

        return wrapper

    return decorator
