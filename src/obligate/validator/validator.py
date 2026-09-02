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
