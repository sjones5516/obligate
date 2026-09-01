import functools
from typing import Callable

"""Set of static methods to validate pre and post-conditions based off functions with boolean return types."""


class BoolValidator:
    @staticmethod
    def pre[**P, R](
        condition: Callable[P, bool],
        error: Callable[P, Exception],
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Guard a function with a precondition, raising a dynamically built exception.

        ``error`` is called with the *same* arguments that were passed to the
        wrapped function, letting the exception message embed the actual
        values that failed validation.

        Args:
            condition: Returns ``True`` if the arguments are valid.
            error: Called as ``error(*args, **kwargs)`` when ``condition``
            fails; must return an ``Exception`` instance to raise.
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                if not condition(*args, **kwargs):
                    raise error(*args, **kwargs)
                return func(*args, **kwargs)

            return wrapper

        return decorator

    @staticmethod
    def post[**P, R](
        condition: Callable[[R], bool],
        error: Callable[[R], Exception],
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Guard a function with a postcondition, raising a dynamically built exception.

        ``error`` is called with the wrapped function's return value, letting
        the exception message embed the actual result that failed validation.

        Args:
            condition: Returns ``True`` if the result is valid.
            error: Called as ``error(result)`` when ``condition`` fails; must
            return an ``Exception`` instance to raise.
        """

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(func)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                result = func(*args, **kwargs)
                if not condition(result):
                    raise error(result)
                return result

            return wrapper

        return decorator
