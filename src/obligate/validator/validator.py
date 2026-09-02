from typing import Callable
import copy
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


def invariant[C](
    condition_callback: Callable[[C], bool],
    error_callback: Callable[[C], Exception],
    *,
    rollback: bool = False,
    deep_copy: bool = False,
) -> Callable[[type[C]], type[C]]:
    """Guard a class with an invariant enforced on every attribute mutation.

    Built on top of :func:`postcondition`: every wrapped call point follows
    the same call-then-check-else-raise shape, just adapted to check
    ``self`` (or the returned instance, for factories) instead of a
    function's return value directly.

    Checked after every attribute assignment (so external code mutating the
    object directly is caught immediately), after every instance method
    call, and on the returned instance of any classmethod or staticmethod
    used as a factory.

    If ``rollback`` is ``True``, a violation restores the instance's
    ``__dict__`` to its state immediately before the offending attribute
    assignment or method call, instead of leaving the object mutated. This
    only applies to mutations of an *existing* instance (attribute
    assignment and instance methods) — classmethod/staticmethod factories
    are never rolled back, since the object doesn't exist yet when the
    check fails there.

    By default the snapshot is a shallow copy of ``__dict__``, so mutable
    attributes (lists, dicts, etc.) mutated *in place* rather than
    reassigned are not restored -- only reassignments of ``self.x = ...``
    are undone. Pass ``deep_copy=True`` to snapshot with
    :func:`copy.deepcopy` instead, which also undoes in-place mutations of
    nested mutable state, at the cost of a deep copy on every guarded call
    (only relevant when ``rollback=True``; ignored otherwise).

    Examples:
        >>> from obligate.validator import invariant
        >>> @invariant(
        ...     lambda self: self.balance >= 0,
        ...     lambda self: ValueError(f"Balance went negative: {self.balance}"),
        ... )
        ... class Account:
        ...     def __init__(self, balance: float) -> None:
        ...         self.balance = balance
        ...     def withdraw(self, amount: float) -> None:
        ...         self.balance -= amount
        ...     @classmethod
        ...     def overdrawn(cls) -> "Account":
        ...         return cls(-1)
        ...     @staticmethod
        ...     def empty() -> "Account":
        ...         return Account(-1)
        >>> a = Account(10)
        >>> a.withdraw(5)
        >>> a.balance = -1  # external mutation, caught immediately
        Traceback (most recent call last):
        ...
        ValueError: Balance went negative: -1
        >>> a.balance  # left mutated, since rollback defaults to False
        -1
        >>> Account.overdrawn()
        Traceback (most recent call last):
        ...
        ValueError: Balance went negative: -1
        >>> Account.empty()
        Traceback (most recent call last):
        ...
        ValueError: Balance went negative: -1

        With ``rollback=True``, a violation restores the prior state:

        >>> @invariant(
        ...     lambda self: self.balance >= 0,
        ...     lambda self: ValueError(f"Balance went negative: {self.balance}"),
        ...     rollback=True,
        ... )
        ... class SafeAccount:
        ...     def __init__(self, balance: float) -> None:
        ...         self.balance = balance
        ...     def withdraw(self, amount: float) -> None:
        ...         self.balance -= amount
        >>> s = SafeAccount(10)
        >>> s.withdraw(5)
        >>> s.balance
        5
        >>> s.withdraw(100)
        Traceback (most recent call last):
        ...
        ValueError: Balance went negative: -95
        >>> s.balance  # restored, not left at -95
        5
        >>> s.balance = -1  # external mutation is rolled back too
        Traceback (most recent call last):
        ...
        ValueError: Balance went negative: -1
        >>> s.balance
        5

        A shallow ``rollback`` doesn't undo in-place mutation of nested
        state, since the same list object is shared between the snapshot
        and ``self``:

        >>> @invariant(
        ...     lambda self: len(self.items) <= 2,
        ...     lambda self: ValueError(f"Too many items: {self.items}"),
        ...     rollback=True,
        ... )
        ... class ShallowCart:
        ...     def __init__(self) -> None:
        ...         self.items = []
        ...     def add(self, item: str) -> None:
        ...         self.items.append(item)
        >>> c = ShallowCart()
        >>> c.add("a")
        >>> c.add("b")
        >>> c.add("c")
        Traceback (most recent call last):
        ...
        ValueError: Too many items: ['a', 'b', 'c']
        >>> c.items  # not restored -- same list object, mutated in place
        ['a', 'b', 'c']

        ``deep_copy=True`` fixes that, since the snapshot holds an
        independent copy of ``items``:

        >>> @invariant(
        ...     lambda self: len(self.items) <= 2,
        ...     lambda self: ValueError(f"Too many items: {self.items}"),
        ...     rollback=True,
        ...     deep_copy=True,
        ... )
        ... class DeepCart:
        ...     def __init__(self) -> None:
        ...         self.items = []
        ...     def add(self, item: str) -> None:
        ...         self.items.append(item)
        >>> d = DeepCart()
        >>> d.add("a")
        >>> d.add("b")
        >>> d.add("c")
        Traceback (most recent call last):
        ...
        ValueError: Too many items: ['a', 'b', 'c']
        >>> d.items  # restored, since the snapshot was an independent copy
        ['a', 'b']

    Args:
        condition_callback: Returns ``True`` when ``self`` satisfies the invariant.
        error_callback: Called with ``self`` when ``condition_callback`` returns ``False``.
        rollback: If ``True``, restore ``self`` to its state before the
        offending mutation instead of leaving it invalid.
        deep_copy: If ``True`` (and ``rollback`` is also ``True``), snapshot
        with :func:`copy.deepcopy` instead of a shallow copy, so
        in-place mutations of nested mutable attributes are undone too.
    """

    def snapshot(self: C) -> dict:
        if deep_copy:
            return copy.deepcopy(vars(self))
        return copy.copy(vars(self))

    def restore(self: C, state: dict) -> None:
        vars(self).clear()
        vars(self).update(state)

    def wrap_method(method: Callable) -> Callable:
        if not rollback:
            return postcondition(
                lambda self, *a, response, **kw: condition_callback(self),
                lambda self, *a, response, **kw: error_callback(self),
            )(method)

        @functools.wraps(method)
        def wrapper(self: C, *args, **kwargs):
            state = snapshot(self)
            result = method(self, *args, **kwargs)
            if not condition_callback(self):
                error = error_callback(self)
                restore(self, state)
                raise error
            return result

        return wrapper

    def wrap_classmethod(func: Callable) -> Callable:
        wrapped = postcondition(
            lambda cls, *a, response, **kw: (
                not isinstance(response, cls) or condition_callback(response)
            ),
            lambda cls, *a, response, **kw: error_callback(response),
        )(func)
        return classmethod(wrapped)  # type: ignore

    def wrap_staticmethod(func: Callable, cls: type[C]) -> Callable:
        wrapped = postcondition(
            lambda *a, response, **kw: (
                not isinstance(response, cls) or condition_callback(response)
            ),
            lambda *a, response, **kw: error_callback(response),
        )(func)
        return staticmethod(wrapped)

    def decorator(cls: type[C]) -> type[C]:
        original_setattr = cls.__setattr__

        if not rollback:
            checked_setattr = postcondition(
                lambda self, name, value, *, response: condition_callback(self),
                lambda self, name, value, *, response: error_callback(self),
            )(
                original_setattr
            )  # type: ignore
        else:

            def checked_setattr(self: C, name: str, value: object) -> None:
                state = snapshot(self)
                original_setattr(self, name, value)
                if not condition_callback(self):
                    error = error_callback(self)
                    restore(self, state)
                    raise error

        cls.__setattr__ = checked_setattr

        for name, member in vars(cls).copy().items():
            if isinstance(member, classmethod):
                setattr(cls, name, wrap_classmethod(member.__func__))
            elif isinstance(member, staticmethod):
                setattr(cls, name, wrap_staticmethod(member.__func__, cls))
            elif isinstance(member, property):
                continue
            elif name.startswith("__") and name != "__init__":
                continue
            elif callable(member):
                setattr(cls, name, wrap_method(member))

        return cls

    return decorator
