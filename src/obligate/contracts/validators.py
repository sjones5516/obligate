from typing import Callable
import copy
import functools

_INITIALIZING = "_obligate_initializing"


class _Contract:
    """Shared composition machinery for the contract types.

    Every contract holds an ordered tuple of ``(condition, error)`` clauses.
    Each clause callback is invoked with whatever arguments the contract's
    ``check`` receives -- the wrapped function's ``*args, **kwargs`` for a
    :class:`Precondition`, those plus ``response=`` for a
    :class:`Postcondition`, and the instance for an :class:`Invariant`.

    ``a & b`` (equivalently :func:`~obligate.contracts.all_of`) builds a new
    contract of the same type whose clauses are the two clause lists
    concatenated, checked in order; the first failing clause raises its own
    error. The combinators in :mod:`obligate.contracts.combinators` build on
    the same clause protocol.
    """

    _clauses: tuple

    def _evaluate(self, *args, **kwargs) -> bool:
        """Return ``True`` only if every own clause accepts the arguments."""
        return all(condition(*args, **kwargs) for condition, _ in self._clauses)

    def _make_error(self, *args, **kwargs) -> Exception:
        """Build the first failing clause's error for the given arguments."""
        for condition, error in self._clauses:
            if not condition(*args, **kwargs):
                return error(*args, **kwargs)
        raise AssertionError(  # pragma: no cover - only called on a violation
            "contract is satisfied; no error to build"
        )

    def _combine(self, other: "_Contract") -> "_Contract":
        raise NotImplementedError

    def __and__(self, other: "_Contract") -> "_Contract":
        if type(self) is not type(other):
            raise TypeError(
                f"cannot compose {type(self).__name__} with "
                f"{type(other).__name__}; both operands must be the same "
                "contract type"
            )
        return self._combine(other)


class Precondition[**P](_Contract):
    """A reusable, composable guard on a function's arguments.

    A :class:`Precondition` bundles a condition with the exception to raise
    when that condition fails. The same instance can decorate any number of
    functions, and several can be combined into one with ``&`` (or
    :func:`~obligate.contracts.all_of`); the combined precondition checks each part in order and
    raises the first failing part's *own* error, so messages stay specific.

    Build one with :func:`precondition` rather than instantiating directly.

    Examples:
        >>> from obligate.contracts import precondition
        >>> non_negative = precondition(
        ...     lambda value: value >= 0,
        ...     lambda value: ValueError(f"got {value}, want >= 0"),
        ... )
        >>> whole = precondition(
        ...     lambda value: value == int(value),
        ...     lambda value: ValueError(f"got {value}, want a whole number"),
        ... )

        Reuse a single precondition across functions:

        >>> @non_negative
        ... def square_root(value: float) -> float:
        ...     return value ** 0.5
        >>> square_root(9)
        3.0

        Layer preconditions by composing them; each part is checked in
        order and the first failure raises its own error:

        >>> count = non_negative & whole
        >>> @count
        ... def countdown(value: float) -> list[int]:
        ...     return list(range(int(value), 0, -1))
        >>> countdown(3)
        [3, 2, 1]
        >>> countdown(-1)
        Traceback (most recent call last):
        ...
        ValueError: got -1, want >= 0
        >>> countdown(1.5)
        Traceback (most recent call last):
        ...
        ValueError: got 1.5, want a whole number
    """

    def __init__(
        self, *clauses: tuple[Callable[P, bool], Callable[P, Exception]]
    ) -> None:
        self._clauses = clauses

    def check(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Raise the first clause's error whose condition rejects the arguments."""
        if not self._evaluate(*args, **kwargs):
            raise self._make_error(*args, **kwargs)

    def __call__[R](self, func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            self.check(*args, **kwargs)
            return func(*args, **kwargs)

        return wrapper

    def _combine(self, other: "Precondition[P]") -> "Precondition[P]":  # type: ignore
        return Precondition(*self._clauses, *other._clauses)


def precondition[**P](
    condition_callback: Callable[P, bool], error_callback: Callable[P, Exception]
) -> Precondition[P]:
    """Guard a function with a precondition, raising a dynamically built exception.

    ``error_callback`` is called with the *same* arguments that were passed
    to the wrapped function, letting the exception message embed the actual
    values that failed validation.

    Returns a :class:`Precondition`: use it as a decorator, reuse it across
    functions, and combine it with others via ``&`` or :func:`~obligate.contracts.all_of`.

    Examples:
        >>> from obligate.contracts import precondition
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
        condition_callback: Returns ``True`` if the arguments are valid.
        error_callback: Called as ``error_callback(*args, **kwargs)`` when
        ``condition_callback`` returns ``False``.
    """

    return Precondition((condition_callback, error_callback))


class Postcondition[**P](_Contract):
    """A reusable, composable guard on a function's return value.

    A :class:`Postcondition` bundles a condition with the exception to raise
    when it fails. Both callbacks receive the wrapped function's arguments
    plus its result as the ``response`` keyword argument. Reuse one instance
    across functions, and combine several with ``&`` or :func:`~obligate.contracts.all_of` --
    the combined postcondition checks each part in order and raises the
    first failure's own error.

    Build one with :func:`postcondition` rather than instantiating directly.

    Examples:
        >>> from obligate.contracts import postcondition
        >>> positive = postcondition(
        ...     lambda *a, response, **kw: response > 0,
        ...     lambda *a, response, **kw: ValueError(f"{response} is not positive"),
        ... )
        >>> even = postcondition(
        ...     lambda *a, response, **kw: response % 2 == 0,
        ...     lambda *a, response, **kw: ValueError(f"{response} is not even"),
        ... )
        >>> @positive & even
        ... def result(n: int) -> int:
        ...     return n
        >>> result(4)
        4
        >>> result(-2)
        Traceback (most recent call last):
        ...
        ValueError: -2 is not positive
        >>> result(3)
        Traceback (most recent call last):
        ...
        ValueError: 3 is not even
    """

    def __init__(
        self, *clauses: tuple[Callable[..., bool], Callable[..., Exception]]
    ) -> None:
        self._clauses = clauses

    def check(self, *args, response, **kwargs) -> None:
        """Raise the first clause's error whose condition rejects ``response``."""
        if not self._evaluate(*args, response=response, **kwargs):
            raise self._make_error(*args, response=response, **kwargs)

    def __call__[R](self, func: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            response = func(*args, **kwargs)
            self.check(*args, response=response, **kwargs)
            return response

        return wrapper

    def _combine(self, other: "Postcondition[P]") -> "Postcondition[P]":  # type: ignore
        return Postcondition(*self._clauses, *other._clauses)


def postcondition[**P, R](
    condition_callback: Callable[..., bool], error_callback: Callable[..., Exception]
) -> Postcondition[P]:
    """Guard a function with a postcondition, raising a dynamically built exception.

    Both callbacks receive the arguments passed to the wrapped function and
    the returned value as the ``response`` keyword argument.

    Returns a :class:`Postcondition`: use it as a decorator, reuse it across
    functions, and combine it with others via ``&`` or :func:`~obligate.contracts.all_of`.

    Examples:
        >>> from obligate.contracts import postcondition
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

    return Postcondition((condition_callback, error_callback))


class Invariant[C](_Contract):
    """A reusable, composable class invariant enforced on every mutation.

    An :class:`Invariant` bundles a condition on an instance with the
    exception to raise when it fails, plus the ``rollback`` / ``deep_copy``
    policy. Apply one instance to any number of classes, and combine several
    with ``&`` or :func:`~obligate.contracts.all_of`; the combined invariant checks every clause
    after each mutation and raises the first failure's own error. When
    combining, ``rollback`` and ``deep_copy`` are OR-ed -- the stricter
    setting wins.

    Build one with :func:`invariant` rather than instantiating directly.

    Examples:
        >>> from obligate.contracts import invariant
        >>> non_negative = invariant(
        ...     lambda self: self.balance >= 0,
        ...     lambda self: ValueError(f"balance {self.balance} < 0"),
        ... )
        >>> under_limit = invariant(
        ...     lambda self: self.balance <= 100,
        ...     lambda self: ValueError(f"balance {self.balance} > 100"),
        ... )
        >>> @non_negative & under_limit
        ... class Account:
        ...     def __init__(self, balance: float) -> None:
        ...         self.balance = balance
        ...     def add(self, amount: float) -> None:
        ...         self.balance += amount
        >>> a = Account(50)
        >>> a.add(30)
        >>> a.balance
        80
        >>> a.add(50)
        Traceback (most recent call last):
        ...
        ValueError: balance 130 > 100
        >>> a.balance = -1
        Traceback (most recent call last):
        ...
        ValueError: balance -1 < 0
    """

    def __init__(
        self,
        *clauses: tuple[Callable[[C], bool], Callable[[C], Exception]],
        rollback: bool = False,
        deep_copy: bool = False,
    ) -> None:
        self._clauses = clauses
        self.rollback = rollback
        self.deep_copy = deep_copy

    def _combine(self, other: "Invariant[C]") -> "Invariant[C]":  # type: ignore
        return Invariant(
            *self._clauses,
            *other._clauses,
            rollback=self.rollback or other.rollback,
            deep_copy=self.deep_copy or other.deep_copy,
        )

    def check(self, instance: C) -> None:
        """Raise the first clause's error that ``instance`` violates."""
        if not self._evaluate(instance):
            raise self._make_error(instance)

    def _snapshot(self, instance: C) -> dict:
        attrs = vars(instance)
        return copy.deepcopy(attrs) if self.deep_copy else copy.copy(attrs)

    @staticmethod
    def _restore(instance: C, state: dict) -> None:
        vars(instance).clear()
        vars(instance).update(state)

    def __call__[T](self, cls: type[T]) -> type[T]:
        satisfied = self._evaluate
        build_error = self._make_error
        rollback = self.rollback
        snapshot = self._snapshot
        restore = self._restore

        def initializing(instance: object) -> bool:
            return bool(getattr(instance, _INITIALIZING, 0))

        def wrap_init(init: Callable) -> Callable:
            @functools.wraps(init)
            def wrapper(self, *args, **kwargs):
                tracked = True
                try:
                    depth = getattr(self, _INITIALIZING, 0)
                    object.__setattr__(self, _INITIALIZING, depth + 1)
                except AttributeError, TypeError:
                    tracked = False
                try:
                    init(self, *args, **kwargs)
                finally:
                    if tracked:
                        depth = getattr(self, _INITIALIZING, 1) - 1
                        if depth:
                            object.__setattr__(self, _INITIALIZING, depth)
                        else:
                            try:
                                object.__delattr__(self, _INITIALIZING)
                            except AttributeError:
                                pass
                if not initializing(self) and not satisfied(self):
                    raise build_error(self)

            return wrapper

        def wrap_method(method: Callable) -> Callable:
            @functools.wraps(method)
            def wrapper(self, *args, **kwargs):
                if initializing(self):
                    return method(self, *args, **kwargs)
                if not rollback:
                    result = method(self, *args, **kwargs)
                    if not satisfied(self):
                        raise build_error(self)
                    return result
                state = snapshot(self)
                result = method(self, *args, **kwargs)
                if not satisfied(self):
                    error = build_error(self)
                    restore(self, state)
                    raise error
                return result

            return wrapper

        def wrap_classmethod(func: Callable) -> classmethod:
            wrapped = postcondition(
                lambda owner, *a, response, **kw: (
                    not isinstance(response, owner) or satisfied(response)
                ),
                lambda owner, *a, response, **kw: build_error(response),
            )(func)
            return classmethod(wrapped)  # type: ignore

        def wrap_staticmethod(func: Callable, owner: type) -> staticmethod:
            wrapped = postcondition(
                lambda *a, response, **kw: (
                    not isinstance(response, owner) or satisfied(response)
                ),
                lambda *a, response, **kw: build_error(response),
            )(func)
            return staticmethod(wrapped)

        original_setattr = cls.__setattr__

        def checked_setattr(self, name: str, value: object) -> None:
            if initializing(self):
                original_setattr(self, name, value)
                return
            if not rollback:
                original_setattr(self, name, value)
                if not satisfied(self):
                    raise build_error(self)
                return
            state = snapshot(self)
            original_setattr(self, name, value)
            if not satisfied(self):
                error = build_error(self)
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
            elif name == "__init__":
                setattr(cls, name, wrap_init(member))
            elif name.startswith("__"):
                continue
            elif callable(member):
                setattr(cls, name, wrap_method(member))

        return cls


def invariant[C](
    condition_callback: Callable[[C], bool],
    error_callback: Callable[[C], Exception],
    *,
    rollback: bool = False,
    deep_copy: bool = False,
) -> Invariant[C]:
    """Guard a class with an invariant enforced on every attribute mutation.

    Every guarded call point follows the same call-then-check-else-raise
    shape as :func:`postcondition`, adapted to check ``self`` (or the
    returned instance, for factories) rather than a function's return value.

    Returns an :class:`Invariant`: reuse it across classes and combine it
    with others via ``&`` or :func:`~obligate.contracts.all_of`.

    The invariant is first checked when ``__init__`` returns -- not on the
    individual assignments inside it -- so ``__init__`` may build the object
    up across several statements (and composed invariants may reference
    attributes set at different points in it). After that, it is checked
    after every attribute assignment (so external code mutating the object
    directly is caught immediately), after every instance method call, and
    on the returned instance of any classmethod or staticmethod used as a
    factory.

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
        >>> from obligate.contracts import invariant
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

    return Invariant(
        (condition_callback, error_callback), rollback=rollback, deep_copy=deep_copy
    )
