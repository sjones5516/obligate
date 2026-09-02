"""Boolean combinators over contracts.

Each combinator takes contracts of one type -- :class:`~obligate.validator.Precondition`,
:class:`~obligate.validator.Postcondition` or :class:`~obligate.validator.Invariant`
-- and returns a fresh contract of that *same* type, so results compose
further (``all_of(a, any_of(b, c))``) and drop in anywhere a plain contract
is expected.

They work by reusing each contract's clause protocol: ``_evaluate(*args,
**kwargs)`` reports whether every clause of that contract accepts the call,
and ``_make_error(*args, **kwargs)`` builds the first failing clause's
exception. The combined contract carries a single synthetic clause that
delegates to its parts.
"""

from typing import Callable

from .validator import Invariant, _Contract

__all__ = ["all_of", "any_of", "none_of", "not_"]


def _require_same_type(name: str, contracts: tuple[_Contract, ...]) -> type[_Contract]:
    if not contracts:
        raise TypeError(f"{name}() requires at least one contract")
    kinds = {type(c) for c in contracts}
    if len(kinds) != 1:
        listed = ", ".join(sorted(k.__name__ for k in kinds))
        raise TypeError(f"{name}() cannot mix contract types: {listed}")
    return kinds.pop()


def _rebuild(
    kind: type[_Contract],
    sources: tuple[_Contract, ...],
    *clauses: tuple[Callable[..., bool], Callable[..., Exception]],
) -> _Contract:
    """Make a ``kind`` contract from ``clauses``, carrying invariant policy."""
    if kind is Invariant:
        return Invariant(
            *clauses,
            rollback=any(getattr(s, "rollback", False) for s in sources),
            deep_copy=any(getattr(s, "deep_copy", False) for s in sources),
        )
    return kind(*clauses)


def all_of[T: _Contract](*contracts: T) -> T:
    """Require every contract to hold.

    Works for :class:`~obligate.validator.Precondition`,
    :class:`~obligate.validator.Postcondition` and
    :class:`~obligate.validator.Invariant`. The result checks each contract's
    clauses in the order given and raises the first failure's own exception,
    so messages stay specific. Nested compositions are flattened, so
    ``all_of(all_of(a, b), c)`` behaves like ``all_of(a, b, c)``.

    ``a & b`` is shorthand for ``all_of(a, b)``. When composing invariants,
    the result's ``rollback`` / ``deep_copy`` are the logical OR of the
    parts -- the stricter setting wins.

    Examples:
        >>> from obligate.validator import precondition, all_of
        >>> positive = precondition(
        ...     lambda n: n > 0, lambda n: ValueError(f"{n} is not positive")
        ... )
        >>> small = precondition(
        ...     lambda n: n < 10, lambda n: ValueError(f"{n} is too large")
        ... )
        >>> @all_of(positive, small)
        ... def clamp(n: int) -> int:
        ...     return n
        >>> clamp(5)
        5
        >>> clamp(-1)
        Traceback (most recent call last):
        ...
        ValueError: -1 is not positive
        >>> clamp(20)
        Traceback (most recent call last):
        ...
        ValueError: 20 is too large

    Args:
        contracts: Two or more contracts of the same type, checked left to
        right. Passing one returns it unchanged; passing none, or mixing
        types, raises :class:`TypeError`.
    """

    _require_same_type("all_of", contracts)
    result, *rest = contracts
    for other in rest:
        result = result & other
    return result  # type: ignore


def any_of[T: _Contract](*contracts: T) -> T:
    """Require at least one of the contracts to hold.

    The result passes as soon as any listed contract is fully satisfied. If
    every one fails, it raises an exception of the *first* alternative's
    error type whose message joins all the alternatives' complaints, so the
    caller sees why each option was rejected.

    Examples:
        >>> from obligate.validator import postcondition, any_of
        >>> is_none = postcondition(
        ...     lambda *a, response, **kw: response is None,
        ...     lambda *a, response, **kw: ValueError(f"{response!r} is not None"),
        ... )
        >>> positive_int = postcondition(
        ...     lambda *a, response, **kw: isinstance(response, int) and response > 0,
        ...     lambda *a, response, **kw: ValueError(f"{response!r} is not a positive int"),
        ... )
        >>> @any_of(is_none, positive_int)
        ... def lookup(found: bool) -> "int | None":
        ...     return 7 if found else None
        >>> lookup(True)
        7
        >>> lookup(False) is None
        True

        >>> @any_of(is_none, positive_int)
        ... def broken(x: int) -> int:
        ...     return -1
        >>> broken(0)
        Traceback (most recent call last):
        ...
        ValueError: no alternative held: -1 is not None; -1 is not a positive int

    Args:
        contracts: Two or more contracts of the same type. Passing one
        returns it unchanged; passing none, or mixing types, raises
        :class:`TypeError`.
    """

    kind = _require_same_type("any_of", contracts)
    if len(contracts) == 1:
        return contracts[0]

    def condition(*args, **kwargs) -> bool:
        return any(c._evaluate(*args, **kwargs) for c in contracts)

    def error(*args, **kwargs) -> Exception:
        failures = [c._make_error(*args, **kwargs) for c in contracts]
        joined = "; ".join(str(f) for f in failures)
        return type(failures[0])(f"no alternative held: {joined}")

    return _rebuild(kind, contracts, (condition, error))  # type: ignore


def not_[T: _Contract](contract: T, error_callback: Callable[..., Exception]) -> T:
    """Invert a contract: the result holds exactly when ``contract`` does not.

    Because the inverted contract fails when ``contract`` *succeeds*, there
    is no clause error to borrow, so you supply ``error_callback``. It is
    called with the same arguments as the contract's own callbacks (the
    wrapped function's ``*args, **kwargs``, plus ``response=`` for a
    postcondition, or the instance for an invariant).

    Examples:
        >>> from obligate.validator import precondition, not_
        >>> is_blank = precondition(
        ...     lambda s: s.strip() == "", lambda s: ValueError("blank")
        ... )
        >>> @not_(is_blank, lambda s: ValueError("must not be blank"))
        ... def shout(s: str) -> str:
        ...     return s.upper() + "!"
        >>> shout("hi")
        'HI!'
        >>> shout("   ")
        Traceback (most recent call last):
        ...
        ValueError: must not be blank

    Args:
        contract: The contract to invert.
        error_callback: Builds the exception to raise when ``contract``
        holds. Receives the same arguments as ``contract``'s callbacks.
    """

    def condition(*args, **kwargs) -> bool:
        return not contract._evaluate(*args, **kwargs)

    return _rebuild(type(contract), (contract,), (condition, error_callback))  # type: ignore


def none_of[T: _Contract](*contracts: T, error: Callable[..., Exception]) -> T:
    """Require that none of the contracts hold.

    ``error`` (keyword-only) builds the exception raised when any listed
    contract is satisfied; it receives the same arguments as the contracts'
    callbacks. Equivalent to ``not_(any_of(*contracts), error)``.

    Examples:
        >>> from obligate.validator import precondition, none_of
        >>> reserved = precondition(
        ...     lambda name: name in {"admin", "root"},
        ...     lambda name: ValueError("reserved"),
        ... )
        >>> has_space = precondition(
        ...     lambda name: " " in name, lambda name: ValueError("has space")
        ... )
        >>> @none_of(reserved, has_space, error=lambda name: ValueError(f"bad username: {name!r}"))
        ... def register(name: str) -> str:
        ...     return name
        >>> register("alice")
        'alice'
        >>> register("admin")
        Traceback (most recent call last):
        ...
        ValueError: bad username: 'admin'

    Args:
        contracts: One or more contracts of the same type.
        error: Keyword-only callback building the exception to raise when
        any contract holds.
    """

    if not contracts:
        raise TypeError("none_of() requires at least one contract")
    return not_(any_of(*contracts), error)
