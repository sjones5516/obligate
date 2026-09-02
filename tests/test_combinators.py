import pytest

from obligate.contracts import (
    Invariant,
    all_of,
    any_of,
    invariant,
    none_of,
    not_,
    postcondition,
    precondition,
)

positive = precondition(lambda n: n > 0, lambda n: ValueError(f"{n} not positive"))
small = precondition(lambda n: n < 10, lambda n: ValueError(f"{n} too large"))
even = precondition(lambda n: n % 2 == 0, lambda n: ValueError(f"{n} not even"))

positive_result = postcondition(
    lambda *a, response, **kw: response > 0,
    lambda *a, response, **kw: ValueError(f"{response} not positive"),
)
even_result = postcondition(
    lambda *a, response, **kw: response % 2 == 0,
    lambda *a, response, **kw: ValueError(f"{response} not even"),
)


# --- all_of / & over preconditions -----------------------------------


def test_all_of_allows_value_satisfying_every_precondition():
    @all_of(positive, small, even)
    def f(n):
        return n

    assert f(4) == 4


def test_all_of_raises_first_failing_preconditions_error():
    @all_of(positive, small, even)
    def f(n):
        return n

    with pytest.raises(ValueError, match="-2 not positive"):
        f(-2)
    with pytest.raises(ValueError, match="20 too large"):
        f(20)
    with pytest.raises(ValueError, match="3 not even"):
        f(3)


def test_all_of_checks_in_given_order():
    @all_of(small, positive)
    def f(n):
        return n

    # -1 fails both "small" (no) and "positive" -- small passes, positive fails
    with pytest.raises(ValueError, match="-1 not positive"):
        f(-1)

    @all_of(positive, small)
    def g(n):
        return n

    # 20 fails "small"; positive passes first, so "small" wins
    with pytest.raises(ValueError, match="20 too large"):
        g(20)


def test_and_operator_composes_two_preconditions():
    combined = positive & small

    @combined  # type: ignore
    def f(n):
        return n

    assert f(5) == 5
    with pytest.raises(ValueError, match="0 not positive"):
        f(0)
    with pytest.raises(ValueError, match="99 too large"):
        f(99)


def test_composition_flattens_nested_preconditions():
    nested = all_of(all_of(positive, small), even)
    flat = all_of(positive, small, even)
    assert len(nested._clauses) == len(flat._clauses) == 3

    chained = positive & small & even
    assert len(chained._clauses) == 3


def test_composition_does_not_mutate_the_parts():
    _ = positive & small
    assert len(positive._clauses) == 1
    assert len(small._clauses) == 1


def test_all_of_with_single_precondition_behaves_like_it():
    @all_of(positive)
    def f(n):
        return n

    assert f(1) == 1
    with pytest.raises(ValueError, match="0 not positive"):
        f(0)


def test_composed_precondition_forwards_all_arguments_to_callbacks():
    both_positive = all_of(
        precondition(lambda a, b: a > 0, lambda a, b: ValueError(f"a={a}")),
        precondition(lambda a, b: b > 0, lambda a, b: ValueError(f"b={b}")),
    )

    @both_positive
    def add(a, b):
        return a + b

    assert add(1, 2) == 3
    with pytest.raises(ValueError, match="b=-1"):
        add(1, -1)


def test_composed_precondition_can_layer_onto_a_new_inline_condition():
    @positive & precondition(lambda n: n != 7, lambda n: ValueError("no sevens"))  # type: ignore
    def f(n):
        return n

    assert f(3) == 3
    with pytest.raises(ValueError, match="no sevens"):
        f(7)


# --- all_of / & over postconditions ---------------------------------


def test_all_of_composes_postconditions_and_raises_first_failure():
    @all_of(positive_result, even_result)
    def identity(n):
        return n

    assert identity(4) == 4
    with pytest.raises(ValueError, match="-2 not positive"):
        identity(-2)
    with pytest.raises(ValueError, match="3 not even"):
        identity(3)


def test_and_operator_composes_postconditions():
    @positive_result & even_result  # type: ignore
    def identity(n):
        return n

    assert identity(2) == 2
    with pytest.raises(ValueError, match="1 not even"):
        identity(1)


def test_composed_postcondition_forwards_arguments_and_response():
    grew = postcondition(
        lambda value, *, response: response > value,
        lambda value, *, response: ValueError(f"{response} <= {value}"),
    )
    bounded = postcondition(
        lambda value, *, response: response < 100,
        lambda value, *, response: ValueError(f"{response} >= 100"),
    )

    @grew & bounded  # type: ignore
    def step(value):
        return value + 1

    assert step(5) == 6
    with pytest.raises(ValueError, match="100 >= 100"):
        step(99)


# --- all_of / & over invariants ------------------------------------


def test_all_of_composes_invariants_over_one_attribute():
    non_negative = invariant(
        lambda self: self.balance >= 0,  # type: ignore
        lambda self: ValueError(f"{self.balance} < 0"),  # type: ignore
    )
    capped = invariant(
        lambda self: self.balance <= 100,  # type: ignore
        lambda self: ValueError(f"{self.balance} > 100"),  # type: ignore
    )

    @all_of(non_negative, capped)
    class Account:
        def __init__(self, balance):
            self.balance = balance

        def add(self, amount):
            self.balance += amount

    account = Account(50)
    account.add(30)
    assert account.balance == 80
    with pytest.raises(ValueError, match="130 > 100"):
        account.add(50)
    with pytest.raises(ValueError, match="-1 < 0"):
        account.balance = -1


def test_composed_invariant_may_reference_attributes_set_across_init():
    has_owner = invariant(
        lambda self: self.owner != "", lambda self: ValueError("no owner")  # type: ignore
    )
    solvent = invariant(
        lambda self: self.balance >= 0,  # type: ignore
        lambda self: ValueError(f"balance {self.balance}"),  # type: ignore
    )

    @has_owner & solvent  # type: ignore
    class Account:
        def __init__(self, owner, balance):
            self.owner = owner  # solvent would see no ``balance`` yet...
            self.balance = balance  # ...but the check is deferred to here

    account = Account("alice", 10)
    assert (account.owner, account.balance) == ("alice", 10)

    with pytest.raises(ValueError, match="no owner"):
        Account("", 10)
    with pytest.raises(ValueError, match="balance -5"):
        Account("bob", -5)


def test_and_operator_composes_invariants():
    a = invariant(lambda self: self.x > 0, lambda self: ValueError("x<=0"))  # type: ignore
    b = invariant(lambda self: self.x < 10, lambda self: ValueError("x>=10"))  # type: ignore

    @a & b  # type: ignore
    class Box:
        def __init__(self, x):
            self.x = x

        def set(self, x):
            self.x = x

    box = Box(5)
    box.set(9)
    with pytest.raises(ValueError, match="x>=10"):
        box.set(10)


def test_composed_invariant_ors_rollback_flag():
    plain = invariant(lambda self: self.n <= 2, lambda self: ValueError("too big"))  # type: ignore
    rolling = invariant(
        lambda self: self.n >= 0, lambda self: ValueError("negative"), rollback=True  # type: ignore
    )

    combined = plain & rolling
    assert combined.rollback is True  # type: ignore

    @combined  # type: ignore
    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self, by):
            self.n += by

    counter = Counter()
    counter.bump(2)
    with pytest.raises(ValueError, match="too big"):
        counter.bump(5)
    assert counter.n == 2  # rolled back because either part asked for it


def test_composed_invariant_ors_deep_copy_flag():
    length_ok = invariant(
        lambda self: len(self.items) <= 2,  # type: ignore
        lambda self: ValueError("too many"),
        rollback=True,
    )
    unique = invariant(
        lambda self: len(set(self.items)) == len(self.items),  # type: ignore
        lambda self: ValueError("dupes"),
        rollback=True,
        deep_copy=True,
    )

    combined = length_ok & unique
    assert combined.deep_copy is True  # type: ignore

    @combined  # type: ignore
    class Bag:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

    bag = Bag()
    bag.add("a")
    bag.add("b")
    with pytest.raises(ValueError, match="too many"):
        bag.add("c")
    assert bag.items == ["a", "b"]  # in-place append undone via deep copy


def test_composed_invariant_checks_classmethod_factory():
    non_negative = invariant(
        lambda self: self.n >= 0, lambda self: ValueError(f"n={self.n}")  # type: ignore
    )
    even = invariant(
        lambda self: self.n % 2 == 0, lambda self: ValueError(f"odd n={self.n}")  # type: ignore
    )

    @non_negative & even  # type: ignore
    class Num:
        def __init__(self, n):
            self.n = n

        @classmethod
        def build(cls, n):
            obj = cls.__new__(cls)
            object.__setattr__(obj, "n", n)
            return obj

    assert Num.build(4).n == 4
    with pytest.raises(ValueError, match="odd n=3"):
        Num.build(3)


# --- composing across contract types ---------------------------------


def test_cannot_compose_different_contract_types():
    with pytest.raises(TypeError, match="same"):
        precondition(lambda: True, lambda: ValueError()) & postcondition(
            lambda *a, response: True, lambda *a, response: ValueError()
        )  # type: ignore


def test_all_of_rejects_mixed_contract_types():
    with pytest.raises(TypeError):
        all_of(
            precondition(lambda: True, lambda: ValueError()),
            invariant(lambda self: True, lambda self: ValueError()),
        )


def test_all_of_requires_at_least_one_contract():
    with pytest.raises(TypeError):
        all_of()


# --- any_of -----------------------------------------------------------


def test_any_of_passes_when_one_alternative_holds():
    is_zero = precondition(lambda n: n == 0, lambda n: ValueError(f"{n} != 0"))

    @any_of(positive, is_zero)
    def f(n):
        return n

    assert f(5) == 5
    assert f(0) == 0
    with pytest.raises(ValueError, match="no alternative held"):
        f(-3)


def test_any_of_error_joins_every_alternatives_complaint():
    is_zero = precondition(lambda n: n == 0, lambda n: ValueError(f"{n} is not zero"))

    @any_of(positive, is_zero)
    def f(n):
        return n

    with pytest.raises(ValueError, match="-3 not positive; -3 is not zero"):
        f(-3)


def test_any_of_error_type_follows_first_alternative():
    class FirstError(Exception):
        pass

    class SecondError(Exception):
        pass

    a = precondition(lambda n: False, lambda n: FirstError("a"))
    b = precondition(lambda n: False, lambda n: SecondError("b"))

    @any_of(a, b)
    def f(n):
        return n

    with pytest.raises(FirstError):
        f(1)


def test_any_of_composes_with_all_of():
    is_zero = precondition(lambda n: n == 0, lambda n: ValueError("not zero"))

    @all_of(small, any_of(positive, is_zero))
    def f(n):
        return n

    assert f(0) == 0
    assert f(4) == 4
    with pytest.raises(ValueError, match="too large"):
        f(20)
    with pytest.raises(ValueError, match="no alternative held"):
        f(-1)


def test_any_of_works_for_postconditions():
    is_none = postcondition(
        lambda *a, response: response is None,
        lambda *a, response: ValueError(f"{response!r} not None"),
    )
    is_positive = postcondition(
        lambda *a, response: isinstance(response, int) and response > 0,
        lambda *a, response: ValueError(f"{response!r} not positive"),
    )

    @any_of(is_none, is_positive)
    def maybe(x):
        return x

    assert maybe(3) == 3
    assert maybe(None) is None
    with pytest.raises(ValueError, match="no alternative held"):
        maybe(-1)


def test_any_of_works_for_invariants():
    frozen = invariant(
        lambda self: self.frozen, lambda self: ValueError("not frozen")  # type: ignore
    )
    solvent = invariant(
        lambda self: self.balance >= 0, lambda self: ValueError("insolvent")  # type: ignore
    )

    @any_of(frozen, solvent)
    class Account:
        def __init__(self, balance, frozen=False):
            self.balance = balance
            self.frozen = frozen

        def charge(self, amount):
            self.balance -= amount

    ok = Account(10)
    ok.charge(5)
    assert ok.balance == 5

    frozen_acct = Account(-100, frozen=True)  # insolvent but frozen -> allowed
    assert frozen_acct.balance == -100

    with pytest.raises(ValueError, match="no alternative held"):
        Account(5).charge(10)


def test_any_of_single_contract_returns_it():
    assert any_of(positive) is positive


def test_any_of_rejects_mixed_types():
    with pytest.raises(TypeError, match="cannot mix"):
        any_of(positive, postcondition(lambda *a, response: True, lambda *a, r: r))


# --- not_ -----------------------------------------------------------


def test_not_inverts_a_precondition():
    is_blank = precondition(lambda s: s.strip() == "", lambda s: ValueError("blank"))

    @not_(is_blank, lambda s: ValueError("must not be blank"))
    def f(s):
        return s

    assert f("hi") == "hi"
    with pytest.raises(ValueError, match="must not be blank"):
        f("   ")


def test_not_error_callback_receives_the_arguments():
    positive_pre = precondition(lambda n: n > 0, lambda n: ValueError("pos"))

    @not_(positive_pre, lambda n: ValueError(f"{n} must not be positive"))
    def f(n):
        return n

    assert f(-2) == -2
    with pytest.raises(ValueError, match="5 must not be positive"):
        f(5)


def test_not_of_invariant_keeps_rollback():
    at_zero = invariant(
        lambda self: self.n == 0, lambda self: ValueError("zero"), rollback=True  # type: ignore
    )

    guard = not_(at_zero, lambda self: ValueError(f"n must not be 0"))
    assert isinstance(guard, Invariant)
    assert guard.rollback is True

    @guard
    class Counter:
        def __init__(self):
            self.n = 5

        def to(self, n):
            self.n = n

    c = Counter()
    c.to(3)
    with pytest.raises(ValueError, match="n must not be 0"):
        c.to(0)
    assert c.n == 3  # rolled back


def test_double_negation_round_trips():
    is_even = precondition(lambda n: n % 2 == 0, lambda n: ValueError("odd"))
    same = not_(
        not_(is_even, lambda n: ValueError("x")), lambda n: ValueError("not even")
    )

    @same
    def f(n):
        return n

    assert f(4) == 4
    with pytest.raises(ValueError, match="not even"):
        f(3)


# --- none_of -------------------------------------------------------


def test_none_of_rejects_when_any_contract_holds():
    reserved = precondition(
        lambda name: name in {"admin", "root"}, lambda name: ValueError("reserved")
    )
    spaced = precondition(lambda name: " " in name, lambda name: ValueError("space"))

    @none_of(reserved, spaced, error=lambda name: ValueError(f"bad: {name!r}"))
    def register(name):
        return name

    assert register("alice") == "alice"
    with pytest.raises(ValueError, match="bad: 'admin'"):
        register("admin")
    with pytest.raises(ValueError, match="bad: 'a b'"):
        register("a b")


def test_none_of_requires_a_contract():
    with pytest.raises(TypeError):
        none_of(error=lambda: ValueError())


def test_none_of_error_is_keyword_only():
    reserved = precondition(lambda n: n == 0, lambda n: ValueError("zero"))
    with pytest.raises(TypeError):
        none_of(reserved, lambda n: ValueError("boom"))  # type: ignore
