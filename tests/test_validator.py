import pytest

from obligate.validator import invariant, postcondition, precondition


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


def account_class(**invariant_kwargs):
    @invariant(
        lambda self: self.balance >= 0,  # type: ignore
        lambda self: ValueError(f"Balance went negative: {self.balance}"),  # type: ignore
        **invariant_kwargs,
    )
    class Account:
        def __init__(self, balance):
            self.balance = balance

        def withdraw(self, amount):
            self.balance -= amount
            return self.balance

        def deposit(self, amount):
            self.balance += amount

        @classmethod
        def with_balance(cls, balance):
            return cls(balance)

        @classmethod
        def label(cls):
            return "account"

        @staticmethod
        def make(balance):
            return Account(balance)

    return Account


# --- construction ----------------------------------------------------------


def test_invariant_allows_valid_construction():
    account = account_class()(10)
    assert account.balance == 10


def test_invariant_rejects_invalid_construction():
    with pytest.raises(ValueError, match="Balance went negative: -1"):
        account_class()(-1)


def test_invariant_condition_error_propagates_when_attribute_missing():
    @invariant(lambda self: self.ready, lambda self: ValueError("not ready"))  # type: ignore
    class Widget:
        def __init__(self):
            self.name = "w"  # invariant checked here, before ``ready`` exists

    with pytest.raises(AttributeError):
        Widget()


# --- instance methods -----------------------------------------------------


def test_invariant_allows_valid_method_call():
    account = account_class()(10)
    account.withdraw(4)
    assert account.balance == 6


def test_invariant_preserves_method_return_value():
    account = account_class()(10)
    assert account.withdraw(3) == 7


def test_invariant_rejects_invalid_method_call():
    account = account_class()(10)
    with pytest.raises(ValueError, match="Balance went negative: -5"):
        account.withdraw(15)


def test_invariant_without_rollback_leaves_object_mutated():
    account = account_class()(10)
    with pytest.raises(ValueError):
        account.withdraw(15)
    assert account.balance == -5


# --- direct attribute assignment ----------------------------------------


def test_invariant_allows_valid_attribute_assignment():
    account = account_class()(10)
    account.balance = 3
    assert account.balance == 3


def test_invariant_catches_external_attribute_mutation():
    account = account_class()(10)
    with pytest.raises(ValueError, match="Balance went negative: -1"):
        account.balance = -1


# --- rollback -----------------------------------------------------------


def test_invariant_rollback_restores_state_after_bad_method():
    account = account_class(rollback=True)(10)
    account.withdraw(5)
    with pytest.raises(ValueError, match="Balance went negative: -95"):
        account.withdraw(100)
    assert account.balance == 5


def test_invariant_rollback_restores_state_after_bad_assignment():
    account = account_class(rollback=True)(10)
    with pytest.raises(ValueError):
        account.balance = -1
    assert account.balance == 10


def test_invariant_rollback_keeps_good_mutations():
    account = account_class(rollback=True)(10)
    account.deposit(5)
    account.withdraw(3)
    assert account.balance == 12


def test_invariant_shallow_rollback_does_not_undo_in_place_mutation():
    @invariant(
        lambda self: len(self.items) <= 2,  # type: ignore
        lambda self: ValueError("too many"),
        rollback=True,
    )
    class Cart:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

    cart = Cart()
    cart.add("a")
    cart.add("b")
    with pytest.raises(ValueError, match="too many"):
        cart.add("c")
    assert cart.items == ["a", "b", "c"]


def test_invariant_deep_copy_rollback_undoes_in_place_mutation():
    @invariant(
        lambda self: len(self.items) <= 2,  # type: ignore
        lambda self: ValueError("too many"),
        rollback=True,
        deep_copy=True,
    )
    class Cart:
        def __init__(self):
            self.items = []

        def add(self, item):
            self.items.append(item)

    cart = Cart()
    cart.add("a")
    cart.add("b")
    with pytest.raises(ValueError, match="too many"):
        cart.add("c")
    assert cart.items == ["a", "b"]


# --- factories --------------------------------------------------------


def test_invariant_allows_valid_classmethod_factory():
    account = account_class().with_balance(10)
    assert account.balance == 10


def test_invariant_rejects_invalid_classmethod_factory():
    with pytest.raises(ValueError, match="Balance went negative: -1"):
        account_class().with_balance(-1)


def test_invariant_rejects_invalid_staticmethod_factory():
    with pytest.raises(ValueError, match="Balance went negative: -1"):
        account_class().make(-1)


def test_invariant_ignores_classmethod_returning_non_instance():
    assert account_class().label() == "account"


# --- other members ---------------------------------------------------


def test_invariant_does_not_wrap_properties():
    @invariant(lambda self: self.value >= 0, lambda self: ValueError("negative"))  # type: ignore
    class Box:
        def __init__(self, value):
            self._value = value

        @property
        def value(self):
            return self._value

        @property
        def doubled(self):
            return self._value * 2

    box = Box(4)
    assert box.value == 4
    assert box.doubled == 8


def test_invariant_supports_custom_exception_type():
    class BrokenInvariant(Exception):
        pass

    @invariant(
        lambda self: self.n == 0,  # type: ignore
        lambda self: BrokenInvariant(f"n is {self.n}"),  # type: ignore
    )
    class Counter:
        def __init__(self):
            self.n = 0

        def bump(self):
            self.n += 1

    counter = Counter()
    with pytest.raises(BrokenInvariant, match="n is 1"):
        counter.bump()


def test_invariant_decorator_returns_same_class():
    Account = account_class()
    assert Account.__name__ == "Account"
    assert isinstance(Account(1), Account)
