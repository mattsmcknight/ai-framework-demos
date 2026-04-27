"""
Tests for knapsack data models.
"""

import pytest

from knapsack.models import Item, KnapsackResult, validate_capacity, validate_items


class TestItem:
    """Tests for the Item dataclass."""

    def test_create_valid_item(self):
        """Should create item with valid weight and value."""
        item = Item(weight=10, value=60)
        assert item.weight == 10
        assert item.value == 60
        assert item.name is None

    def test_create_item_with_name(self):
        """Should create item with optional name."""
        item = Item(weight=5, value=30, name="laptop")
        assert item.name == "laptop"

    def test_item_is_immutable(self):
        """Item should be frozen (immutable)."""
        item = Item(weight=10, value=60)
        with pytest.raises(AttributeError):
            item.weight = 20  # type: ignore

    def test_zero_weight_raises_error(self):
        """Should reject item with zero weight."""
        with pytest.raises(ValueError, match="weight must be positive"):
            Item(weight=0, value=60)

    def test_negative_weight_raises_error(self):
        """Should reject item with negative weight."""
        with pytest.raises(ValueError, match="weight must be positive"):
            Item(weight=-5, value=60)

    def test_negative_value_raises_error(self):
        """Should reject item with negative value."""
        with pytest.raises(ValueError, match="value must be non-negative"):
            Item(weight=10, value=-10)

    def test_zero_value_is_valid(self):
        """Zero value should be allowed (edge case: mandatory items)."""
        item = Item(weight=10, value=0)
        assert item.value == 0

    def test_value_density(self):
        """Should calculate correct value-to-weight ratio."""
        item = Item(weight=10, value=60)
        assert item.value_density() == 6.0

    def test_value_density_non_integer(self):
        """Should handle non-integer density."""
        item = Item(weight=3, value=10)
        assert abs(item.value_density() - 3.333) < 0.01

    def test_repr_without_name(self):
        """Should have clean repr without name."""
        item = Item(weight=10, value=60)
        assert repr(item) == "Item(weight=10, value=60)"

    def test_repr_with_name(self):
        """Should include name in repr when present."""
        item = Item(weight=10, value=60, name="gold")
        assert repr(item) == "Item(weight=10, value=60, name='gold')"


class TestKnapsackResult:
    """Tests for the KnapsackResult dataclass."""

    @pytest.fixture
    def sample_items(self):
        return (Item(10, 60), Item(20, 100))

    def test_create_result(self, sample_items):
        """Should create result with all required fields."""
        result = KnapsackResult(
            selected_items=sample_items,
            total_value=160,
            total_weight=30,
            capacity=50,
            algorithm="dynamic_programming",
        )
        assert result.total_value == 160
        assert result.total_weight == 30
        assert result.capacity == 50
        assert result.is_optimal is True  # default

    def test_utilization_calculation(self, sample_items):
        """Should calculate capacity utilization correctly."""
        result = KnapsackResult(
            selected_items=sample_items,
            total_value=160,
            total_weight=30,
            capacity=50,
            algorithm="dp",
        )
        assert result.utilization == 0.6

    def test_utilization_zero_capacity(self):
        """Should handle zero capacity edge case."""
        result = KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=0,
            algorithm="dp",
        )
        assert result.utilization == 0.0

    def test_item_count(self, sample_items):
        """Should return correct item count."""
        result = KnapsackResult(
            selected_items=sample_items,
            total_value=160,
            total_weight=30,
            capacity=50,
            algorithm="dp",
        )
        assert result.item_count == 2

    def test_empty_result(self):
        """Should handle empty solution."""
        result = KnapsackResult(
            selected_items=(),
            total_value=0,
            total_weight=0,
            capacity=50,
            algorithm="dp",
        )
        assert result.item_count == 0
        assert result.utilization == 0.0


class TestValidateItems:
    """Tests for validate_items function."""

    def test_validate_list_of_items(self):
        """Should accept list and return tuple."""
        items = [Item(10, 60), Item(20, 100)]
        result = validate_items(items)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_validate_tuple_of_items(self):
        """Should accept tuple."""
        items = (Item(10, 60),)
        result = validate_items(items)
        assert result == items

    def test_validate_non_iterable_raises(self):
        """Should reject non-iterable input."""
        with pytest.raises(TypeError, match="Expected iterable"):
            validate_items(42)  # type: ignore

    def test_validate_non_item_in_sequence_raises(self):
        """Should reject non-Item elements."""
        with pytest.raises(TypeError, match="Expected Item at index 1"):
            validate_items([Item(10, 60), "not an item"])  # type: ignore


class TestValidateCapacity:
    """Tests for validate_capacity function."""

    def test_valid_positive_capacity(self):
        """Should accept positive integer capacity."""
        assert validate_capacity(100) == 100

    def test_zero_capacity_is_valid(self):
        """Should accept zero capacity (edge case)."""
        assert validate_capacity(0) == 0

    def test_negative_capacity_raises(self):
        """Should reject negative capacity."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_capacity(-10)

    def test_non_integer_capacity_raises(self):
        """Should reject non-integer capacity."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_capacity(10.5)  # type: ignore

    def test_string_capacity_raises(self):
        """Should reject string capacity."""
        with pytest.raises(TypeError, match="must be an integer"):
            validate_capacity("100")  # type: ignore
