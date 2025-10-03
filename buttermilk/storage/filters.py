"""Example filter implementations for storage record filtering/sampling.

These filters can be used with Storage.iterate_async() to implement
custom sampling strategies for pipelines.
"""

import random
from typing import Any

from buttermilk._core.types import BaseRecord
from buttermilk.storage.base import RecordFilter


class RandomSampler(RecordFilter):
    """Randomly sample records with a given probability.

    Example:
        # Sample 10% of records
        filter = RandomSampler(0.1)
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, probability: float = 0.1):
        """Initialize random sampler.

        Args:
            probability: Probability of including each record (0.0 to 1.0)
        """
        if not 0.0 <= probability <= 1.0:
            raise ValueError("Probability must be between 0.0 and 1.0")
        self.probability = probability

    async def should_include(self, record: BaseRecord) -> bool:
        """Randomly decide whether to include record."""
        return random.random() < self.probability


class DatasetFilter(RecordFilter):
    """Filter records by dataset_name field.

    Example:
        # Only include records from specific datasets
        filter = DatasetFilter(["tmdb", "imdb"])
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, datasets: list[str] | str):
        """Initialize dataset filter.

        Args:
            datasets: Dataset name(s) to include
        """
        if isinstance(datasets, str):
            datasets = [datasets]
        self.datasets = set(datasets)

    async def should_include(self, record: BaseRecord) -> bool:
        """Check if record is from allowed dataset."""
        return record.dataset_name in self.datasets


class SplitTypeFilter(RecordFilter):
    """Filter records by split_type field.

    Example:
        # Only include test records
        filter = SplitTypeFilter("test")
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, split_types: list[str] | str):
        """Initialize split type filter.

        Args:
            split_types: Split type(s) to include (e.g., "train", "test", "val")
        """
        if isinstance(split_types, str):
            split_types = [split_types]
        self.split_types = set(split_types)

    async def should_include(self, record: BaseRecord) -> bool:
        """Check if record is from allowed split."""
        return record.split_type in self.split_types


class MetadataFilter(RecordFilter):
    """Filter records based on metadata field values.

    Example:
        # Filter movies by year and popularity
        filter = MetadataFilter({
            "year": lambda y: 2020 <= y <= 2023,
            "popularity": lambda p: p > 50
        })
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, conditions: dict[str, Any]):
        """Initialize metadata filter.

        Args:
            conditions: Dict mapping metadata keys to filter conditions.
                       Values can be:
                       - A specific value (equality check)
                       - A callable that returns True/False
                       - A list/set of allowed values
        """
        self.conditions = conditions

    async def should_include(self, record: BaseRecord) -> bool:
        """Check if record metadata matches all conditions."""
        for key, condition in self.conditions.items():
            value = record.metadata.get(key)

            # Handle different condition types
            if callable(condition):
                # Callable predicate
                try:
                    if not condition(value):
                        return False
                except Exception:
                    return False
            elif isinstance(condition, (list, set, tuple)):
                # Value must be in collection
                if value not in condition:
                    return False
            # Direct equality check
            elif value != condition:
                return False

        return True


class YearRangeFilter(RecordFilter):
    """Filter Title records by year range.

    Example:
        # Only include movies from 2015-2020
        filter = YearRangeFilter(2015, 2020)
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, min_year: int | None = None, max_year: int | None = None):
        """Initialize year range filter.

        Args:
            min_year: Minimum year (inclusive), None for no minimum
            max_year: Maximum year (inclusive), None for no maximum
        """
        self.min_year = min_year
        self.max_year = max_year

    async def should_include(self, record: BaseRecord) -> bool:
        """Check if record has year in range."""
        # Check if record has a year attribute (like Title records)
        if hasattr(record, "year"):
            year = record.year
            if year is None:
                return False
            if self.min_year is not None and year < self.min_year:
                return False
            if self.max_year is not None and year > self.max_year:
                return False
            return True

        # For regular records, check metadata
        year = record.metadata.get("year")
        if year is None:
            return True  # No year to filter on
        try:
            year = int(year)
            if self.min_year is not None and year < self.min_year:
                return False
            if self.max_year is not None and year > self.max_year:
                return False
            return True
        except (ValueError, TypeError):
            return True  # Can't parse year, don't filter


class CompositeFilter(RecordFilter):
    """Combine multiple filters with AND/OR logic.

    Example:
        # Sample 10% of test records from 2020-2023
        filter = CompositeFilter.all_of(
            RandomSampler(0.1),
            SplitTypeFilter("test"),
            YearRangeFilter(2020, 2023)
        )
        async for record in storage(filter=filter):
            process(record)
    """

    def __init__(self, filters: list[RecordFilter], mode: str = "and"):
        """Initialize composite filter.

        Args:
            filters: List of filters to combine
            mode: "and" (all must pass) or "or" (at least one must pass)
        """
        if mode not in ("and", "or"):
            raise ValueError("Mode must be 'and' or 'or'")
        self.filters = filters
        self.mode = mode

    @classmethod
    def all_of(cls, *filters: RecordFilter) -> "CompositeFilter":
        """Create a filter requiring ALL conditions to pass."""
        return cls(list(filters), mode="and")

    @classmethod
    def any_of(cls, *filters: RecordFilter) -> "CompositeFilter":
        """Create a filter requiring ANY condition to pass."""
        return cls(list(filters), mode="or")

    async def should_include(self, record: BaseRecord) -> bool:
        """Apply composite filtering logic."""
        if self.mode == "and":
            # All filters must pass
            for filter in self.filters:
                if not await filter.should_include(record):
                    return False
            return True
        else:  # mode == "or"
            # At least one filter must pass
            for filter in self.filters:
                if await filter.should_include(record):
                    return True
            return False
