"""Shared DataFrame utility functions."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def compact_dataframe(
    dataframe: pd.DataFrame,
    categorical_columns: Iterable[str] = (),
    integer_columns: Iterable[str] = (),
    float_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """
    Reduce a DataFrame's memory use by converting selected columns.

    The input DataFrame is modified in place and returned for convenience.

    Integer columns containing missing values may remain nullable/floating
    depending on their contents and the pandas version.
    """
    for column in categorical_columns:
        if column in dataframe.columns:
            dataframe[column] = dataframe[column].astype("category")

    for column in integer_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
                downcast="integer",
            )

    for column in float_columns:
        if column in dataframe.columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
                downcast="float",
            )

    return dataframe