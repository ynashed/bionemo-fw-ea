# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from abc import ABC
from dataclasses import dataclass
from typing import List, NamedTuple, Sequence, Type

import pandas as pd


__all__: Sequence[str] = (
    # main abstraction & utilities for columns required in a dataframe
    "RequiredColumns",
    "check_cols_adhere_contract",
    "columns_of",
    "schema_of",
    "raise_value_error_if_missing_columns",
    "ColumnSchema",
    "missing_columns",
    # groups of columns
    "RequiredCiJobColumns",
    "RequiredTotalDurationCols",
    "RequiredPipelineCols",
    "RequiredStageCols",
    "RequiredTimeCols",
    "RequiredDateCols",
    "RequiredTimePlotCols",
    # individual columns
    "RequiredPipelineId",
    "RequiredStage",
    "RequiredName",
    "RequiredFailureReason",
    "RequiredStartedAt",
    "RequiredCreatedAt",
    "RequiredFinishedAt",
    "RequiredStatus",
    "RequiredDuration",
    "RequiredSource",
    "RequiredRef",
    "RequiredSha",
    "RequiredQueuedDuration",
    "RequiredTotalDurationS",
    "RequiredTotalDurationTs",
    "RequiredTotalElapsedS",
    "RequiredTotalElapsedTs",
    "RequiredTotalQueuedS",
    "RequiredTotalQueuedTs",
)


@dataclass(frozen=True)
class RequiredColumns(ABC):
    """Used for a function to explicitly declare the columns it requires to use in input dataframes.

    ## Motivation
    Consider two dataframes that describe similiar data, but come from different systems. Their column naming schema may not
    align with one another. In this case, it's important to be able to write code that is agnostic to the specific, actual names
    of the columns. Rather, the code should refer to each column conceptually, and then use a mapping from concept to actual
    column name for each of the specific data sources.

    If we write code like:

         def failed_only(df: pd.DataFrame) -> pd.DataFrame:
            return df['status'] == 'failed'

    Then we can't use it if one of our dataframes names this column e.g. `"outcome"`. To work around this, we could write:

         def failed_only(df: pd.DataFrame, col_status: str = 'status') -> pd.DataFrame:
            return df[col_status] == 'failed'

    And use `failed_only(df0, 'status')` and `failed_only(df1, "outcome")` for our two different dataframes.

    However, we have to repeat ourselves on every function that needs to refer to our `status` column: copying-and-pasting both
    the column name in our code (`col_status`) and our default value for it (`"status"`). This quickly becomes unmanageable when
    we get to supporting sets of columns: it turns our code into a keyword-argument soup! It also means that we need to keep
    repeating documentation of what the columns mean.

    Thus, it would be better if we could refer to a group of columns together. This is precisely what a `RequiredColumns` class is!

    ## Use
    One must extend this class and define fields with default values that correspond to columns in a dataframe. The field name
    is the conceptual name of the column -- it's stable and can be used in code. The field value is the actual column name as
    it appears in data (i.e. a :class:`pd.DataFrame`).

    NOTE: Extending classes **MUST** only have fields that correspond to column names!

    ## Example
    Going back to our `failed_only` function example, we would refactor it to use `RequiredColumns` as:

        class RequiredStatus(RequiredColumns):
            status: str = 'status'

        def failed_only(df: pd.DataFrame, cols: RequiredStatus = RequiredStatus()) -> pd.DataFrame:
            return df[cols.status] == 'failed'

    And thus we'd use this as:

        df0: pd.DataFrame = ...
        df1: pd.DataFrame = ...

        failed_only(df0)
        failed_only(df1, cols=RequiredStatus(status='outcome'))

    If we had a dataframe with another column:

        class RequiredIsMain(RequiredColumns):
            is_main: str = 'is_main'

    Then we can use multiple-inheritence to easily create a group of required columns without needing
    to re-define the columns nor their default values:

        class RequiredGroup(RequiredIsMain, RequiredStatus):
            pass

    And use it in a function:

        def successful_in_group(
            pipeline_ids: List[int],
            df: pd.DataFrame,
            cols: RequiredGroup = RequiredGroup(),
        ) -> pd.DataFrame:
            idx = df[cols.is_main]
            return df[idx, cols.status] == 'success'
    """

    def columns(self) -> List[str]:
        """Returns the instance's required columns."""
        return columns_of(type(self), self)


def columns_of(contract: Type[RequiredColumns], cols: RequiredColumns) -> List[str]:
    """Obtains the actual column names from :params:`cols` using :param:`contract` to control the subset of selected columns.

    The :param:`contract` makes it possible for :params:`cols` to have more columns than are necessary
    for a particular application.

    NOTE: This function is aligned 1:1 with :func:`schema_of`: `zip(columns_of(C, cols), schema_of(C))`
    """
    check_cols_adhere_contract(contract, cols)
    return [getattr(cols, c) for c in schema_of(contract)]


def schema_of(contract: Type[RequiredColumns]) -> List[str]:
    """Obtains the schema column names from :params:`contract` (aka field names).


    NOTE: This function is aligned 1:1 with :func:`columns_of`: `zip(columns_of(C, cols), schema_of(C))`
    """
    if not isinstance(contract, type):
        raise TypeError(f"Expecting contrac to be a type, not a {type(contract)} ({contract})")

    if not issubclass(contract, RequiredColumns):
        raise TypeError(f"Contract type ({contract}) must be a {RequiredColumns}!")

    return list(contract.__dataclass_fields__.keys())


def check_cols_adhere_contract(contract: Type[RequiredColumns], cols: RequiredColumns) -> None:
    """Rasies a ValueError iff the columns (:param:`cols`) do not adhere to the specified schema (:param:`contract`).

    Raises a TypeError if the :param:`contract` is not a :class:`RequiredColumns` type or if :param:`cols` is not
    a :class:`RequiredColumns` instance. Note that :param:`cols` may be a subclass of the :param:`contract` type,
    but it is not strictly necessary. It must be a _structural subtype_: i.e. it must have the same defined columns.
    """
    if not isinstance(contract, type):
        raise TypeError(f"Expecting contrac to be a type, not a {type(contract)} ({contract})")

    if not issubclass(contract, RequiredColumns):
        raise TypeError(f"Contract type ({contract}) must be a {RequiredColumns}!")

    if not isinstance(cols, RequiredColumns):
        raise TypeError(f"Columns instance must be a {RequiredColumns}, not a {type(cols)}!")

    if not isinstance(cols, contract):
        missing_from_cols: List[str] = [
            column_generic_name
            for column_generic_name in contract.__dataclass_fields__.keys()
            if getattr(cols, column_generic_name, None) is None
        ]
        if len(missing_from_cols) > 0:
            raise ValueError(
                f"Expecting {type(cols)} to be a have all required columns of {contract}. "
                f"However, it is missing {len(missing_from_cols)} required columns: {missing_from_cols}"
            )


class ColumnSchema(NamedTuple):
    """A reference to a column: maps a name to use for a column in code to the actual column name as it appears in data."""

    column_generic_name: str
    """The name of a field of some `RequiredColumns`-extending `class`.
    """

    actual_column_name: str
    """The actual column name in a Pandas DataFrame.
    """


def missing_columns(contract: Type[RequiredColumns], cols: RequiredColumns, df: pd.DataFrame) -> List[ColumnSchema]:
    """Returns a list of columns missing from the dataframe.

    The required columns are specified by :param:`contract`, while the actual column value names are in :param:`cols`.

    The returned list contains each column specified in :param:`contract` that's missing
    """
    check_cols_adhere_contract(contract, cols)
    return [
        ColumnSchema(column_generic_name, getattr(cols, column_generic_name, None))
        for column_generic_name in contract.__dataclass_fields__.keys()
        if getattr(cols, column_generic_name) not in df.columns
    ]


def raise_value_error_if_missing_columns(
    contract: Type[RequiredColumns], cols: RequiredColumns, df: pd.DataFrame
) -> None:
    """Raises an error if the dataframe is missing any required columns.

    Uses :func:`missing_columns` (see its docs for details on :param:`contract` and :param:`cols`) to determine
    if the input Pandas DataFrame (:param:`df`) is missing any columns from the :param:`contract`.

    If so, the resulting :class:`ValueError`'s message will contain the number of missing columns, the specific missing
    columns (their schema name and their expected dataframe name), and the dataframe's actual columns.
    """
    validation_check = missing_columns(contract, cols, df)
    if len(validation_check) > 0:
        missing_msg = "\n".join([f"- missing '{col}' (set as {field})" for field, col in validation_check])
        raise ValueError(
            f"Missing {len(validation_check)} required columns from dataframe:\n"
            f"{missing_msg}\n"
            f"Dataframe columns ({len(df.columns)}): {df.columns}"
        )


#
#
#
# All individual columns used in data representations using any Gitlab API response or any CI pipeline-processing
# function in the `bionemo.ci`` package.
#
#
#


@dataclass(frozen=True)
class RequiredPipelineId(RequiredColumns):
    pipeline_id: str = 'pipeline_id'


@dataclass(frozen=True)
class RequiredProjectId(RequiredColumns):
    project_id: str = 'project_id'


@dataclass(frozen=True)
class RequiredStage(RequiredColumns):
    stage: str = 'stage'


@dataclass(frozen=True)
class RequiredName(RequiredColumns):
    name: str = "name"


@dataclass(frozen=True)
class RequiredFailureReason(RequiredColumns):
    failure_reason: str = "failure_reason"


@dataclass(frozen=True)
class RequiredStartedAt(RequiredColumns):
    started_at: str = "started_at"


@dataclass(frozen=True)
class RequiredCreatedAt(RequiredColumns):
    created_at: str = "created_at"


@dataclass(frozen=True)
class RequiredFinishedAt(RequiredColumns):
    finished_at: str = "finished_at"


@dataclass(frozen=True)
class RequiredStatus(RequiredColumns):
    status: str = "status"


@dataclass(frozen=True)
class RequiredDuration(RequiredColumns):
    duration: str = "duration"


@dataclass(frozen=True)
class RequiredSource(RequiredColumns):
    source: str = "source"


@dataclass(frozen=True)
class RequiredRef(RequiredColumns):
    ref: str = "ref"


@dataclass(frozen=True)
class RequiredSha(RequiredColumns):
    sha: str = "sha"


@dataclass(frozen=True)
class RequiredQueuedDuration(RequiredColumns):
    queued_duration: str = "queued_duration"


@dataclass(frozen=True)
class RequiredTotalDurationS(RequiredColumns):
    total_duration_s: str = "total_duration_s"


@dataclass(frozen=True)
class RequiredTotalDurationTs(RequiredColumns):
    total_duration_ts: str = "total_duration_ts"


@dataclass(frozen=True)
class RequiredTotalElapsedS(RequiredColumns):
    total_elapsed_s: str = "total_elapsed_s"


@dataclass(frozen=True)
class RequiredTotalElapsedTs(RequiredColumns):
    total_elapsed_ts: str = "total_elapsed_ts"


@dataclass(frozen=True)
class RequiredTotalQueuedS(RequiredColumns):
    total_queued_s: str = "total_queued_s"


@dataclass(frozen=True)
class RequiredTotalQueuedTs(RequiredColumns):
    total_queued_ts: str = "total_queued_ts"


#
#
#
# All groups of required columns that dataframe-manipulating functions require in the `bionemo.ci` package.
#
#
#


@dataclass(frozen=True)
class RequiredCiJobColumns(
    RequiredPipelineId,
    RequiredDuration,
    RequiredFailureReason,
    RequiredStage,
    RequiredName,
    RequiredStartedAt,
    RequiredStatus,
):
    """Columns required for any dataframe describing CI jobs/stages that were executed in a pipeline."""


@dataclass(frozen=True)
class RequiredTotalDurationCols(
    RequiredTotalDurationS,
    RequiredTotalDurationTs,
    RequiredTotalElapsedS,
    RequiredTotalElapsedTs,
    RequiredTotalQueuedS,
    RequiredTotalQueuedTs,
):
    """Columns required for calculating the duration of a CI pipeline."""


@dataclass(frozen=True)
class RequiredPipelineCols(
    RequiredSource,
    RequiredRef,
    RequiredSha,
    RequiredStatus,
    RequiredCreatedAt,
):
    """Columns required for dataframes that describe CI pipelines defined on a project."""


@dataclass(frozen=True)
class RequiredCompletionCols(
    RequiredStatus,
    RequiredCreatedAt,
):
    """Columns required to determine which pipelines were most recently completed."""


@dataclass(frozen=True)
class RequiredStageCols(
    RequiredStage,
    RequiredName,
    RequiredStatus,
    RequiredCreatedAt,
    RequiredStartedAt,
    RequiredFinishedAt,
    RequiredDuration,
    RequiredQueuedDuration,
    RequiredFailureReason,
    RequiredPipelineId,
):
    """Columns required to merge bridge (triggered) CI stages from other projects into a project's CI stages."""


@dataclass(frozen=True)
class RequiredTimeCols(
    RequiredPipelineId,
    RequiredDuration,
    RequiredStatus,
    RequiredQueuedDuration,
    RequiredCreatedAt,
    RequiredFinishedAt,
):
    """Columns required to calculate all timing metrics."""


@dataclass(frozen=True)
class RequiredElapsedTimeCols(
    RequiredStatus,
    RequiredCreatedAt,
    RequiredFinishedAt,
):
    """Columns required to calculate elapsed time."""


@dataclass(frozen=True)
class RequiredDateCols(
    RequiredPipelineId,
    RequiredProjectId,
    RequiredRef,
    RequiredCreatedAt,
    RequiredFinishedAt,
    RequiredStartedAt,
):
    """Columns required for pipeline dates."""


@dataclass(frozen=True)
class RequiredTimePlotCols(
    RequiredStatus,
    RequiredCreatedAt,
    RequiredStartedAt,
    RequiredFinishedAt,
    RequiredDuration,
    RequiredQueuedDuration,
):
    """Columns required to plot CI pipeline duration."""
