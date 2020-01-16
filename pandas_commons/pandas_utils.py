import pandas_commons.string_utils as string_utils
import copy
import logging
import numpy
import pandas
from difflib import Differ
from diff_match_patch import diff_match_patch
from openpyxl import Workbook
from pandas import DataFrame, Index, RangeIndex, Series
from pathlib import Path
from typing import FrozenSet, List


"""Pandas utility module to assist with common Pandas constructs/functionality."""


def clean_dataframe(df: DataFrame, mandatory_columns: List[str] = None, index_col: str = None) -> DataFrame:
    """
    Removes all DataFrame rows containing NaN in any of the columns specified
    in keep_cols as well as any remaining NaNs in the DataFrame's index.
    Also formats the column names to be lower case, be separated strictly by
    underscores (not by any whitespace), and it will remove any parentheses
    or brackets.

    :param df: The DataFrame object to be cleaned.
    :param mandatory_columns: List of strings representing columns that will
        have their entire row dropped if it contains a cell with a NaN value.
    :param index_col: String value of the index column to be set after cleaning.
    :return: Equivalent cleaned DataFrame object.
    """

    df_cpy: DataFrame = df.copy()

    df_cpy.columns = df_cpy.columns.str.strip()
    df_cpy.columns = df_cpy.columns.str.replace(' ', '')
    df_cpy.columns = df_cpy.columns.str.replace('\r', '')
    df_cpy.columns = df_cpy.columns.str.replace('\n', '')
    df_cpy.columns = df_cpy.columns.str.replace('\t', '')
    df_cpy.columns = df_cpy.columns.str.replace('(', '')
    df_cpy.columns = df_cpy.columns.str.replace(')', '')
    df_cpy.columns = df_cpy.columns.str.replace('[', '')
    df_cpy.columns = df_cpy.columns.str.replace(']', '')
    df_cpy.columns = df_cpy.columns.str.replace('+', '')
    df_cpy.columns = df_cpy.columns.str.replace('?', '')

    if not index_is_default(df_cpy.index):
        df_cpy = reset_dataframe_index(df_cpy)

    df_cpy.columns = Series([string_utils.snake_case(column_name) for column_name in df_cpy.columns])

    try:
        df_cpy = df_cpy.dropna(how='all', subset=mandatory_columns)
    except KeyError as e:
        logging.warning(e)

        df_cpy = df_cpy.dropna(how='all')
    finally:
        df_cpy = df_cpy.fillna(value=str())

    if index_col and index_col in df_cpy.columns:
        if str() in df_cpy[index_col].values or numpy.nan in df_cpy[index_col].values:
            logging.warning(f'Index column {index_col} cannot be set. This column contains blank values.')
            logging.warning(
                f'Blank values are at: '
                f'{find_rows_with(df=DataFrame(df_cpy[index_col]), tokens=[str(), numpy.nan])}'
            )

            return df_cpy

        if not column_has_duplicates(column=df_cpy[index_col]):
            try:
                df_cpy.set_index(keys=index_col, drop=True, inplace=True, verify_integrity=True)
            except ValueError as e:
                logging.error(f'Index column {index_col} cannot be set.\r\n{str(e)}')
        else:
            logging.warning(f'Index column {index_col} cannot be set. This column contains duplicates.')
            logging.warning(f'Column duplicates are at: {get_column_duplicates(column=df_cpy[index_col])}')

    return df_cpy


def column_has_duplicates(column: Series) -> bool:
    """Indicates whether a DataFrame's column contains any duplicates."""

    return column.duplicated().any()


def drop_rows_with(df: DataFrame, values: List[str], columns: List[str]) -> DataFrame:
    """
    Drops rows containing that contain the passed values in the
    passed columns.

    :param df: DataFrame object to work on.
    :param values: List of values to search for in the given columns.
    :param columns: List of columns to base dropping off of.
    :return: A new DataFrame without the rows that had the passed values for
    the specified columns.
    """

    df_cpy: DataFrame = df.copy()

    for column_name in columns:
        for value in values:
            if column_name in df_cpy.columns:
                rows_with_value: Series = df_cpy[column_name].str.contains(value)
                value_index: Index = rows_with_value[rows_with_value == True].index

                df_cpy.drop(value_index, inplace=True)

    return df_cpy


def find_rows_with(df: DataFrame, tokens: List[str]) -> DataFrame:
    """
    Finds the row number index for each string in a list of strings. The
    returned DataFrame is not representative of the order or the specific column
    of each specific string in the list of strings (i.e. tokens).

    It is simply a collective representing;

        "all of these strings are contained within these rows, and none others"

    :param df: DataFrame object to search through.
    :param tokens: The list of strings to look for within the DataFrame.
    :return: A DataFrame of rows that contain the list of strings passed.
    """

    indices_containing_tokens: List[int] = list()

    for column_name, column_data in df.iteritems():
        for token in tokens:
            if pandas.isna(token):
                contains_token = column_data.isna()
            else:
                contains_token = column_data.str.contains(token)

            token_rows = contains_token[contains_token == True]

            if len(token_rows) > 0:
                indices_containing_tokens = indices_containing_tokens + list(token_rows.index.values)

    return df.iloc[list(set(indices_containing_tokens))]


def get_changed_columns(old: DataFrame, new: DataFrame) -> List[str]:
    """
    Returns a list of all the columns that contain any changes.

    :param old: DataFrame representing the older revision.
    :param new: DataFrame representing the newer revision.
    :return: List of changed columns.
    """

    changed_columns: List[str] = list()
    is_in_axis: DataFrame

    try:
        if len(old.index) >= len(new.index):
            is_in_axis = old.isin(new)
        elif len(new.index) > len(old.index):
            is_in_axis = new.isin(old)
    except ValueError:
        logging.error(
            'pandas_commons.pandas_utils.get_changed_columns: \r\n'
            '\tOld revision and new revision do not share indices.\r\n'
            '\tCannot perform pandas.DataFrame.isin().\r\n'
            '\tResetting indexes for old and new revisions.'
        )

        old.reset_index(inplace=True)
        new.reset_index(inplace=True)
    finally:
        if len(old.index) >= len(new.index):
            is_in_axis = old.isin(new)
        elif len(new.index) >= len(old.index):
            is_in_axis = new.isin(old)

    for (column_name, is_unchanged) in is_in_axis.all().iteritems():
        if not is_unchanged:
            changed_columns.append(column_name)

    return changed_columns


def get_column_diff(old: Series, new: Series) -> Series:
    """
    Returns the diff of every changed element between two columns
    in the form of pandas Series objects.

    :param old: Series object representing older revision of the column.
    :param new: Series object representing newer revision of the column.
    :return: Series object containing all diffed elements between revisions.
    """

    differ = Differ()
    diff_result: List[str] = list()

    old = old.str.strip()
    new = new.str.strip()

    # Returns Series of booleans.
    # False values indicate that the revisions were not equal at that index.
    column_comparison: Series = old.eq(new)
    column_comparison = column_comparison[column_comparison == False]

    compare_result: List[str] = list()

    for index in column_comparison.index:
        if index in old.index and index in new.index:
            # Changed row.
            old_val = str(old[index]).splitlines()
            new_val = str(new[index]).splitlines()

            compare_result = list(differ.compare(old_val, new_val))

        if index in old.index and not (index in new.index):
            # Deleted row.
            compare_result = list(str(old[index]).splitlines())

        if not (index in old.index) and index in new.index:
            # Added row.
            compare_result = list(str(new[index]).splitlines())

        diff_result.append('<br/>'.join(compare_result))

    if diff_result:
        return Series(data=diff_result, index=column_comparison.index, name='diff')

    return Series()


def get_column_duplicates(column: Series) -> List[str]:
    """
    Returns the row indexes of the duplicated attempted index column.

    :param column: The attempted column to set as the DataFrame index.
    :return: List of the duplicate row indexes.
    """

    duplicated: Series = column.duplicated()
    duplicates: Series = duplicated[duplicated == True]

    duplicated_indices = list()

    for index, value in duplicates.iteritems():
        duplicated_indices.append(column[index])

    return sorted(set(duplicated_indices))


def get_columns_with(
        df: DataFrame,
        include_columns: List[str] = None,
        exclude_columns: List[str] = None,
        include_columns_with: List[str] = None,
        exclude_columns_with: List[str] = None
) -> DataFrame:
    """
    Returns a subset of a DataFrame containing only the columns
    with titles that contain certain keywords. This function also gives the
    option to include and exclude specific columns by title or by keywords.

    :param df: The original DataFrame to search.
    :param include_columns: List of column names to include/keep.
    :param exclude_columns: List of column names to exclude/discard.
    :param include_columns_with: List of keywords to include any columns that may contain these words.
    :param exclude_columns_with: List of keywords to exclude any columns that may contain these words.
    :return: DataFrame containing only the columns that contain the substring and/or included columns.
    """

    columns: List[str] = list()
    df_cpy: DataFrame = df.copy()

    if exclude_columns:
        exclude: List[str] = list(filter(lambda x: x in df_cpy.columns.values, exclude_columns))
        df_cpy.drop(exclude, inplace=True, axis=1)

    if include_columns:
        columns = list(filter(lambda x: x in df_cpy.columns.values, include_columns))

    if include_columns_with:
        for column_name in df_cpy.columns:
            for include_with in include_columns_with:
                if include_with.lower() in column_name:
                    columns.append(column_name)

    if exclude_columns_with:
        columns_buffer: List[str] = copy.deepcopy(columns)

        for exclude_with in exclude_columns_with:
            for column_name in columns_buffer:
                if exclude_with in column_name:
                    columns.remove(column_name)

    return df_cpy[set(columns)]


def get_dataframe_diff_list(old: DataFrame, new: DataFrame, changed_columns: List[str]) -> List[DataFrame]:
    """
    Returns a list of DataFrames that displays side-by-side the
    difference of the old and new revision of the DataFrames for each specified
    column in the list passed.

    :param old: DataFrame object representing older revision.
    :param new: DataFrame object representing newer revision.
    :param changed_columns: List of column names that contain changes.
    :return: List of DataFrame objects that compare changes per column.
    """

    diff_list: List[DataFrame] = list()

    old: DataFrame = old[changed_columns]
    new: DataFrame = new[changed_columns]

    full_outer_join: DataFrame = pandas.merge(
        left=old,
        right=new,
        how='outer',
        left_index=True,
        right_index=True,
        suffixes=('_OLD', '_NEW')
    )

    # Boolean DataFrame indicating whether the value in a given cell from the
    # old DataFrame is equal to the value from the new DataFrame.
    compare_cells: DataFrame = old.eq(new)

    for column_name, column in compare_cells.iteritems():
        changed_indices: Index = column[column == False].index
        changed_subset_df: DataFrame = full_outer_join.loc[changed_indices]

        old_column_name = '{}_{}'.format(column_name, 'OLD')
        new_column_name = '{}_{}'.format(column_name, 'NEW')

        diffed_column: Series = get_column_diff(
            old=changed_subset_df[old_column_name],
            new=changed_subset_df[new_column_name]
        )

        diffed_df: DataFrame = changed_subset_df[[old_column_name, new_column_name]]

        # Append diffed_column here;
        merged = pandas.merge(
            left=diffed_df,
            right=diffed_column,
            how='outer',
            left_index=True,
            right_index=True,
            suffixes=('_compare', 'diff')
        )

        diff_list.append(merged)

    return diff_list


def index_has_duplicates(index: Index) -> bool:
    """Indicates whether a DataFrame's Index contains any duplicates."""

    return index.duplicated().any()


def index_is_default(index: Index) -> bool:
    """Indicates whether a pandas Index is the default."""

    return type(index) is RangeIndex


def merge_dataframes(dataframes: List[DataFrame]) -> DataFrame:
    result = dataframes[0]

    for dataframe in dataframes[1:]:
        difference: DataFrame = dataframe.eq(result)
        columns_with_differences = dict()

        for column_name, column in difference.iteritems():
            column_inverted: Series = ~column
            columns_with_differences[column_name] = dataframe[column_name][column_inverted]

        unequal_rows = clean_dataframe(df=DataFrame(columns_with_differences))

        result = pandas.concat([result, unequal_rows])
        result = reset_dataframe_index(df=result)

    return result


def reset_dataframe_index(df: DataFrame) -> bool:
    """
    Resets a DataFrame's index and then deletes the accompanied
    left-over "index" column if available.

    :param df: The DataFrame whose index will be reset.
    :return: The DataFrame with it's index reset.
    """
    df.reset_index(inplace=True)

    if 'index' in df.columns:
        return df.drop(labels=['index'], axis=1)

    return df


def write_dataframe_to_excel(df: DataFrame, output_file: Path) -> None:
    """
    Writes a DataFrame to an Excel file.

    :param df: DataFrame object to write as an Excel spreadsheet.
    :param path: Output path to the location of the generated Excel file.
    """

    if not output_file.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    with pandas.ExcelWriter(path=str(output_file), engine='openpyxl') as xl_writer:
        workbook = Workbook()  # openpyxl.load_workbook(output_file)
        workbook.save(output_file)

        xl_writer.book = workbook
        xl_writer.sheets = dict((worksheet.title, worksheet) for worksheet in workbook.worksheets)

        df.to_excel(xl_writer, index=False)

        workbook.save(output_file)
        workbook.close()
