from typing import TYPE_CHECKING, List, Optional
from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput
from pydantic import Field

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput

class DropEmpty(GlobalStep):
    """Drop rows containing empty values in specified columns.

    A `GlobalStep` that filters out rows where specified columns (or all columns if none specified)
    contain empty values (None, empty string, or whitespace).

    Attributes:
        columns: Optional list of column names to check for empty values. If not provided,
            all columns will be checked. Defaults to None.

    Runtime parameters:
        - `columns`: List of columns to check for empty values.

    Input columns:
        - dynamic (`all`): All columns from the input will be checked based on the columns parameter.

    Output columns:
        - dynamic (`all`): All input columns are preserved for non-empty rows.

    Categories:
        - preprocessing
        - filter
        - cleaning

    Examples:
        Drop rows with empty values in specific columns:

        ```python
        from distilabel.steps import DropEmpty

        # Drop rows with empty values in 'instruction' or 'response' columns
        drop_step = DropEmpty(columns=["instruction", "response"])
        drop_step.load()

        result = next(
            drop_step.process(
                [
                    {
                        "instruction": "What is 2+2?",
                        "response": "4",
                        "metadata": ""
                    },
                    {
                        "instruction": "",  # This row will be dropped
                        "response": "Some response",
                        "metadata": "test"
                    }
                ]
            )
        )
        # >>> result
        # [
        #     {
        #         "instruction": "What is 2+2?",
        #         "response": "4",
        #         "metadata": ""
        #     }
        # ]
        ```

        Drop rows with empty values in any column:

        ```python
        # Drop rows with empty values in any column
        drop_step = DropEmpty()
        drop_step.load()

        result = next(
            drop_step.process(
                [
                    {
                        "instruction": "What is 2+2?",
                        "response": "4"
                    },
                    {
                        "instruction": "Another question",
                        "response": ""  # This row will be dropped
                    }
                ]
            )
        )
        ```
    """

    columns: Optional[RuntimeParameter[List[str]]] = Field(
        default=None,
        description="Optional list of column names to check for empty values. If not provided, "
        "all columns will be checked."
    )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """Process the input data by filtering out rows with empty values.

        Args:
            inputs: The input data to be filtered.

        Yields:
            Filtered data with empty rows removed based on specified columns.
        """
        filtered_inputs = []

        for item in inputs:
            # Determine which columns to check
            columns_to_check = self.columns if self.columns is not None else item.keys()

            # Check if any specified column contains empty values
            has_empty = any(
                not str(item[col]).strip() if item[col] is not None else True
                for col in columns_to_check
                if col in item
            )

            if not has_empty:
                filtered_inputs.append(item)

        yield filtered_inputs
