import json
from typing import TYPE_CHECKING

from pydantic import Field

from distilabel.mixins.runtime_parameters import RuntimeParameter
from distilabel.steps.base import GlobalStep, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepOutput


class SaveToJsonl(GlobalStep):
    """
    Save messages to a JSONL file.

    A GlobalStep that extracts the 'messages' column from the input data and saves each
    instance to a JSON Lines file. Each line in the file will be a JSON object following the format:
    {"messages": [{"role": "user", "content": "hi"}, ...]}

    Attributes:
        file_path: The file path where the JSONL data will be saved. Defaults to 'output.jsonl'.

    Input columns:
        - messages: A column expected to contain a list of message objects with keys 'role' and 'content'.

    Categories:
        - save
        - file

    Examples:
        Save your messages to a JSONL file:

        ```python
        from distilabel.steps import SaveToJsonl

        save_step = SaveToJsonl(file_path='my_messages.jsonl')
        list(save_step.process([
            {"messages": [{"role": "user", "content": "hi"}]}
        ]))
        # The file 'my_messages.jsonl' will contain a line:
        # {"messages": [{"role": "user", "content": "hi"}]}
        ```
    """

    file_path: RuntimeParameter[str] = Field(
        default="output.jsonl",
        description="The file path where the messages will be saved as JSONL.",
    )

    def process(self, inputs: StepInput) -> "StepOutput":  # type: ignore
        """
        Process the input data to extract the 'messages' column and write it to a JSONL file.
        Each line in the file corresponds to one input instance with the structure:
        {"messages": [{"role": "user", "content": "hi"}, ...]}

        Args:
            inputs: A list of dictionaries, each expected to contain a 'messages' key.

        Yields:
            The original inputs unmodified.
        """
        with open(self.file_path, "w") as f:
            for input_item in inputs:
                messages = input_item.get("messages")
                if messages is not None:
                    # Ensure messages is a list
                    if not isinstance(messages, list):
                        messages = [messages]
                    json_obj = {"messages": messages}
                    f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
        yield inputs
