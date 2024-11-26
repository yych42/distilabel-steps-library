from typing import TYPE_CHECKING
from distilabel.steps.base import Step, StepInput
from pydantic import Field

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput

class InsertMessage(Step):
    """Insert a new message into the chat messages at a specified index.

    `InsertMessage` is a `Step` that takes existing chat messages and inserts a new message
    at the specified index with the given role and content.

    Attributes:
        index (int): The position where the new message should be inserted (0-based).
            If negative, counts from the end like Python list indexing.
        role (str): The role for the inserted message ('user', 'assistant', or 'system').

    Input columns:
        - messages (`List[Dict[str, str]]`): A list of message dictionaries
        - content (`str`): The content for the message to be inserted

    Output columns:
        - messages (`List[Dict[str, str]]`): The modified list of messages with the new
            message inserted

    Categories:
        - format
        - chat
        - message

    Examples:
        Insert a system message at the beginning:

        ```python
        from distilabel.steps import InsertMessage

        insert = InsertMessage(index=0, role="system")
        insert.load()

        result = next(
            insert.process(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "Hi"},
                            {"role": "assistant", "content": "Hello"}
                        ],
                        "content": "Be helpful and concise"
                    }
                ]
            )
        )
        # >>> result
        # [
        #     {
        #         'messages': [
        #             {'role': 'system', 'content': 'Be helpful and concise'},
        #             {'role': 'user', 'content': 'Hi'},
        #             {'role': 'assistant', 'content': 'Hello'}
        #         ],
        #         'content': 'Be helpful and concise'
        #     }
        # ]
        ```
    """

    index: int = Field(
        ...,
        description="The position where the new message should be inserted (0-based)"
    )
    role: str = Field(
        ...,
        description="The role for the inserted message ('user', 'assistant', or 'system')"
    )

    @property
    def inputs(self) -> "StepColumns":
        """List of inputs required by the `Step`: `messages` and `content`."""
        return ["messages", "content"]

    @property
    def outputs(self) -> "StepColumns":
        """List of outputs generated by the `Step`: `messages`."""
        return ["messages"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method inserts a new message into the existing messages.

        Args:
            *inputs: A list of `StepInput` containing messages and content to process.

        Yields:
            A `StepOutput` with batches of `StepInput` with modified messages.
        """
        for input in inputs:
            for item in input:
                # Create the new message
                new_message = {"role": self.role, "content": item["content"]}

                # Create a new list with the inserted message
                messages = item["messages"].copy()
                messages.insert(self.index, new_message)

                # Update the messages in the item
                item["messages"] = messages

            yield input
