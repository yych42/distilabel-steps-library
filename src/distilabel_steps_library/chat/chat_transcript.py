from typing import TYPE_CHECKING
from distilabel.steps.base import Step, StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class FormatPlaintextChatTranscript(Step):
    """Format chat messages into a plaintext transcript format.

    `FormatPlaintextChatTranscript` is a `Step` that formats a list of chat messages into a plaintext
    transcript where each message is represented as "<role>: <content>" on a new line.

    Input columns:
        - messages (`List[Dict[str, str]]`): A list of message dictionaries, where each message has
            'role' and 'content' keys. The role can be 'user', 'assistant', or 'system'.

    Output columns:
        - transcript (`str`): A plaintext representation of the chat messages, with each message on
            a new line in the format "<role>: <content>".

    Categories:
        - format
        - chat
        - transcript

    Examples:
        Format chat messages into a plaintext transcript:

        ```python
        from distilabel.steps import FormatPlaintextChatTranscript

        format_transcript = FormatPlaintextChatTranscript()
        format_transcript.load()

        result = next(
            format_transcript.process(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "What's 2+2?"},
                            {"role": "assistant", "content": "4"}
                        ]
                    }
                ]
            )
        )
        # >>> result
        # [
        #     {
        #         'messages': [
        #             {'role': 'user', 'content': "What's 2+2?"},
        #             {'role': 'assistant', 'content': '4'}
        #         ],
        #         'transcript': 'user: What\'s 2+2?\nassistant: 4'
        #     }
        # ]
        ```
    """

    @property
    def inputs(self) -> "StepColumns":
        """List of inputs required by the `Step`, which in this case is: `messages`."""
        return ["messages"]

    @property
    def outputs(self) -> "StepColumns":
        """List of outputs generated by the `Step`, which is: `transcript`."""
        return ["transcript"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        """The `process` method formats the chat messages into a plaintext transcript format.

        Args:
            *inputs: A list of `StepInput` to be formatted.

        Yields:
            A `StepOutput` with batches of `StepInput` with added transcript format.
        """
        for input in inputs:
            for item in input:
                # Create transcript by joining messages with newlines
                item["transcript"] = "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in item["messages"]
                )

            yield input
