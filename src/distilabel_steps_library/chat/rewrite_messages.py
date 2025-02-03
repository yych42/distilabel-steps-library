from typing import Any, Callable, Dict, Generator, TYPE_CHECKING

from distilabel.steps.tasks import Task
from distilabel.steps.base import StepInput

if TYPE_CHECKING:
    from distilabel.steps.typing import StepColumns, StepOutput


class RewriteMessages(Task):
    """
    A custom Task that processes chat messages by rewriting the content of messages
    with a specific target role using an LLM.

    For each message in the input column `messages` that has the role specified by
    `target_role`, the task applies a user-provided function (`should_process_fn`)
    to the message's `content`. If the function returns True, the message is sent to
    an LLM for rewriting. The LLM prompt is composed by wrapping the original content
    in triple backticks followed by a user-provided `instructions` string. The response
    (i.e., the `generation`) from the LLM replaces the original message content. After
    all messages are processed, the task outputs the updated `messages`.

    Input columns:
        - messages (List[Dict[str, str]]): A list of message dictionaries. Each message
          should contain at least the keys:
            - role (str): The role of the message (e.g., 'assistant', 'user', etc.).
            - content (str): The text content of the message.

    Output columns:
        - messages (List[Dict[str, str]]): The updated list of messages after rewriting.

    Args:
        name (str): The name of the task.
        llm: An LLM instance with a callable `generate` method that accepts a prompt string.
        instructions (str): A string with instructions that will be appended to the prompt.
        should_process_fn (Callable[[str], bool]): A function that accepts a message's content
            and returns True if that message should be processed by the LLM, False otherwise.
        target_role (str): The role of the messages to target for rewriting.
    """

    def __init__(
        self,
        name: str,
        llm: Any,
        instructions: str,
        should_process_fn: Callable[[str], bool],
        target_role: str,
    ) -> None:
        super().__init__(name=name, llm=llm)
        self.instructions = instructions
        self.should_process_fn = should_process_fn
        self.target_role = target_role

    @property
    def inputs(self) -> "StepColumns":
        return ["messages"]

    @property
    def outputs(self) -> "StepColumns":
        return ["messages"]

    def process(self, *inputs: StepInput) -> "StepOutput":
        """
        Processes each input item by applying the provided function to determine
        if a message with the specified target role should be rewritten by the LLM.
        If so, it composes a prompt by wrapping the original content in triple backticks
        and appending the instructions, then replaces the message content with the LLM's
        generated response.

        Yields:
            A generator that yields the input batches with the updated messages.
        """
        for input_batch in inputs:
            for item in input_batch:
                messages = item.get("messages", [])
                for message in messages:
                    if message.get(
                        "role"
                    ) == self.target_role and self.should_process_fn(
                        message.get("content", "")
                    ):
                        prompt = f"```{message['content']}```\n\n{self.instructions}"
                        generation = self.llm.generate(prompt)
                        message["content"] = generation
            yield input_batch
