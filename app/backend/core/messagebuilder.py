from core.modelhelper import num_tokens_from_messages
import numpy as np

class MessageBuilder:
    """
      A class for building and managing messages in a chat conversation.
      Attributes:
          message (list): A list of dictionaries representing chat messages.
          model (str): The name of the ChatGPT model.
          token_count (int): The total number of tokens in the conversation.
      Methods:
          __init__(self, system_content: str, chatgpt_model: str): Initializes the MessageBuilder instance.
          append_message(self, role: str, content: str, index: int = 1): Appends a new message to the conversation.
      """

    def __init__(self, system_content: str, chatgpt_model: str, max_tokens=None):
        self.messages = [{'role': 'system', 'content': system_content}]
        self.model = chatgpt_model
        self.token_length = num_tokens_from_messages(
            self.messages[-1], self.model)
        self.token_limit = max_tokens if max_tokens else np.inf

    def append_message(self, role: str, content: str, index: int = 1):
        self.messages.insert(index, {'role': role, 'content': content})
        self.token_length += num_tokens_from_messages(
            self.messages[index], self.model)
        if self.token_length > self.token_limit:
            del self.messages[index]
        