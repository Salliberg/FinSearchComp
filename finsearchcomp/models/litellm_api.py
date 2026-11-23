"""Judge client for any LiteLLM provider/model string (e.g. openai/gpt-5.4).

Same call shape as models/openai_api.py: history + user prompt, max_tokens from
config chat_defaults, up to 3 attempts with exponential backoff (4-10 s). On a
final failure it returns {"error": ...}, which eval.py records as an error score.
"""
import os
import sys

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config.config_wrapper import get_config_wrapper
from logger import get_logger

logger = get_logger(__name__)


class LiteLLMChat:
    def __init__(self, model_name):
        config = get_config_wrapper()
        self.max_tokens = config.config['chat_defaults']['max_tokens']
        self.model_name = model_name

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
    def _complete(self, messages):
        response = litellm.completion(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content

    def __call__(self, prompts, histories, **kwargs):
        responses = []
        for prompt, history in zip(prompts, histories):
            messages = list(history or [])
            messages.append({"role": "user", "content": prompt})
            try:
                responses.append(self._complete(messages))
            except Exception as e:
                logger.error("LiteLLM request failed: %s", e)
                responses.append({"error": str(e)})
        return responses


def load_model(model_name, **kwargs):
    return LiteLLMChat(model_name)
