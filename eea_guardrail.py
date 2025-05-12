from typing import Optional, Union, Literal
import os

import litellm
from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
import requests

LLM_GUARD_API_URL = os.environ.get("LLM_GUARD_API_URL")
LLM_GUARD_RESPONSE_PROMPT = os.environ.get("LLM_GUARD_RESPONSE_PROMPT", "Return exactly this text: 'I'm sorry, but I can't provide an answer to that question'.")

class eeaGuardrail(CustomGuardrail):
    def __init__(
        self,
        **kwargs,
    ):
        self.optional_params = kwargs

        super().__init__(**kwargs)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
        ],
        handle_error: bool=False
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Runs before the LLM API call
        Runs on only Input
        """
        _messages = data.get("messages")
        error = None
        if _messages:
            for message in _messages:
                _content = message.get("content")
                if isinstance(_content, str):
                    payload = {
                      "prompt": _content
                    }
                    llm_guard_response = requests.post(LLM_GUARD_API_URL, json=payload)
                    if llm_guard_response.ok:
                        result = llm_guard_response.json()
                        if not result.get("is_valid", False):
                            error=llm_guard_response.text
                            message["content"] = LLM_GUARD_RESPONSE_PROMPT

                    else:
                        error=llm_guard_response.text
                        message["content"] = LLM_GUARD_RESPONSE_PROMPT
        if error is not None:
            if not handle_error:
                raise ValueError(error)

        return data


class eeaGuardrail_noerror(eeaGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type, True)
