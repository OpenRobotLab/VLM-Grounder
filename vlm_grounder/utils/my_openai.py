import base64
import os
import random
import time

import openai
from httpx import HTTPStatusError, ReadTimeout, RemoteProtocolError, Timeout
from openai import AzureOpenAI, OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessage


class StreamError(Exception):
    """
    Error for stream request not completed
    """

    def __init__(self, message):
        self.message = message
        super().__init__(message)


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 40,
    max_delay: int = 30,
    errors: tuple = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.BadRequestError,
        HTTPStatusError,
        ReadTimeout,
        RemoteProtocolError,
        StreamError,
    ),
):
    """
    Retry a function with exponential backoff.
    """

    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                # * print the error info
                num_retries += 1
                if num_retries > max_retries:
                    print(
                        f"[OPENAI] Encounter error of type: {type(e).__name__}, message: {e}"
                    )
                    raise Exception(
                        f"[OPENAI] Maximum number of retries ({max_retries}) exceeded."
                    )

                print(
                    f"[OPENAI] Retrying after {delay} seconds due to error of type: {type(e).__name__}, message: {e}"
                )
                delay *= exponential_base * (1 + jitter * random.random())
                time.sleep(min(delay, max_delay))
            except OpenAIError as e:
                print(
                    f"[OPENAI] Encounter error of type: {type(e).__name__}, message: {e}"
                )
                raise e
            except Exception as e:
                print(
                    f"[OPENAI] Unkown error of type: {type(e).__name__}, message: {e}"
                )
                raise e

    return wrapper


GPT_PRICES = {
    "gpt-4o": {"price_1k_prompt_tokens": 0.005, "price_1k_completion_tokens": 0.015},
    "deepseek-chat": {"price_1k_prompt_tokens": 0.0, "price_1k_completion_tokens": 0.0},
    "gpt-3.5-turbo-0613": {
        "price_1k_prompt_tokens": 0.0015,
        "price_1k_completion_tokens": 0.002,
    },
    "gpt-3.5-turbo-1106": {
        "price_1k_prompt_tokens": 0.0010,
        "price_1k_completion_tokens": 0.002,
    },
    "gpt-3.5-turbo-0125": {
        "price_1k_prompt_tokens": 0.0005,
        "price_1k_completion_tokens": 0.0015,
    },
    "gpt-4-0613": {"price_1k_prompt_tokens": 0.03, "price_1k_completion_tokens": 0.06},
    "gpt-4-1106-preview": {
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03,
    },
    "gpt-4-0125-preview": {
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03,
    },
    "gpt-4-turbo-2024-04-09": {
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03,
    },
    "gpt-4-vision-preview": {
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03,
    },
    "gpt-4-1106-vision-preview": {
        "price_1k_prompt_tokens": 0.01,
        "price_1k_completion_tokens": 0.03,
    },
    "gpt-4o-2024-05-13": {
        "price_1k_prompt_tokens": 0.005,
        "price_1k_completion_tokens": 0.015,
    },
}


class OpenAIGPT:
    def __init__(
        self,
        model="gpt-3.5-turbo-0613",
        temperature=1,
        top_p=1,
        max_tokens=2048,
        api_key=None,
        base_url=None,
        use_Azure=False,
        **kwargs,
    ) -> None:
        if use_Azure:
            model = "gpt-4o"
        setup_openai(model)  # OpenAI api key is set in the os env
        self.default_chat_parameters = {
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            **kwargs,
        }
        # * price
        self.api_key = api_key
        self.base_url = base_url
        self.price_1k_prompt_tokens = GPT_PRICES[model]["price_1k_prompt_tokens"]
        self.price_1k_completion_tokens = GPT_PRICES[model][
            "price_1k_completion_tokens"
        ]
        if use_Azure:
            # * please set AzureOpenAI key, api version and endpoint
            self.client = AzureOpenAI(
                api_key=api_key,
                api_version="2024-04-01-preview",
                azure_endpoint="https://gpt-4o-pjm.openai.azure.com/",
            )
        else:
            self.client = OpenAI(api_key=api_key, base_url=base_url)

    @retry_with_exponential_backoff
    def safe_chat_complete(
        self, messages, content_only=True, return_cost=True, **kwargs
    ):
        """
        GPT request.

        Args:
            messages: list of messages
            content_only: return only the content of the response if True, else return the full response
            return_cost: return the token cost, default is true
        """
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)
        response = self.client.chat.completions.create(
            messages=messages, **chat_parameters
        )

        if content_only:
            result = {"content": response.choices[0].message.content}
        else:
            result = {"content": response}

        if return_cost:
            result["cost"] = self.get_costs(response)
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens

        return result

    @retry_with_exponential_backoff
    def stream_chat(self, messages, **kwargs):
        """
        Use stream request, more stable with long messages.
        """
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)
        response = self.client.chat.completions.create(
            messages=messages,
            stream=True,  # this time, we set stream=True
            **chat_parameters,
        )

        content = ""
        print("[stream]Receiving...")

        for chunk in response:
            content += (
                chunk.choices[0].delta.content
                if chunk.choices[0].delta.content is not None
                else ""
            )

        if chunk.choices[0].finish_reason != "stop":
            print(f"[stream]Error occured: {chunk.choices[0].finish_reason}")
            raise StreamError(chunk.choices[0].finish_reason + f"{len(content)}")

        print("[stream]Done.")

        result = {"content": content}

        # not implemented
        result["cost"] = 0
        result["prompt_tokens"] = 0
        result["completion_tokens"] = 0

        return result

    @retry_with_exponential_backoff
    def safe_chat_complete_with_raw_response(
        self, messages, content_only=True, return_cost=True, **kwargs
    ):
        """
        GPT request with raw response head, include openai process time.

        Args:
            messages: list of messages
            content_only: return only the content of the response if True, else return the full response
            return_cost: return the token cost, default is true
        """
        chat_parameters = self.default_chat_parameters.copy()
        if len(kwargs) > 0:
            chat_parameters.update(**kwargs)

        raw_response = self.client.chat.completions.with_raw_response.create(
            messages=messages, timeout=Timeout(None, connect=5.0), **chat_parameters
        )

        headers = raw_response.headers
        print("[OPENAI]Openai processing time:", headers["openai-processing-ms"])
        response = raw_response.parse()

        if content_only:
            result = {"content": response.choices[0].message.content}
        else:
            result = {"content": response}

        if return_cost:
            result["cost"] = self.get_costs(response)
            result["prompt_tokens"] = response.usage.prompt_tokens
            result["completion_tokens"] = response.usage.completion_tokens

        return result

    def single_round_chat(self, system_prompt, input):
        """
        Return:
            result['content']: str
            result['cost']: cost
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input},
        ]

        return self.safe_chat_complete(messages)

    def get_costs(self, response):
        total_prompt_tokens = response.usage.prompt_tokens
        total_completion_tokens = response.usage.completion_tokens
        # print("total prompt tokens:", total_prompt_tokens)
        # print("total completion tokens:", total_completion_tokens)

        cost = (
            total_prompt_tokens * self.price_1k_prompt_tokens / 1000
            + total_completion_tokens * self.price_1k_completion_tokens / 1000
        )
        return cost

    def load_and_encoder_image_for_gpt(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        except Exception as e:
            raise Exception(f"Error loading image: {e}")

        return base64_image


def setup_openai(model_name, is_eval=False):
    # Setup OpenAI API Key
    print("[OPENAI] Setting OpenAI api_key...")
    # * openai.api_key = os.getenv('OPENAI_API_KEY')

    # * OPENAI_API_KEY
    api_key = "your_api_key"

    os.environ["OPENAI_API_KEY"] = api_key

    print(f"[OPENAI] OpenAI organization: {openai.organization}")
    # * print http_proxy
    print(
        f"[OPENAI] http_proxy: {os.environ['HTTP_PROXY']}. https_proxy: {os.environ['HTTPS_PROXY']}"
    )
    print(f"[OPENAI] Using MODEL: {model_name}")


def filter_image_url(message_his):
    """
    filter all image data in messages
    """
    filtered_message_his = []
    for message in message_his:
        if isinstance(message, dict):
            if isinstance(message["content"], str):
                filtered_message_his.append(
                    {"role": message["role"], "content": [message["content"]]}
                )
            elif isinstance(message["content"], list):
                filtered_message_his.append(
                    {
                        "role": message["role"],
                        "content": [
                            item["text"]
                            for item in message["content"]
                            if item["type"] == "text"
                        ],
                    }
                )
        elif isinstance(message, ChatCompletionMessage):
            filtered_message_his.append(
                {"role": message.role, "content": [message.content]}
            )
        else:
            raise Exception(f"Unknown message type: {type(message)}")

    return filtered_message_his


def print_message_his(message_his):
    """
    Print history messages
    """
    filtered_message_his = filter_image_url(message_his)

    for message in filtered_message_his:
        print(f"{message['role']}:")
        for content in message["content"]:
            print(content)
        print("")


# * format of prompts
# * images:
# messages = [
#         {
#             "role": "system",
#             "content": self.system_prompt
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": gpt_input
#                 },
#                 *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}", "detail": "high"}}, base64Frames),
#             ],
#         }
# ]

# * texts-only:
# messages = [
#         {
#             "role": "system",
#             "content": self.system_prompt
#         },
#         {
#             "role": "user",
#             "content": gpt_input
#         }
# ]
