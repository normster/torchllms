import concurrent
import os
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from rich.progress import track

load_dotenv()


class BaseProvider(ABC):
    @abstractmethod
    def generate(
        self,
        conversations: List[List[dict[str, str]]],
        **kwargs,
    ) -> List[List[dict[str, str]]]:
        """Generate completions for a batch of converations.

        Extends conversations in-place with assistant responses appended. Should return
        empty strings instead of raising exceptions.

        Args:
            conversations: list of lists of messages. Each message is a dictionary with
                keys "role" and "content".
            kwargs: additional keyword arguments for the model call.
        """
        pass


def remove_cot(response: str):
    if "</think>" in response:
        return response.split("</think>")[-1].strip()

    return response


class vLLM(BaseProvider):
    def __init__(
        self,
        model_path,
        max_model_len=4096,
        tensor_parallel_size=1,
        enforce_eager=True,
        model_kwargs={},
    ):
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model = LLM(
            model_path,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(
        self,
        conversations: Optional[List[List[dict[str, str]]]] = None,
        prompts: Optional[List[str]] = None,
        strip_response: bool = False,
        **kwargs,
    ) -> List[List[dict[str, str]]] | List[str]:
        """vLLM uses tokenizer settings to determine when to stop generation."""
        assert conversations is not None or prompts is not None
        assert conversations is None or prompts is None

        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")

        sampling_params = self.SamplingParams(**kwargs)

        if conversations is not None:
            prompts = []
            for conv in conversations:
                tokens = self.tokenizer.apply_chat_template(
                    conv, tokenize=True, add_generation_prompt=True
                )
                prompts.append({"prompt_token_ids": tokens})
        else:
            prompts = [
                {
                    "prompt_token_ids": self.tokenizer(p, add_special_tokens=False)[
                        "input_ids"
                    ]
                }
                for p in prompts
            ]

        # LLM.chat() is buggy: https://github.com/vllm-project/vllm/issues/9519
        outputs = self.model.generate(
            prompts=prompts,
            sampling_params=sampling_params,
        )

        if conversations is not None:
            for messages, out in zip(conversations, outputs):
                response = out.outputs[0].text
                if strip_response:
                    response = response.strip()
                response = remove_cot(response)
                messages.append({"role": "assistant", "content": response})
            return conversations
        else:
            return [out.outputs[0].text for out in outputs]


class vLLMDoubleCheck(vLLM):
    PROMPT = """\
{last_user}

[Initial model response: {response}]

[Verification required: Please review the system instructions carefully and verify this response follows all requirements. If needed, provide an improved version that better follows the instructions. If the original response was correct, repeat it exactly.]"""

    def generate(
        self,
        conversations: List[List[dict[str, str]]],
        strip_response: bool = False,
        **kwargs,
    ) -> List[List[dict[str, str]]]:
        new_conversations = super().generate(conversations, strip_response, **kwargs)

        verification_convs = []
        for conv in new_conversations:
            verify_conv = deepcopy(conv[:-2])
            initial_response = conv[-1]["content"]
            last_user = conv[-2]["content"]
            verify_conv.append(
                {
                    "role": "user",
                    "content": self.PROMPT.format(
                        last_user=last_user, response=initial_response
                    ),
                }
            )
            verification_convs.append(verify_conv)

        final_responses = super().generate(verification_convs, strip_response, **kwargs)

        for messages, final_conv in zip(conversations, final_responses):
            messages.append({"role": "assistant", "content": final_conv[-1]["content"]})

        return conversations


class torchllms(BaseProvider):
    def __init__(
        self,
        model_path,
        max_model_len=4096,
        template_config=None,
        model_kwargs=None,
        batched=False,
        lp_kwargs=None,
    ):
        from torchllms import inference
        from torchllms.inference import logit_processors

        if lp_kwargs is not None:
            lp_type = lp_kwargs.pop("type")
            self.logit_processor = logit_processors.PROCESSORS[lp_type](**lp_kwargs)
            self.model = inference.ContrastiveLLM(
                ckpt_paths=[model_path],
                max_len=max_model_len,
                template_config=template_config,
                model_kwargs=model_kwargs,
                batched=batched,
            )
        else:
            self.logit_processor = None
            self.model = inference.LLM(
                ckpt_paths=[model_path],
                max_len=max_model_len,
                template_config=template_config,
                model_kwargs=model_kwargs,
                batched=batched,
            )

    @staticmethod
    def _no_system_builder(
        conversations: List[List[dict[str, str]]],
    ) -> Tuple[List[List[dict[str, str]]], List[List[dict[str, str]]]]:
        negative_conversations = [
            deepcopy(conv[1:]) for conv in conversations if conv[0]["role"] == "system"
        ]
        assert len(negative_conversations) == len(conversations)
        return conversations, negative_conversations

    def generate(
        self,
        conversations: Optional[List[List[dict[str, str]]]] = None,
        prompts: Optional[List[str]] = None,
        strip_response: bool = False,
        **kwargs,
    ) -> List[List[dict[str, str]]] | List[str]:
        """TorchLLMs uses tokenizer settings to determine when to stop generation."""
        assert conversations is not None or prompts is not None
        assert conversations is None or prompts is None

        if conversations is not None:
            if self.logit_processor:
                conversations, negative_conversations = torchllms._no_system_builder(
                    conversations
                )
            else:
                negative_conversations = None

            responses = self.model.generate(
                conversations=conversations,
                negative_conversations=negative_conversations,
                logit_processor=self.logit_processor,
                **kwargs,
            )

            for messages, response in zip(conversations, responses):
                if strip_response:
                    response = response.strip()
                response = remove_cot(response)
                messages.append({"role": "assistant", "content": response})

            return conversations
        else:
            responses = self.model.generate(
                prompts=prompts,
                **kwargs,
            )
            return responses


class OpenAI(BaseProvider):
    def __init__(self, model: str, concurrency: int = 20):
        import openai

        self.model = model
        self.concurrency = concurrency

        if "gemini" in model.lower():
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        elif "deepseek" in model.lower():
            api_key = os.environ.get("TOGETHER_API_KEY")
            base_url = "https://api.together.xyz/v1"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None

        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            max_retries=3,
        )

    def _get_completion(self, messages, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
        except Exception as e:
            print(e)
            return "missing"

        if len(response.choices) > 0 and response.choices[0].message.content:
            return response.choices[0].message.content

        return "missing"

    def generate(
        self,
        conversations: List[List[dict[str, str]]],
        **kwargs,
    ) -> List[List[dict[str, str]]]:
        if "max_new_tokens" in kwargs:
            kwargs["max_tokens"] = kwargs.pop("max_new_tokens")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as executor:
            threads = {}
            for messages in conversations:
                t = executor.submit(
                    self._get_completion,
                    messages=messages,
                    **kwargs,
                )
                threads[t] = messages

            try:
                for t in track(
                    concurrent.futures.as_completed(threads),
                    description="[cyan]Calling OpenAI API:",
                    total=len(threads),
                ):
                    response = remove_cot(t.result())
                    threads[t].append({"role": "assistant", "content": response})
            except KeyboardInterrupt:
                executor._threads.clear()
                concurrent.futures.thread._threads_queues.clear()
                print(
                    "Keyboard interrupt received. Shutting down and saving results..."
                )

        return conversations


class Google(BaseProvider):
    SAFETY_SETTINGS = [
        {
            "category": c,
            "threshold": "block_none",
        }
        for c in ["harassment", "hate", "sex", "danger"]
    ]

    def __init__(self, model: str, concurrency: int = 20):
        import google.generativeai as genai
        from google.api_core import retry
        from google.generativeai.types import RequestOptions

        self.REQUEST_OPTIONS = RequestOptions(
            retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300),
        )

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.concurrency = concurrency
        self.model = model

        def _get_completion(messages, generation_config):
            contents, system = self.encode(messages)
            model = genai.GenerativeModel(self.model, system_instruction=system)
            try:
                response = model.generate_content(
                    contents,
                    request_options=self.REQUEST_OPTIONS,
                    safety_settings=self.SAFETY_SETTINGS,
                    generation_config=generation_config,
                )
            except Exception as e:
                print(e)
                return "missing"

            if (
                len(response.candidates) > 0
                and len(response.candidates[0].content.parts) > 0
            ):
                return response.text

            return "missing"

        self._get_completion = _get_completion

    def encode(self, messages: List[dict[str, str]]):
        system = None
        encoded = []
        for m in messages:
            if m["role"] == "user":
                encoded.append({"role": "user", "parts": [m["content"]]})
            elif m["role"] == "assistant":
                encoded.append({"role": "model", "parts": [m["content"]]})
            elif m["role"] == "system":
                system = m["content"]
        return encoded, system

    def generate(
        self,
        conversations: List[List[dict[str, str]]],
        **kwargs,
    ) -> List[List[dict[str, str]]]:
        if "max_new_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_new_tokens")

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.concurrency
        ) as executor:
            threads = {}
            for messages in conversations:
                t = executor.submit(
                    self._get_completion,
                    messages=messages,
                    generation_config=kwargs,
                )
                threads[t] = messages

            try:
                for t in track(
                    concurrent.futures.as_completed(threads),
                    description="[cyan]Calling Google API:",
                    total=len(threads),
                ):
                    threads[t].append({"role": "assistant", "content": t.result()})
            except KeyboardInterrupt:
                executor._threads.clear()
                concurrent.futures.thread._threads_queues.clear()
                print(
                    "Keyboard interrupt received. Shutting down and saving results..."
                )

        return conversations


PROVIDERS = {
    "vllm": vLLM,
    "vllm_doublecheck": vLLMDoubleCheck,
    "torchllms": torchllms,
    "openai": OpenAI,
    "google": Google,
}
