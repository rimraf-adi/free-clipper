import os
import itertools
import time
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from groq import Groq, APIError, RateLimitError
from clipper.logger import log_info, log_success, log_warning, log_error

load_dotenv()

LLM_MODEL_POOL = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
]

WHISPER_MODEL_POOL = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

class GroqModelPool:
    """Manages round-robin rotation and automatic fallback across Groq models with rate-limit retry."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY environment variable is missing or empty. Please set it in .env file."
            )
        self.client = Groq(api_key=self.api_key)
        self._llm_indices = itertools.cycle(range(len(LLM_MODEL_POOL)))
        self._whisper_indices = itertools.cycle(range(len(WHISPER_MODEL_POOL)))

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Runs a chat completion using round-robin model selection with automatic failover."""

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        start_index = next(self._llm_indices)
        num_models = len(LLM_MODEL_POOL)

        errors = []
        for i in range(num_models):
            model_idx = (start_index + i) % num_models
            model_name = LLM_MODEL_POOL[model_idx]

            try:
                log_info("GroqLLM", f"Attempting LLM request with model: \033[1m{model_name}\033[0m")
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                log_success("GroqLLM", f"Success with model: \033[1m{model_name}\033[0m")
                return {
                    "model": model_name,
                    "content": content,
                    "response": response,
                }
            except (RateLimitError, APIError) as exc:
                err_str = str(exc)
                log_warning("GroqLLM", f"Model {model_name} failed: {exc}. Retrying with next model...")
                errors.append((model_name, err_str))
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    time.sleep(2)
            except Exception as exc:
                log_warning("GroqLLM", f"Unexpected error with model {model_name}: {exc}. Trying next...")
                errors.append((model_name, str(exc)))

        log_warning("GroqLLM", "All pool models rate limited. Sleeping 5 seconds before final fallback retry...")
        time.sleep(5)
        fallback_model = LLM_MODEL_POOL[0]
        try:
            response = self.client.chat.completions.create(
                model=fallback_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return {"model": fallback_model, "content": content, "response": response}
        except Exception as exc:
            errors.append((fallback_model, str(exc)))

        raise RuntimeError(f"All Groq LLM models failed. Errors: {errors}")

    def transcribe_audio(self, audio_file_path: str) -> Optional[Any]:
        """Transcribes audio using Groq Whisper API with round-robin fallback across Whisper models."""
        start_index = next(self._whisper_indices)
        num_models = len(WHISPER_MODEL_POOL)

        errors = []
        for i in range(num_models):
            model_idx = (start_index + i) % num_models
            model_name = WHISPER_MODEL_POOL[model_idx]

            try:
                log_info("GroqWhisper", f"Attempting transcription with: \033[1m{model_name}\033[0m")
                with open(audio_file_path, "rb") as file:
                    transcription = self.client.audio.transcriptions.create(
                        file=(os.path.basename(audio_file_path), file.read()),
                        model=model_name,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                    )
                log_success("GroqWhisper", f"Success with model: \033[1m{model_name}\033[0m")
                return transcription
            except Exception as exc:
                log_warning("GroqWhisper", f"Model {model_name} failed: {exc}. Trying next...")
                errors.append((model_name, str(exc)))

        log_error("GroqWhisper", f"All Groq Whisper models failed: {errors}")
        return None
