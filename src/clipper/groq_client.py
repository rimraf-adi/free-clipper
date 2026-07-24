import os
import time
from typing import List, Dict, Any, Optional, cast
from dotenv import load_dotenv
from groq import Groq, APIError, RateLimitError
from .logger import log_info, log_success, log_warning, log_error

load_dotenv()

LLM_MODEL_POOL = [
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "llama-3.3-70b-versatile",
]

WHISPER_MODEL_POOL = [
    "whisper-large-v3-turbo",
    "whisper-large-v3",
]

def load_all_api_keys() -> List[str]:
    """Harvests all Groq API keys from environment variables (LLM_API_KEY_1..15, GROQ_API_KEY_1..15, GROQ_API_KEY)."""
    keys = []
    
    # 1. Check LLM_API_KEY_1 .. LLM_API_KEY_15
    for i in range(1, 16):
        val = os.getenv(f"LLM_API_KEY_{i}")
        if val and val.strip() and val.strip() not in keys:
            keys.append(val.strip())
            
    # 2. Check GROQ_API_KEY_1 .. GROQ_API_KEY_15
    for i in range(1, 16):
        val = os.getenv(f"GROQ_API_KEY_{i}")
        if val and val.strip() and val.strip() not in keys:
            keys.append(val.strip())
            
    # 3. Check default GROQ_API_KEY
    default_key = os.getenv("GROQ_API_KEY")
    if default_key and default_key.strip() and default_key.strip() not in keys:
        keys.append(default_key.strip())
        
    return keys

class GroqModelPool:
    """Intelligent Multi-Key & Multi-Model Router for Groq API with 2D rotation matrix and dynamic rate-limit cooldown tracking."""

    def __init__(self, api_keys: Optional[List[str]] = None, api_key: Optional[str] = None):
        if api_keys:
            raw_keys = api_keys
        elif api_key:
            raw_keys = [api_key]
        else:
            raw_keys = load_all_api_keys()
            
        if not raw_keys:
            raise ValueError(
                "No Groq API keys found in environment. Please set GROQ_API_KEY or LLM_API_KEY_1..10 in .env file."
            )
            
        self.api_keys = raw_keys
        self.clients = [Groq(api_key=k) for k in self.api_keys]
        
        self._key_index = 0
        self._model_index = 0
        self._whisper_key_index = 0
        self._whisper_model_index = 0
        
        # Cooldown tracker: maps (key_index, model_name) -> timestamp until cooldown expires
        self._cooldowns: Dict[str, float] = {}

        log_info("GroqRouter", f"Initialized Intelligent Router with \033[1m{len(self.api_keys)}\033[0m API key(s) and \033[1m{len(LLM_MODEL_POOL)}\033[0m LLM model(s).")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> Dict[str, Any]:
        """Executes LLM request using 2D (Key x Model) rotation matrix with instant rate-limit failover."""
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + list(messages)

        now = time.time()
        num_keys = len(self.api_keys)
        num_models = len(LLM_MODEL_POOL)
        total_combos = num_keys * num_models

        start_k_idx = self._key_index
        start_m_idx = self._model_index

        errors = []
        for i in range(total_combos):
            k_idx = (start_k_idx + i) % num_keys
            m_idx = (start_m_idx + (i // num_keys)) % num_models
            
            model_name = LLM_MODEL_POOL[m_idx]
            client = self.clients[k_idx]
            combo_key = f"{k_idx}_{model_name}"

            # Skip if currently on cooldown
            if self._cooldowns.get(combo_key, 0.0) > now:
                continue

            try:
                log_info("GroqLLM", f"Request via Key #{k_idx+1}/{num_keys} with model: \033[1m{model_name}\033[0m")
                response = client.chat.completions.create(
                    model=model_name,
                    messages=cast(Any, messages),
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                log_success("GroqLLM", f"Success with Key #{k_idx+1} & Model: \033[1m{model_name}\033[0m")
                
                # Advance starting key index for next request
                self._key_index = (k_idx + 1) % num_keys
                self._model_index = m_idx
                
                return {
                    "model": model_name,
                    "content": content,
                    "key_index": k_idx + 1,
                    "response": response,
                }
            except (RateLimitError, APIError) as exc:
                err_str = str(exc)
                log_warning("GroqLLM", f"Key #{k_idx+1} on {model_name} failed: {exc}. Rotating to next Key/Model...")
                errors.append((k_idx + 1, model_name, err_str))
                
                if "rate_limit" in err_str.lower() or "429" in err_str:
                    self._cooldowns[combo_key] = time.time() + 60.0
            except Exception as exc:
                log_warning("GroqLLM", f"Unexpected error with Key #{k_idx+1} on {model_name}: {exc}. Rotating...")
                errors.append((k_idx + 1, model_name, str(exc)))

        log_warning("GroqRouter", "All Key x Model combinations on cooldown. Performing short backoff retry...")
        time.sleep(3.0)
        
        fallback_client = self.clients[0]
        fallback_model = LLM_MODEL_POOL[0]
        try:
            response = fallback_client.chat.completions.create(
                model=fallback_model,
                messages=cast(Any, messages),
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return {"model": fallback_model, "content": content, "key_index": 1, "response": response}
        except Exception as exc:
            errors.append((1, fallback_model, str(exc)))

        raise RuntimeError(f"All Groq API Keys and Models failed. Attempted errors: {errors}")

    def transcribe_audio(self, audio_file_path: str) -> Optional[Any]:
        """Transcribes audio using Groq Whisper API rotating across keys and Whisper models."""
        num_keys = len(self.api_keys)
        num_models = len(WHISPER_MODEL_POOL)
        total_combos = num_keys * num_models

        start_k_idx = self._whisper_key_index
        start_m_idx = self._whisper_model_index

        errors = []
        for i in range(total_combos):
            k_idx = (start_k_idx + i) % num_keys
            m_idx = (start_m_idx + (i // num_keys)) % num_models
            
            model_name = WHISPER_MODEL_POOL[m_idx]
            client = self.clients[k_idx]

            try:
                log_info("GroqWhisper", f"Attempting transcription via Key #{k_idx+1}/{num_keys} with: \033[1m{model_name}\033[0m")
                with open(audio_file_path, "rb") as file:
                    transcription = client.audio.transcriptions.create(
                        file=(os.path.basename(audio_file_path), file.read()),
                        model=model_name,
                        response_format="verbose_json",
                        timestamp_granularities=["word", "segment"],
                    )
                log_success("GroqWhisper", f"Success with Key #{k_idx+1} & Model: \033[1m{model_name}\033[0m")
                
                self._whisper_key_index = (k_idx + 1) % num_keys
                self._whisper_model_index = m_idx
                return transcription
            except Exception as exc:
                log_warning("GroqWhisper", f"Key #{k_idx+1} on {model_name} failed: {exc}. Trying next...")
                errors.append((k_idx + 1, model_name, str(exc)))

        log_error("GroqWhisper", f"All Groq Whisper keys and models failed: {errors}")
        return None
