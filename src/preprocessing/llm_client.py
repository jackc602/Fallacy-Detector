import os
import time
import random
from dotenv import load_dotenv
from pathlib import Path
import ollama
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / ".env")

# Substrings that identify retryable API errors
_RETRYABLE = frozenset([
    '503', '429', '500', 'quota', 'unavailable', 'deadline',
    'timeout', 'rate limit', 'resource exhausted', 'service unavailable',
    'internal server error', 'connection', 'reset by peer',
])


class LLMClient:
    def __init__(self, backend='ollama', model=None, api_key=None):
        self.backend = backend.lower()
        self.api_key = api_key or os.environ.get(self._env_key())

        if self.backend == 'ollama':
            self.model = model or 'mistral:latest'
        elif self.backend == 'openai':
            self.model = model or 'gpt-4o'
        elif self.backend == 'gemini':
            self.model_name = model or 'gemini-2.5-flash'
            self._genai_client = genai.Client(
                vertexai=True,
                project=os.getenv("PROJECT_ID"),
                location="us-east1",
            )
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}. Use 'ollama', 'openai', or 'gemini'.")

    def _env_key(self):
        return {'ollama': '', 'openai': 'OPENAI_API_KEY', 'gemini': 'GEMINI_API_KEY'}.get(self.backend, '')

    def _with_retry(self, fn, max_retries=5):
        """Call fn(), retrying with exponential backoff on transient errors."""
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                err_str = (type(e).__name__ + ' ' + str(e)).lower()
                is_retryable = any(code in err_str for code in _RETRYABLE)
                last_exc = e
                if not is_retryable or attempt == max_retries - 1:
                    raise
                wait = min(60.0, (2 ** attempt) + random.uniform(0, 1))
                print(f"  [retry {attempt + 1}/{max_retries - 1}] {type(e).__name__}: waiting {wait:.1f}s")
                time.sleep(wait)
        raise last_exc  # unreachable, but satisfies type checkers

    def chat(self, message, system_prompt=None):
        if self.backend == 'ollama':
            return self._chat_ollama(message, system_prompt)
        elif self.backend == 'openai':
            return self._chat_openai(message, system_prompt)
        elif self.backend == 'gemini':
            return self._chat_gemini(message, system_prompt)

    def _chat_ollama(self, message, system_prompt=None):
        msgs = []
        if system_prompt:
            msgs.append({'role': 'system', 'content': system_prompt})
        msgs.append({'role': 'user', 'content': message})

        def call():
            response = ollama.chat(model=self.model, messages=msgs)
            return response['message']['content']

        return self._with_retry(call)

    def _chat_openai(self, message, system_prompt=None):
        import httpx
        msgs = []
        if system_prompt:
            msgs.append({'role': 'system', 'content': system_prompt})
        msgs.append({'role': 'user', 'content': message})

        def call():
            resp = httpx.post(
                'https://api.openai.com/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                json={'model': self.model, 'messages': msgs, 'temperature': 0.1},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content']

        return self._with_retry(call)

    def _chat_gemini(self, message, system_prompt=None):
        """Call Gemini via Vertex AI using the google-genai SDK."""
        config = types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=2048,
            system_instruction=system_prompt,
        )

        def call():
            response = self._genai_client.models.generate_content(
                model=self.model_name,
                contents=message,
                config=config,
            )
            candidate = response.candidates[0]
            finish = candidate.finish_reason.name if candidate.finish_reason else 'UNKNOWN'
            if finish == 'MAX_TOKENS':
                raise RuntimeError(
                    f"Response truncated (MAX_TOKENS): partial output was {len(response.text)} chars. "
                    f"Increase max_output_tokens or shorten the input."
                )
            return response.text

        return self._with_retry(call)


if __name__ == '__main__':
    client = LLMClient(backend='gemini', model='gemini-2.5-flash')
    print(client.chat("What is first-order logic? One sentence."))