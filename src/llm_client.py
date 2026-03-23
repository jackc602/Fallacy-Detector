import json
import os
import ollama


class LLMClient:
    def __init__(self, backend='ollama', model=None, api_key=None):
        self.backend = backend.lower()
        self.api_key = api_key or os.environ.get(self._env_key())

        if self.backend == 'ollama':
            self.model = model or 'mistral:latest'
        elif self.backend == 'openai':
            self.model = model or 'gpt-4o'
        elif self.backend == 'gemini':
            self.model = model or 'gemini-2.0-flash'
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'ollama', 'openai', or 'gemini'.")



    def _env_key(self):
        return {
            'ollama': '',
            'openai': 'OPENAI_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
        }.get(self.backend, '')



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
        response = ollama.chat(model=self.model, messages=msgs)
        return response['message']['content']



    def _chat_openai(self, message, system_prompt=None):
        import httpx
        msgs = []
        if system_prompt:
            msgs.append({'role': 'system', 'content': system_prompt})
        msgs.append({'role': 'user', 'content': message})

        resp = httpx.post(
            'https://api.openai.com/v1/chat/completions',
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': self.model,
                'messages': msgs,
                'temperature': 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()['choices'][0]['message']['content']



    def _chat_gemini(self, message, system_prompt=None):
        import httpx
        url = (
            f'https://generativelanguage.googleapis.com/v1beta/models/'
            f'{self.model}:generateContent?key={self.api_key}'
        )
        contents = []
        if system_prompt:
            contents.append({'role': 'user', 'parts': [{'text': system_prompt}]})
            contents.append({'role': 'model', 'parts': [{'text': 'Understood.'}]})
        contents.append({'role': 'user', 'parts': [{'text': message}]})

        resp = httpx.post(
            url,
            headers={'Content-Type': 'application/json'},
            json={
                'contents': contents,
                'generationConfig': {'temperature': 0.2},
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data['candidates'][0]['content']['parts'][0]['text']




if __name__ == '__main__':
    # Quick test — change backend/key as needed
    client = LLMClient(backend='ollama', model='mistral:latest')
    print(client.chat("What is first-order logic? One sentence."))