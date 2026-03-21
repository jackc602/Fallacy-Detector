import ollama

class OllamaClient:
    """
    Simple client for Ollama integration with easy model switching.
    """
    def __init__(self, model='llama3.2:3b'):
        self.model = model

    def set_model(self, model):
        """Switch to a different model."""
        self.model = model
        print(f"Switched to model: {self.model}")

    def chat(self, message, stream=False):
        """
        Send a message to the current model and get response.
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{'role': 'user', 'content': message}],
                stream=stream
            )
            if stream:
                # For streaming, return the generator
                return response
            else:
                return response['message']['content']
        except Exception as e:
            return f"Error: {str(e)}"

    def list_models(self):
        """List available models."""
        try:
            models = ollama.list()
            return [model['name'] for model in models['models']]
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    client = OllamaClient()
    print("Available models:", client.list_models())
    response = client.chat("Hello, can you explain what first-order logic is?")
    print("Response:", response)