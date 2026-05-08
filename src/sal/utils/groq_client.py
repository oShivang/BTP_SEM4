import os
import time
import logging
from itertools import cycle
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)

class GroqClient:
    def __init__(self, keys=None, model="llama-3.1-8b-instant"):
        """
        Initialize the GroqClient with a Round-Robin API key manager.
        Requires exactly 2 Groq API keys either via `keys` param or env variables `GROQ_API_KEY_1` and `GROQ_API_KEY_2`.
        """
        if keys is None:
            key1 = os.getenv("GROQ_API_KEY_1")
            key2 = os.getenv("GROQ_API_KEY_2")
            if not key1 or not key2:
                raise ValueError("Must provide exactly 2 Groq API keys either via keys param or GROQ_API_KEY_1 and GROQ_API_KEY_2 env vars.")
            keys = [key1, key2]
        
        if len(keys) != 2:
            raise ValueError("Exactly 2 Groq API keys are required for Round-Robin.")
            
        self.clients = [Groq(api_key=k) for k in keys]
        self.client_cycle = cycle(self.clients)
        self.model = model
        
    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=1.0, stop=None):
        """
        Generates a completion using Groq with exponential backoff on 429 Too Many Requests.
        """
        client = next(self.client_cycle)
        backoff = 1.0
        
        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                logger.warning(f"Rate limit hit. Retrying in {backoff} seconds... Error: {e}")
                time.sleep(backoff)
                backoff *= 2
                if backoff > 60:
                    backoff = 60
            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e):
                    logger.warning(f"Rate limit hit (General Exception). Retrying in {backoff} seconds... Error: {e}")
                    time.sleep(backoff)
                    backoff *= 2
                    if backoff > 60:
                        backoff = 60
                else:
                    raise e
                    
    def generate_batch(self, messages_batch, max_tokens=1024, temperature=0.7, top_p=1.0, stop=None):
        """
        Convenience method to generate completions for a batch of messages sequentially.
        """
        results = []
        for messages in messages_batch:
            result = self.generate(messages, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=stop)
            results.append(result)
        return results
