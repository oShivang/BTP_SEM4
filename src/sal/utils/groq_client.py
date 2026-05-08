import os
import time
import random
import logging
from itertools import cycle
from groq import Groq, RateLimitError

logger = logging.getLogger(__name__)


class GroqClient:
    """
    Resilient Groq API client that acts as the instructor/helper LLM for the
    Qwen 1.5B SMART pipeline.

    Key features:
      - Round-Robin cycling across exactly 2 Groq API keys (from separate accounts)
        to distribute load and stay under the 30 RPM per-key limit.
      - On every 429 / RateLimitError the client advances to the *next* key before
        sleeping, so retries always hit a fresh account.
      - Exponential backoff with ±25 % random jitter to avoid thundering-herd
        behaviour when both keys are temporarily exhausted.
    """

    def __init__(self, keys=None, model="llama-3.1-8b-instant"):
        """
        Parameters
        ----------
        keys : list[str] | None
            Exactly 2 Groq API keys.  If omitted, reads GROQ_API_KEY_1 and
            GROQ_API_KEY_2 from the environment.
        model : str
            Groq model identifier to use for all completions.
        """
        if keys is None:
            key1 = os.getenv("GROQ_API_KEY_1")
            key2 = os.getenv("GROQ_API_KEY_2")
            if not key1 or not key2:
                raise ValueError(
                    "Must provide exactly 2 Groq API keys either via the `keys` "
                    "parameter or the GROQ_API_KEY_1 / GROQ_API_KEY_2 env vars."
                )
            keys = [key1, key2]

        if len(keys) != 2:
            raise ValueError("Exactly 2 Groq API keys are required for Round-Robin key cycling.")

        self.clients = [Groq(api_key=k) for k in keys]
        # cycle() gives an infinite iterator; each call to next() advances the pointer.
        self.client_cycle = cycle(self.clients)
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_client(self):
        """Advance the round-robin pointer and return the next client."""
        return next(self.client_cycle)

    @staticmethod
    def _backoff_sleep(backoff: float) -> float:
        """
        Sleep for `backoff` seconds (with ±25 % jitter) and return the next
        backoff value (doubled, capped at 64 s).
        """
        jitter = backoff * random.uniform(-0.25, 0.25)
        sleep_time = max(1.0, backoff + jitter)
        logger.warning(f"Rate limit hit. Sleeping {sleep_time:.1f}s before retrying …")
        time.sleep(sleep_time)
        return min(backoff * 2, 64.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, messages, max_tokens=1024, temperature=0.7, top_p=1.0, stop=None):
        """
        Generate a single completion.

        On a 429 error the method:
          1. Advances to the next API key (round-robin).
          2. Sleeps with exponential back-off + jitter.
          3. Retries indefinitely until success.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style chat messages list.
        max_tokens : int
        temperature : float
        top_p : float
        stop : list[str] | None
            Optional stop sequences.

        Returns
        -------
        str
            The assistant's reply text.
        """
        # Pick the starting client for this request.
        client = self._next_client()
        backoff = 1.0

        while True:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                logger.warning(f"RateLimitError on key {id(client)}: {e}")
                # Advance to the next key BEFORE sleeping so the retry uses a
                # fresh account — this is the fix for the original cycling bug.
                client = self._next_client()
                backoff = self._backoff_sleep(backoff)

            except Exception as e:
                if "429" in str(e) or "Too Many Requests" in str(e).lower():
                    logger.warning(f"429 via generic Exception on key {id(client)}: {e}")
                    client = self._next_client()
                    backoff = self._backoff_sleep(backoff)
                else:
                    raise

    def generate_batch(self, messages_batch, max_tokens=1024, temperature=0.7, top_p=1.0, stop=None):
        """
        Generate completions for a list of message threads sequentially.

        Each call internally round-robins between the two API keys, so a batch
        of N requests naturally alternates: key-1, key-2, key-1, key-2, …

        Parameters
        ----------
        messages_batch : list[list[dict]]
            A list of OpenAI-style chat message threads.

        Returns
        -------
        list[str]
            One reply string per input thread, in the same order.
        """
        results = []
        for messages in messages_batch:
            result = self.generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            results.append(result)
        return results
