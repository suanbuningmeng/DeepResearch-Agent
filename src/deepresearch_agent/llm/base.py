class BaseLLM:
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        """Generate text asynchronously from a prompt using a concrete LLM backend."""
        raise NotImplementedError
