class BaseLLM:
    async def agenerate(self, prompt: str, **kwargs: object) -> str:
        raise NotImplementedError
