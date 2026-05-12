from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")


async def run_with_timeout(coro: Awaitable[T], timeout_seconds: int) -> T:
    """Run an awaitable with an asyncio timeout."""
    return await asyncio.wait_for(coro, timeout=timeout_seconds)
