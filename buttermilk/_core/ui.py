from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class IOInterface(BaseModel, ABC):

    @abstractmethod
    async def send_output(self, message: Any, source: str = "") -> None:
        """Send output to the user interface"""

    @abstractmethod
    async def get_input(self, message: Any, source: str = "") -> None:
        """Request input from the user interface"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface"""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
