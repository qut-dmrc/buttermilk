from abc import ABC, abstractmethod
from typing import Any
from distutils.util import strtobool
from pydantic import BaseModel


class IOInterface(BaseModel, ABC):

    @abstractmethod
    async def send_output(self, message: Any, source: str = "") -> None:
        """Send output to the user interface"""

    @abstractmethod
    async def get_input(self, message: Any, source: str = "") -> str:
        """Request input from the user interface"""

    async def confirm(self, message: str = "") -> bool:
        """Request confirmation from the user interface"""
        user_input = await self.get_input(
            message or "Proceed? (y/n): ",
        )
        return bool(strtobool(user_input))
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface"""

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
