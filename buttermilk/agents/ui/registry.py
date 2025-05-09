"""UI Registry for dynamically managing UI implementations.

This module provides a registry system for UI implementations, allowing the UIProxyAgent
to dynamically connect to the appropriate UI implementation at runtime.
"""

from collections import defaultdict
from typing import Any, Dict, Type, Optional, Callable, ClassVar, TypeVar, cast, Union

from buttermilk import logger
from buttermilk._core.agent import Agent

# Define a type variable for any Agent subclass
AgentType = TypeVar('AgentType', bound=Agent)


class UIRegistry:
    """Registry for UI implementations.
    
    This class manages available UI implementations and provides methods for registering
    and retrieving them. It follows a singleton pattern to ensure a single registry exists.
    """
    
    _instance: ClassVar[Optional['UIRegistry']] = None
    _registry: Dict[str, Type[Any]]
    _default_ui: Optional[str] = None
    
    def __new__(cls) -> 'UIRegistry':
        """Create a singleton instance of the registry."""
        if cls._instance is None:
            cls._instance = super(UIRegistry, cls).__new__(cls)
            cls._instance._registry = {}
            cls._instance._default_ui = None
        return cls._instance
    
    def register(self, ui_name: str, ui_class: Type[Agent], default: bool = False) -> None:
        """Register a UI implementation.
        
        Args:
            ui_name: The name of the UI implementation
            ui_class: The class for the UI implementation
            default: Whether this should be the default UI implementation
        """
        self._registry[ui_name] = ui_class
        logger.debug(f"Registered UI implementation: {ui_name}")
        
        if default or self._default_ui is None:
            self._default_ui = ui_name
            logger.debug(f"Set default UI implementation to: {ui_name}")
    
    def get_implementation(self, ui_name: Optional[str] = None) -> Type[Agent]:
        """Get a UI implementation class by name.
        
        Args:
            ui_name: The name of the UI implementation to retrieve
            
        Returns:
            The UI implementation class
            
        Raises:
            ValueError: If the requested UI implementation is not registered
        """
        if ui_name is None:
            if self._default_ui is None:
                raise ValueError("No default UI implementation registered")
            ui_name = self._default_ui
            
        if ui_name not in self._registry:
            available = ", ".join(self._registry.keys())
            raise ValueError(f"UI implementation '{ui_name}' not found. Available: {available}")
            
        return self._registry[ui_name]
    
    def list_implementations(self) -> list[str]:
        """List all registered UI implementations.
        
        Returns:
            A list of registered UI implementation names
        """
        return list(self._registry.keys())


# Convenience functions for accessing the registry

def register_ui(ui_name: str, ui_class: Type[Any], default: bool = False) -> None:
    """Register a UI implementation in the registry.
    
    Args:
        ui_name: The name of the UI implementation
        ui_class: The class for the UI implementation
        default: Whether this should be the default UI implementation
    """
    registry = UIRegistry()
    registry.register(ui_name, ui_class, default)


def get_ui_implementation(ui_name: Optional[str] = None) -> Type[Any]:
    """Get a UI implementation class by name.
    
    Args:
        ui_name: The name of the UI implementation to retrieve
        
    Returns:
        The UI implementation class
    """
    registry = UIRegistry()
    return registry.get_implementation(ui_name)


def list_ui_implementations() -> list[str]:
    """List all registered UI implementations.
    
    Returns:
        A list of registered UI implementation names
    """
    registry = UIRegistry()
    return registry.list_implementations()
