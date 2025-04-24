import importlib
import pkgutil

from buttermilk._core.agent import Agent
from buttermilk._core.log import logger


class AgentRegistry:
    """
    Manages the registration and retrieval of Agent subclasses.

    This class acts as a central repository for all available Agent types
    within the application. Agents can be registered explicitly using the
    `register` method (often used as a decorator) or discovered automatically
    from specified packages using the `discover` method. This allows for
    dynamic loading and instantiation of agents based on configuration.
    """

    _agents: dict[str, type[Agent]] = {}

    @classmethod
    def register(cls, agent_class: type[Agent]) -> type[Agent]:
        """Register an agent class with the registry."""
        cls._agents[agent_class.__name__] = agent_class
        return agent_class  # Allow use as a decorator

    @classmethod
    def get(cls, name: str) -> type[Agent]:
        """
        Get an agent class by name.

        If the agent is not found initially, it triggers the discovery
        process before raising an error.
        """
        agent_class = cls._agents.get(name)
        if agent_class is None:
            # Attempt discovery if not found, might help in dynamic scenarios
            cls.discover()
            agent_class = cls._agents.get(name)
            if agent_class is None:
                raise ValueError(f"Agent class '{name}' not found in registry after discovery.")
        return agent_class

    @classmethod
    def get_all(cls) -> dict[str, type[Agent]]:
        """Get all registered agents."""
        return cls._agents.copy()

    @classmethod
    def discover(cls, package_name: str = "buttermilk") -> None:
        """
        Discover and register all Agent subclasses in the specified package.

        Recursively walks through the package and its subpackages, importing
        modules to trigger agent registration (e.g., via decorators).
        """
        try:
            package = importlib.import_module(package_name)
        except ModuleNotFoundError:
            print(f"Warning: Package '{package_name}' not found for agent discovery.")
            return

        # Use pkgutil.walk_packages for better handling of subpackages
        prefix = package.__name__ + "."
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package.__path__, prefix=prefix, onerror=lambda name: logger.warning(f"AgentRegistry hit error importing {name}")
        ):
            try:
                # Import the module to trigger registration via decorators or class loading
                importlib.import_module(modname)
            except Exception as e:
                # Log error, but continue discovery
                logger.warning(f"Error importing module {modname}: {e}")

        # Now find all Agent subclasses that have been loaded
        def get_all_subclasses(cls):
            all_subclasses = []
            for subclass in cls.__subclasses__():
                all_subclasses.append(subclass)
                all_subclasses.extend(get_all_subclasses(subclass))
            return all_subclasses

        # Register all found subclasses
        for subclass in get_all_subclasses(Agent):
            cls.register(subclass)


# Discover all agent classes
AgentRegistry.discover("buttermilk.agents")
# Optionally discover from other packages if needed
# AgentRegistry.discover("other_agent_package")
