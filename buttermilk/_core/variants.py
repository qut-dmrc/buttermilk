import importlib
import pkgutil

import shortuuid
from pydantic import Field, field_validator

from buttermilk._core.agent import Agent
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import convert_omegaconf_objects


class AgentRegistry:
    _agents: dict[str, type[Agent]] = {}

    @classmethod
    def register(cls, agent_class: type[Agent]) -> type[Agent]:
        """Register an agent class with the registry."""
        cls._agents[agent_class.__name__] = agent_class
        return agent_class  # Allow use as a decorator

    @classmethod
    def get(cls, name: str) -> type[Agent]:
        """Get an agent class by name."""
        return cls._agents.get(name)

    @classmethod
    def get_all(cls) -> dict[str, type[Agent]]:
        """Get all registered agents."""
        return cls._agents.copy()

    @classmethod
    def discover(cls, package_name: str = "buttermilk", module: str = "agents") -> None:
        """Discover and register all Agent subclasses in the package."""
        # Import all submodules to make sure they're loaded
        package = importlib.import_module(package_name)
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + ".", onerror=lambda x: print(f"Error importing {x}")):
            try:
                importlib.import_module(name)
            except Exception as e:
                print(f"Error importing {name}: {e}")

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


class AgentVariants(Agent):
    agent_id: str = "unconfigured"

    agent_obj: str = Field(..., description="The object name to instantiate")
    num_runs: int = Field(
        default=1,
        description="Number of times to run the agent for each variant",
    )
    variants: dict = {}
    """A factory for creating Agent instance variants for a single
    step of a workflow.

    Creates a new agent for every combination of parameters in a given
    step of the workflow to run. Agents have a variants mapping;
    each permutation of these is multiplied by num_runs. Agents also
    have an inputs mapping that does not get multiplied.
    """

    validate_parameters = field_validator(
        "parameters", "variants", "inputs", "outputs", mode="before"
    )(convert_omegaconf_objects())

    def get_configs(self) -> list[tuple[type, dict]]:
        # Create variants (permutations of vars multiplied by num_runs)
        variant_configs = self.num_runs * expand_dict(self.variants)

        agents = []
        for variant in variant_configs:
            # Start with static config
            cfg = dict(
                **self.model_dump(
                    exclude={"agent_id", "num_runs", "variants", "agent", "agent_obj"}
                )
            )

            # Generate unique ID for this agent
            cfg["agent_id"] = f"{self.name}-{shortuuid.uuid()[:6]}"

            # Add variant config to parameters
            cfg["parameters"].update(variant)

            # Instantiate
            agent_class = AgentRegistry.get(self.agent_obj)
            if agent_class is None:
                raise ValueError(f"Agent class '{self.agent_obj}' not found in registry")

            agents.append((agent_class, cfg))

        return agents


# Discover all agent classes
AgentRegistry.discover("buttermilk.agents")
