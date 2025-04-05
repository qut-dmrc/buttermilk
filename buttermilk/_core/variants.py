import importlib
import pkgutil

import shortuuid
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict, Field, field_validator, model_validator

from buttermilk._core.agent import Agent, AgentConfig
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
        for _, name, is_pkg in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + ".",
            onerror=lambda x: print(f"Error importing {x}"),
        ):
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


class AgentVariants(AgentConfig):
    """A factory for creating Agent instance variants.

    Creates a new agent for every combination of parameters in a given
    step of the workflow to run. Agents have a variants mapping;
    each permutation of these is multiplied by num_runs. Agents also
    have an inputs mapping that does not get multiplied.

    Example:
    ```yaml
    - id: ANALYST
      role: "Analyst"
      agent_obj: LLMAgent
      num_runs: 2  # Creates 2 instances
      variants:
        model: ["gpt-4", "claude-3"]  # Creates variants with different models
      inputs:
        history: history
    ```
    """

    num_runs: int = Field(
        default=1,
        description="Number of times to run the agent for each variant",
        exclude=True,
    )
    variants: dict = Field(
        default={},
        description="Variables that will be cross-multiplied to generate multiple agents",
        exclude=True,
    )

    validate_parameters = field_validator(
        "variants",
        mode="before",
    )(convert_omegaconf_objects())

    @model_validator(mode="after")
    def validate_extra_fields(self):
        """Convert any omegaconf objects in extra fields to plain Python objects."""
        for key, value in dict(self.model_extra).items():
            if isinstance(value, (DictConfig, ListConfig)):
                self.model_extra[key] = OmegaConf.to_container(value, resolve=True)
        return self

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=False)

    def get_configs(self) -> list[tuple[type, AgentConfig]]:
        # Get static config
        static = dict(**self.model_dump())

        # Get object
        agent_class = AgentRegistry.get(self.agent_obj)
        if agent_class is None:
            raise ValueError(f"Agent class '{self.agent_obj}' not found in registry")

        # Create variants (permutations of vars multiplied by num_runs)
        variant_configs = self.num_runs * expand_dict(self.variants)

        if not variant_configs:
            return [(agent_class, AgentConfig(**static))]

        agents = []
        for variant in variant_configs:
            # Start with static config
            cfg = dict(**static)

            # Generate unique ID for this agent
            cfg["id"] = f"{self.id}-{shortuuid.uuid()[:6]}"

            # Add variant config to parameters
            cfg["parameters"].update(variant)

            agents.append((agent_class, AgentConfig(**cfg)))

        return agents


# Discover all agent classes
AgentRegistry.discover("buttermilk.agents")
