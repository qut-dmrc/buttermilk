import importlib
import inspect
import pkgutil
import sys
from typing import Type, List, Dict

from pydantic import Field, field_validator

from buttermilk._core.agent import Agent
from buttermilk.bm import Singleton
from buttermilk.utils.utils import expand_dict
from buttermilk.utils.validators import convert_omegaconf_objects

class AgentRegistry:
    _agents: Dict[str, Type[Agent]] = {}
    
    @classmethod
    def register(cls, agent_class: Type[Agent]) -> Type[Agent]:
        """Register an agent class with the registry."""
        cls._agents[agent_class.__name__] = agent_class
        return agent_class  # Allow use as a decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Agent]:
        """Get an agent class by name."""
        return cls._agents.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type[Agent]]:
        """Get all registered agents."""
        return cls._agents.copy()
    
    @classmethod
    def discover(cls, package_name: str = "buttermilk", module: str = "agents") -> None:
        """Discover and register all Agent subclasses in the package."""
        # Import all submodules to make sure they're loaded
        package = importlib.import_module(package_name)
        for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.', onerror=lambda x: print(f"Error importing {x}")):
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
    agent: str = Field(..., description="The object name to instantiate")
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

    validate_parameters = field_validator("parameters", "variants", "inputs", "outputs", mode="before")(convert_omegaconf_objects())

    def create(self) -> list[Agent]:
        # Create variants (permutations of vars multiplied by num_runs)
        variant_configs = self.num_runs * expand_dict(self.variants)

        agents = []
        for cfg in variant_configs:
            # Start with static config
            variant = dict(**self.model_dump(exclude={"num_runs","variants","agent"}))
            
            # Add variant to parameters
            variant["parameters"].update(cfg)

            # Instantiate
            agent_class = AgentRegistry.get(self.agent)
            if agent_class is None:
                raise ValueError(f"Agent class '{self.agent}' not found in registry")
            agent = agent_class(**variant)
            agents.append(agent)

        return agents


# Discover all agent classes
AgentRegistry.discover("buttermilk", "agents")
