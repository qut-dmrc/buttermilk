import importlib
import pkgutil

import shortuuid
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict, Field, field_validator, model_validator

from buttermilk._core.log import logger
from buttermilk._core.agent import Agent, AgentConfig, FatalError
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
        """Discover and register all Agent subclasses in the package."""
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
class AgentVariants(AgentConfig):
    """
    A factory for creating Agent instance variants based on parameter combinations.

    Defines two types of variants:
    1. `parallel_variants`: Parameters whose combinations create distinct agent instances
       (e.g., different models). These agents can potentially run in parallel.
    2. `sequential_variants`: Parameters whose combinations define sequential tasks
       executed by *each* agent instance created from `parallel_variants`.

    Example:
    ```yaml
    - id: ANALYST
      role: "Analyst"
      agent_obj: LLMAgent
      num_runs: 1
      variants:
        model: ["gpt-4", "claude-3"]    # Creates 2 parallel agent instances
      tasks:
        criteria: ["accuracy", "speed"] # Each agent instance runs 2 tasks sequentially
        temperature: [0.5, 0.8]         # Total 4 sequential tasks per agent 
                                        # (accuracy/0.5, accuracy/0.8, speed/0.5, speed/0.8)
      parameters:
        template: analyst               # parameter sets shared for each task
      inputs:
        results: othertask.outputs.results  # dynamic inputs mapped from other data
    ```
    """

    @model_validator(mode="after")
    def validate_extra_fields(self):
        """Convert any omegaconf objects in extra fields to plain Python objects."""
        for key, value in dict(self.model_extra).items():
            if isinstance(value, (DictConfig, ListConfig)):
                self.model_extra[key] = OmegaConf.to_container(value, resolve=True)
        return self 

    def get_configs(self) -> list[tuple[type, AgentConfig]]:
        """
        Generates agent configurations based on parallel and sequential variants.
        """
        # Get static config (base attributes excluding variant fields)
        static_config = self.model_dump(exclude={'parallel_variants', 'sequential_variants', 'num_runs', 'parameters', 'tasks'})
        base_parameters = self.parameters.copy() # Base parameters common to all

        # Get agent class
        agent_class = AgentRegistry.get(self.agent_obj)

        # Expand parallel variants
        parallel_variant_combinations = expand_dict(self.variants)
        if not parallel_variant_combinations:
            parallel_variant_combinations = [{}] # Ensure at least one base agent config

        # Expand sequential variants
        sequential_task_sets = expand_dict(self.tasks)
        if not sequential_task_sets:
            sequential_task_sets = [{}] # Default: one task with no specific sequential params

        generated_configs = []
        # Create agent configs based on parallel variants and num_runs
        for i in range(self.num_runs):
            for parallel_params in parallel_variant_combinations:
                # Start with static config and base parameters
                cfg_dict = static_config.copy()
                # Combine base parameters with the current parallel variant parameters
                # Parallel variant parameters overwrite base parameters if keys conflict
                cfg_dict["parameters"] = {**base_parameters, **parallel_params}

                # Assign the sequential task sets
                cfg_dict["sequential_tasks"] = sequential_task_sets

                # Generate unique ID
                cfg_dict["id"] = f"{cfg_dict['role'].upper()}-{shortuuid.uuid()[:4]}"

                # Create and add the AgentConfig instance
                try:
                    agent_config_instance = AgentConfig(**cfg_dict)
                    generated_configs.append((agent_class, agent_config_instance))
                except Exception as e:
                    print(f"Error creating AgentConfig for {cfg_dict.get('id', 'unknown')}: {e}")
                    # Decide whether to raise, log, or skip this config
                    raise # Re-raise by default

        if len(generated_configs) == 0:
            raise FatalError(f"Could not create agent variant configs for {self.role} {self.name}")

        return generated_configs


# Discover all agent classes
AgentRegistry.discover("buttermilk.agents")
# Optionally discover from other packages if needed
# AgentRegistry.discover("other_agent_package")
