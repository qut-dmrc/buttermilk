import importlib
import pkgutil

import shortuuid
from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import ConfigDict, Field, field_validator, model_validator

from buttermilk._core.log import logger
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
            path=package.__path__, prefix=prefix, onerror=lambda name: print(f"Error importing {name}")
        ):
            try:
                # Import the module to trigger registration via decorators or class loading
                importlib.import_module(modname)
            except Exception as e:
                # Log error, but continue discovery
                logger.warning(f"Error importing module {modname}: {e}")


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
      parallel_variants:
        model: ["gpt-4", "claude-3"]    # Creates 2 parallel agent instances
      sequential_variants:
        criteria: ["accuracy", "speed"] # Each agent instance runs 2 tasks sequentially
        temperature: [0.5, 0.8]         # Total 4 sequential tasks per agent 
                                        # (accuracy/0.5, accuracy/0.8, speed/0.5, speed/0.8)
      inputs:
        history: history
    ```
    """

    num_runs: int = Field(
        default=1,
        description="Number of times to replicate each parallel variant agent instance.",
        exclude=True,
    )
    parallel_variants: dict = Field(
        default={},
        description="Parameters to create parallel agent instances via cross-multiplication.",
        exclude=True,
    )
    sequential_variants: dict = Field(
        default={},
        description="Parameters defining sequential tasks for each agent instance via cross-multiplication.",
        exclude=True,
    )

    _validate_variants = field_validator(
        "parallel_variants", "sequential_variants", mode="before"
    )(convert_omegaconf_objects())

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
        parallel_variant_combinations = expand_dict(self.parallel_variants)
        if not parallel_variant_combinations:
            parallel_variant_combinations = [{}] # Ensure at least one base agent config

        # Expand sequential variants
        sequential_task_sets = expand_dict(self.sequential_variants)
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
                cfg_dict["tasks"] = sequential_task_sets

                # Generate unique ID incorporating parallel variants and run number
                id_parts = [self.id]
                # Add parallel variant info to ID if there are multiple combinations
                if len(parallel_variant_combinations) > 1:
                    param_str = "_".join(f"{k}-{v}" for k, v in sorted(parallel_params.items()))
                    # Basic sanitization and shortening for ID
                    param_str = ''.join(c if c.isalnum() or c in ['-','_'] else '' for c in param_str)[:20]
                    if param_str: # Avoid adding empty strings
                         id_parts.append(param_str)
                if self.num_runs > 1:
                    id_parts.append(f"run{i}")
                # Add hash only if needed for uniqueness (multiple runs or variants)
                if self.num_runs > 1 or len(parallel_variant_combinations) > 1:
                    id_parts.append(shortuuid.uuid()[:4])

                cfg_dict["id"] = "-".join(id_parts)[:63] # Ensure reasonable length

                # Create and add the AgentConfig instance
                try:
                    agent_config_instance = AgentConfig(**cfg_dict)
                    generated_configs.append((agent_class, agent_config_instance))
                except Exception as e:
                    print(f"Error creating AgentConfig for {cfg_dict.get('id', 'unknown')}: {e}")
                    # Decide whether to raise, log, or skip this config
                    raise # Re-raise by default

        # Handle the edge case: No variants, num_runs=1 (should result in one config)
        if not self.parallel_variants and not self.sequential_variants and self.num_runs == 1:
             if len(generated_configs) == 1:
                 # Ensure the single generated config has the original ID if possible
                 generated_configs[0][1].id = self.id
                 return generated_configs
             else: # Should not happen with the logic above, but as a fallback:
                 cfg_dict = static_config.copy()
                 cfg_dict["parameters"] = base_parameters
                 cfg_dict["tasks"] = [{}]
                 cfg_dict["id"] = self.id
                 return [(agent_class, AgentConfig(**cfg_dict))]


        return generated_configs


# Discover all agent classes
AgentRegistry.discover("buttermilk.agents")
# Optionally discover from other packages if needed
# AgentRegistry.discover("other_agent_package")
