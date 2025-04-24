from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

# Assuming AgentConfig and Agent are importable for type hinting/instance checks
# Adjust imports based on your actual project structure
from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.variants import AgentRegistry, AgentVariants


# Dummy Agent class for testing
class MockAgent(Agent):
    async def _process(self, *args, **kwargs):  # pragma: no cover
        pass  # pragma: no cover


@pytest.fixture(autouse=True)
def mock_agent_registry():
    """Automatically mock AgentRegistry for all tests in this module."""
    # Register the mock agent
    AgentRegistry.register(MockAgent)
    # Patch 'get' to always return MockAgent if requested by name 'MockAgent'
    original_get = AgentRegistry.get

    def mock_get(name):
        if name == "MockAgent":
            return MockAgent
        return original_get(name)  # Call original for other agents if needed

    with patch.object(AgentRegistry, "get", side_effect=mock_get):
        yield
    # Cleanup: Remove mock agent after tests (optional, depends on test isolation needs)
    if "MockAgent" in AgentRegistry._agents:
        del AgentRegistry._agents["MockAgent"]


@pytest.fixture
def base_variant_config():
    """Provides a base configuration for AgentVariants."""
    return {
        "id": "TEST_AGENT",
        "role": "Tester",
        "name": "Test Agent Variants",  # Add required name field
        "description": "A test agent",
        "agent_obj": "MockAgent",  # Use the name of the registered mock agent
        "parameters": {"base_param": "base_value"},
        "inputs": {},
        "outputs": {},
    }


# --- Test Cases ---


def test_no_variants_single_run(base_variant_config):
    """Test config generation with no variants and num_runs=1."""
    variant_factory = AgentVariants(**base_variant_config)
    configs = variant_factory.get_configs()

    assert len(configs) == 1
    agent_class, config = configs[0]

    assert agent_class is MockAgent
    assert isinstance(config, AgentConfig)
    assert config.id == "TEST_AGENT"  # Should retain original ID
    assert config.parameters == {"base_param": "base_value"}
    assert config.sequential_tasks == [{}]  # Default single task


def test_only_parallel_variants(base_variant_config):
    """Test config generation with only parallel variants."""
    config_data = {
        **base_variant_config,
        "parallel_variants": {"model": ["model_a", "model_b"]},
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 2
    ids = set()
    for agent_class, config in configs:
        assert agent_class is MockAgent
        assert isinstance(config, AgentConfig)
        assert config.sequential_tasks == [{}]  # Default single task
        assert "model" in config.parameters
        ids.add(config.id)
        if config.parameters["model"] == "model_a":
            assert config.parameters["base_param"] == "base_value"
            assert "model_a" in config.id
        else:
            assert config.parameters["model"] == "model_b"
            assert config.parameters["base_param"] == "base_value"
            assert "model_b" in config.id
    assert len(ids) == 2  # Ensure unique IDs generated


def test_only_sequential_variants(base_variant_config):
    """Test config generation with only sequential variants."""
    config_data = {
        **base_variant_config,
        "sequential_variants": {"criteria": ["c1", "c2"], "temp": [0.5, 0.8]},
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 1
    agent_class, config = configs[0]

    assert agent_class is MockAgent
    assert isinstance(config, AgentConfig)
    assert config.id == "TEST_AGENT"  # Original ID, no parallel variants
    assert config.parameters == {"base_param": "base_value"}  # Base parameters only
    assert len(config.sequential_tasks) == 4  # 2 criteria * 2 temp
    expected_tasks = [
        {"criteria": "c1", "temp": 0.5},
        {"criteria": "c1", "temp": 0.8},
        {"criteria": "c2", "temp": 0.5},
        {"criteria": "c2", "temp": 0.8},
    ]
    # Convert to set of tuples for order-independent comparison
    assert set(tuple(sorted(d.items())) for d in config.sequential_tasks) == set(tuple(sorted(d.items())) for d in expected_tasks)


def test_both_parallel_and_sequential_variants(base_variant_config):
    """Test config generation with both parallel and sequential variants."""
    config_data = {
        **base_variant_config,
        "parallel_variants": {"model": ["m1", "m2"]},
        "sequential_variants": {"temp": [0.1, 0.9]},
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 2  # 2 parallel variants
    expected_tasks = [{"temp": 0.1}, {"temp": 0.9}]
    task_set_tuples = set(tuple(sorted(d.items())) for d in expected_tasks)

    for agent_class, config in configs:
        assert agent_class is MockAgent
        assert isinstance(config, AgentConfig)
        assert "model" in config.parameters
        assert config.parameters["base_param"] == "base_value"
        assert len(config.sequential_tasks) == 2
        assert set(tuple(sorted(d.items())) for d in config.sequential_tasks) == task_set_tuples
        if config.parameters["model"] == "m1":
            assert "m1" in config.id
        else:
            assert config.parameters["model"] == "m2"
            assert "m2" in config.id


def test_num_runs_greater_than_one(base_variant_config):
    """Test config generation with num_runs > 1."""
    config_data = {
        **base_variant_config,
        "num_runs": 3,
        "parallel_variants": {"model": ["m_a"]},  # Single parallel variant
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 3  # 1 parallel variant * 3 runs
    ids = set()
    for i, (agent_class, config) in enumerate(configs):
        assert agent_class is MockAgent
        assert isinstance(config, AgentConfig)
        assert config.parameters == {"base_param": "base_value", "model": "m_a"}
        assert config.sequential_tasks == [{}]
        assert f"run{i}" in config.id
        assert "m_a" in config.id  # Parallel param should still be in ID
        ids.add(config.id)
    assert len(ids) == 3  # Ensure unique IDs across runs


@patch("buttermilk._core.variants.shortuuid.uuid")
def test_id_generation_uniqueness(mock_uuid, base_variant_config):
    """Test ID generation includes hash for uniqueness when needed."""
    mock_uuid.side_effect = ["abcd", "efgh", "ijkl", "mnop"]  # Provide unique suffixes

    # Case 1: Multiple parallel variants (needs hash)
    config_data_p = {
        **base_variant_config,
        "parallel_variants": {"model": ["m1", "m2"]},
    }
    variant_factory_p = AgentVariants(**config_data_p)
    configs_p = variant_factory_p.get_configs()
    assert len(configs_p) == 2
    assert "m1" in configs_p[0][1].id and "abcd" in configs_p[0][1].id
    assert "m2" in configs_p[1][1].id and "efgh" in configs_p[1][1].id

    # Case 2: Multiple runs (needs hash)
    config_data_r = {
        **base_variant_config,
        "num_runs": 2,
    }
    variant_factory_r = AgentVariants(**config_data_r)
    configs_r = variant_factory_r.get_configs()
    assert len(configs_r) == 2
    assert "run0" in configs_r[0][1].id and "ijkl" in configs_r[0][1].id
    assert "run1" in configs_r[1][1].id and "mnop" in configs_r[1][1].id

    # Case 3: Single config (no hash needed)
    variant_factory_s = AgentVariants(**base_variant_config)
    configs_s = variant_factory_s.get_configs()
    assert len(configs_s) == 1
    assert configs_s[0][1].id == "TEST_AGENT"  # Original ID, no hash


def test_parameter_overwriting(base_variant_config):
    """Test that parallel variants overwrite base parameters."""
    config_data = {
        **base_variant_config,
        "parameters": {"base_param": "original", "model": "base_model"},
        "parallel_variants": {"model": ["override_model"]},
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 1
    agent_class, config = configs[0]
    assert config.parameters["base_param"] == "original"  # Unchanged base param
    assert config.parameters["model"] == "override_model"  # Overwritten by parallel variant


def test_omegaconf_conversion(base_variant_config):
    """Test that OmegaConf dicts/lists in variants are converted."""
    config_data = {
        **base_variant_config,
        "parallel_variants": OmegaConf.create({"model": ["m1", "m2"]}),
        "sequential_variants": OmegaConf.create({"temp": [0.1, 0.9]}),
    }
    # The validator runs on initialization
    variant_factory = AgentVariants(**config_data)

    # Check internal state after validation (optional, but good for debugging)
    assert isinstance(variant_factory.variants, dict)
    assert not isinstance(variant_factory.variants, OmegaConf)
    assert isinstance(variant_factory.tasks, dict)
    assert not isinstance(variant_factory.tasks, OmegaConf)

    # Check generated configs
    configs = variant_factory.get_configs()
    assert len(configs) == 2
    assert len(configs[0][1].sequential_tasks) == 2
    assert isinstance(configs[0][1].sequential_tasks[0], dict)


def test_empty_variants(base_variant_config):
    """Test config generation with empty variant dicts."""
    config_data = {
        **base_variant_config,
        "parallel_variants": {},
        "sequential_variants": {},
    }
    variant_factory = AgentVariants(**config_data)
    configs = variant_factory.get_configs()

    assert len(configs) == 1
    agent_class, config = configs[0]
    assert config.id == "TEST_AGENT"
    assert config.parameters == {"base_param": "base_value"}
    assert config.sequential_tasks == [{}]  # Default task


def test_agent_not_found(base_variant_config):
    """Test that ValueError is raised if agent_obj is not found."""
    config_data = {**base_variant_config, "agent_obj": "NonExistentAgent"}
    variant_factory = AgentVariants(**config_data)
    with pytest.raises(ValueError, match="Agent class 'NonExistentAgent' not found"):
        variant_factory.get_configs()
