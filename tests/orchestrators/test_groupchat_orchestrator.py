from unittest.mock import MagicMock, patch

import pytest
from autogen_core import AgentType, DefaultTopicId, SingleThreadedAgentRuntime

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig, AgentVariants
from buttermilk._core.types import RunRequest
from buttermilk.orchestrators.groupchat import AutogenOrchestrator


# Minimal Agent for testing
class MockTestAgent(Agent):
    async def _process(self, *, message, **kwargs):
        # This method is not called directly in these unit tests.
        # It's here to satisfy the Agent abstract class requirements.
        pass


@pytest.mark.anyio
async def test_agent_registry_population_and_get_config():
    # Initialize the orchestrator
    # The 'config' argument for Orchestrator base class is not strictly needed if
    # we are directly setting attributes like 'agents', 'parameters'.
    # Passing a minimal dict to satisfy Pydantic model initialization if it expects one.
    orchestrator = AutogenOrchestrator(name="TestOrch", orchestrator="AutogenOrchestrator", parameters={})

    # Define mock AgentConfig data for MockTestAgent
    # Note: AgentConfig expects 'name' and 'role'. 'unique_identifier' is used to form 'agent_id'.
    # 'agent_obj' in AgentVariants becomes 'cls' for AgentConfig if type is str.
    # Let's ensure the structure matches AgentConfig and AgentVariants expectations.
    agent_config_data = {
        "name": "TestAgent001",  # Agent name
        "role": "TESTER",
        "unique_identifier": "001",  # Used to create agent_id
        "parameters": {"model": "test_model", "some_other_param": "value"},
        "agent_obj": f"{__name__}.MockTestAgent",  # Fully qualified path to MockTestAgent
    }

    # AgentConfig generates agent_id like ROLE-UNIQUE_IDENTIFIER
    expected_agent_id = f"{agent_config_data['role']}-{agent_config_data['unique_identifier']}"

    # Create AgentVariants using the config data
    # The AgentVariants class will internally create an AgentConfig instance.
    mock_agent_variant = AgentVariants(**agent_config_data)

    # Assign the mock agent variant to the orchestrator's 'agents' attribute
    # The key "TESTER_ROLE_KEY" is arbitrary for this test setup.
    orchestrator.agents = {"TESTER_ROLE_KEY": mock_agent_variant}

    # The RunRequest for _register_agents
    test_run_request = RunRequest(flow="test_flow", session_id="test_session_123")

    # Mock the AutogenAgentAdapter.register to simulate it calling the registration_callback.
    # This is crucial for testing the registry logic without a full Autogen runtime.
    async def mock_adapter_register(runtime, type, factory):
        # The factory passed to AutogenAgentAdapter.register is a lambda:
        #   lambda orch=self, v_cfg=config_with_session, a_cls=agent_cls, t_type=self._topic.type:
        #       agent_factory(orch, cfg=v_cfg, cls=a_cls, topic_type=t_type)
        #
        # The agent_factory itself is:
        #   def agent_factory(orchestrator_ref, cfg, cls, topic_type):
        #       return AutogenAgentAdapter(
        #           agent_cfg=cfg,
        #           agent_cls=cls,
        #           topic_type=topic_type,
        #           registration_callback=orchestrator_ref._register_buttermilk_agent_instance
        #       )
        #
        # When AutogenAgentAdapter is initialized (by the factory call), it will instantiate
        # the actual Buttermilk agent (MockTestAgent in this case) and then call the
        # registration_callback.

        # We need to retrieve the actual AgentConfig and agent_cls as prepared by
        # AgentVariants.get_configs() because this is what _register_agents uses.
        # get_configs returns a list of (agent_cls, agent_config_instance)
        agent_cls_from_variant, agent_config_from_variant = mock_agent_variant.get_configs(params=test_run_request)[0]

        # Simulate the instantiation of the agent and the callback
        # The agent_id is generated within AgentConfig and then within the Agent itself.
        # The callback expects agent_id and the agent_instance.

        # The config_with_session is created inside _register_agents,
        # so agent_config_from_variant already includes session_id if parameters are merged.
        # Let's refine how config_with_session is handled.
        # The parameters from orchestrator and session_id are merged into variant_config
        # *before* calling get_configs if we follow the _register_agents logic closely.
        # However, for this mock, we can assume agent_config_from_variant is what the agent gets.

        # The Agent class itself generates its agent_id if not provided in config.
        # The AgentConfig from AgentVariants should have the agent_id.

        agent_instance = agent_cls_from_variant(**agent_config_from_variant.model_dump())

        # Directly call the orchestrator's registration method,
        # simulating what AutogenAgentAdapter's __init__ would do.
        orchestrator._register_buttermilk_agent_instance(agent_instance.agent_id, agent_instance)

        # AutogenAgentAdapter.register returns an AgentType (which is a string alias)
        return AgentType(agent_config_from_variant.agent_id)  # Return a mock AgentType

    # Patch AutogenAgentAdapter.register
    with patch("buttermilk.libs.autogen.AutogenAgentAdapter.register", side_effect=mock_adapter_register) as mock_register_call:
        # Minimally mock _runtime and _topic as they are accessed in _register_agents
        orchestrator._runtime = MagicMock(spec=SingleThreadedAgentRuntime)
        orchestrator._topic = DefaultTopicId(type="test_topic")  # Needs to be a TopicId instance

        # Call _register_agents to trigger the mocked registration process
        await orchestrator._register_agents(params=test_run_request)

    # 1. Test registry population
    mock_register_call.assert_called()  # Ensure the mocked register was actually called
    assert expected_agent_id in orchestrator._agent_registry, \
        f"Agent ID {expected_agent_id} not found in registry. Found: {list(orchestrator._agent_registry.keys())}"

    registered_agent = orchestrator._agent_registry[expected_agent_id]
    assert isinstance(registered_agent, MockTestAgent), \
        f"Registered agent is not an instance of MockTestAgent, but {type(registered_agent)}"
    assert registered_agent.agent_id == expected_agent_id
    assert registered_agent.role == agent_config_data["role"]
    assert registered_agent.name == agent_config_data["name"]  # Check name
    assert registered_agent.parameters.get("model") == agent_config_data["parameters"]["model"]  # Check parameters

    # 2. Test get_agent_config for an existing agent
    retrieved_config = orchestrator.get_agent_config(expected_agent_id)
    assert isinstance(retrieved_config, AgentConfig), \
        f"retrieved_config is not AgentConfig, but {type(retrieved_config)}"

    assert retrieved_config.agent_id == expected_agent_id
    assert retrieved_config.role == agent_config_data["role"]
    assert retrieved_config.name == agent_config_data["name"]
    # Check that parameters are correctly retrieved
    assert retrieved_config.parameters.get("model") == agent_config_data["parameters"]["model"]
    assert retrieved_config.parameters.get("some_other_param") == agent_config_data["parameters"]["some_other_param"]
    # Ensure session_id is not part of the *returned* AgentConfig unless it was in original params
    assert "session_id" not in retrieved_config.parameters

    # Test get_agent_config for a non-existent agent
    non_existent_agent_id = "NON_EXISTENT_ID"
    assert orchestrator.get_agent_config(non_existent_agent_id) is None, \
        f"get_agent_config for {non_existent_agent_id} should return None"

    # Test with another agent to ensure registry handles multiple entries
    agent_config2_data = {
        "name": "AnotherAgent",
        "role": "REVIEWER",
        "unique_identifier": "002",
        "parameters": {"style": "detailed"},
        "agent_obj": f"{__name__}.MockTestAgent",
    }
    expected_agent_id2 = f"{agent_config2_data['role']}-{agent_config2_data['unique_identifier']}"
    mock_agent_variant2 = AgentVariants(**agent_config2_data)
    orchestrator.agents["REVIEWER_ROLE_KEY"] = mock_agent_variant2  # Add to orchestrator.agents

    # Re-run parts of the setup for the second agent if _register_agents clears previous state or is selective
    # _register_agents iterates self.agents, so existing ones would be re-processed.
    # The mock_adapter_register needs to be flexible or reset if it holds state.
    # In this case, it's stateless enough.

    # We need to ensure the mock_adapter_register can handle different agent_cls and agent_config
    # The current mock_adapter_register uses `mock_agent_variant` from the outer scope.
    # This needs to be more dynamic if we call _register_agents again for all agents.

    # To test multiple agents, it's better to define all agents in orchestrator.agents
    # before a single call to _register_agents.
    # Let's adjust the test to register multiple agents in one go.

    # Reset orchestrator.agents and re-register
    orchestrator._agent_registry.clear()  # Clear previous registrations for a clean multi-agent test run
    orchestrator.agents = {
        "TESTER_ROLE_KEY": mock_agent_variant,
        "REVIEWER_ROLE_KEY": mock_agent_variant2,
    }

    # The mock_adapter_register needs to be more dynamic to handle different variants.
    # The current side_effect directly calls orchestrator._register_buttermilk_agent_instance
    # based on the variant_config passed into the factory.
    # The factory itself is recreated for each agent variant inside _register_agents.
    # So, the current mock_adapter_register should actually work correctly as it's called
    # for each agent type registration. The key is that the factory it receives
    # will be specific to the agent variant being processed by _register_agents.

    async def dynamic_mock_adapter_register(runtime, type_id_from_register_call, factory_lambda):
        # The factory_lambda is:
        # lambda orch=self, v_cfg=config_with_session, a_cls=agent_cls, t_type=self._topic.type:
        #   agent_factory(orch, cfg=v_cfg, cls=a_cls, topic_type=t_type)
        # We need to execute this factory_lambda to simulate the creation of AutogenAgentAdapter,
        # which in turn creates the Buttermilk Agent and calls the registration callback.

        # The `agent_factory` (inner factory) is defined in AutogenOrchestrator._register_agents
        # It expects: orchestrator_ref, cfg, cls, topic_type
        # The `v_cfg` (config_with_session) and `a_cls` (agent_cls) are crucial.
        # These are captured by the factory_lambda.

        # To truly simulate, we'd need to call factory_lambda, then the agent_factory it returns,
        # then the AutogenAgentAdapter's __init__.
        # However, the current mock directly calls _register_buttermilk_agent_instance.
        # Let's refine the mock to be closer to the actual call sequence.

        # The key is that `factory_lambda` when called, will invoke `agent_factory`
        # which then constructs `AutogenAgentAdapter`. The adapter's `__init__`
        # then calls `_register_buttermilk_agent_instance`.

        # The `type_id_from_register_call` is `variant_config.agent_id`.
        # We can find the corresponding variant_config from orchestrator.agents.

        found_variant_config = None
        found_agent_cls = None

        # This is a bit complex because we need to find which variant_config led to this call.
        # _register_agents iterates through self.agents.items(), then step_config.get_configs()
        # This iteration order determines which variant_config is processed.

        # A simpler approach for the mock: assume the factory_lambda has the correct
        # agent_cls and agent_config (as v_cfg) baked in.
        # The factory_lambda is: (orch, v_cfg, a_cls, t_type) -> agent_factory_call_result
        # The agent_factory_call_result is an AutogenAgentAdapter instance.

        # Let's assume the original mock logic is okay for now, as it ensures the callback is made.
        # The factory passed to AutogenAgentAdapter.register is specific to each agent variant.
        # The `type` argument to `register` is `variant_config.agent_id`.
        # We can use this `type` to find the correct `agent_cls` and `variant_config`.

        processed_agent_cls = None
        processed_variant_config = None

        for role_name_loop, step_config_loop in orchestrator.agents.items():
            # params already includes session_id from the main call
            agent_cls_list_loop, variant_config_list_loop = step_config_loop.get_configs(params=test_run_request)[0]
            if variant_config_list_loop.agent_id == type_id_from_register_call:
                processed_agent_cls = agent_cls_list_loop
                # We need the config that *includes* session_id and orchestrator.parameters
                # This is `config_with_session` in `_register_agents`.
                # For the mock, we can reconstruct it or assume variant_config_list_loop is close enough
                # if AgentVariants merges parameters correctly.
                # Let's assume variant_config_list_loop is what's used to create the agent.
                # The important part is that the agent_id matches.
                config_with_session_for_mock = {
                    **variant_config_list_loop.model_dump(),
                    **orchestrator.parameters,
                    "session_id": test_run_request.session_id,
                }
                agent_instance = processed_agent_cls(**config_with_session_for_mock)
                orchestrator._register_buttermilk_agent_instance(agent_instance.agent_id, agent_instance)
                return AgentType(agent_instance.agent_id)

        raise Exception(f"Mock adapter register could not find agent for type_id: {type_id_from_register_call}")

    with patch("buttermilk.libs.autogen.AutogenAgentAdapter.register", side_effect=dynamic_mock_adapter_register) as mock_register_call_multi:
        orchestrator._runtime = MagicMock(spec=SingleThreadedAgentRuntime)
        orchestrator._topic = DefaultTopicId(type="test_topic_multi")
        await orchestrator._register_agents(params=test_run_request)

    assert mock_register_call_multi.call_count == 2  # Called for each agent variant

    # Check first agent again
    assert expected_agent_id in orchestrator._agent_registry
    registered_agent1 = orchestrator._agent_registry[expected_agent_id]
    assert isinstance(registered_agent1, MockTestAgent)
    assert registered_agent1.agent_id == expected_agent_id
    assert registered_agent1.role == agent_config_data["role"]

    retrieved_config1 = orchestrator.get_agent_config(expected_agent_id)
    assert isinstance(retrieved_config1, AgentConfig)
    assert retrieved_config1.agent_id == expected_agent_id
    assert retrieved_config1.role == agent_config_data["role"]

    # Check second agent
    assert expected_agent_id2 in orchestrator._agent_registry
    registered_agent2 = orchestrator._agent_registry[expected_agent_id2]
    assert isinstance(registered_agent2, MockTestAgent)
    assert registered_agent2.agent_id == expected_agent_id2
    assert registered_agent2.role == agent_config2_data["role"]

    retrieved_config2 = orchestrator.get_agent_config(expected_agent_id2)
    assert isinstance(retrieved_config2, AgentConfig)
    assert retrieved_config2.agent_id == expected_agent_id2
    assert retrieved_config2.role == agent_config2_data["role"]
    assert retrieved_config2.parameters.get("style") == agent_config2_data["parameters"]["style"]
