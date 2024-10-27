import pytest
import shortuuid

from buttermilk.automod.mod import JobProcessor
from buttermilk.runner._runner_types import AgentInfo

@pytest.fixture
def run_info():
    return AgentInfo(job='testing', run_id='test', project='dict')

def test_agent_field_validator(run_info):
    # Test case 1: Basic functionality
    worker = JobProcessor(step_name="process", agent="bot", run_info=run_info,flow_obj=dict)
    assert worker.agent.startswith("process_bot_")

    # Test case 2: Verify UUID part
    _, _, uuid_part = worker.agent.split("_")
    assert len(uuid_part) == 6
    assert shortuuid.ShortUUID().is_valid(uuid_part)

    # Test case 3: Different step_name and agent
    worker2 = JobProcessor(step_name="analyze", agent="ai", run_info=run_info,flow_obj=dict)
    assert worker2.agent.startswith("analyze_ai_")

    # Test case 4: Ensure different instances have different UUIDs
    assert worker.agent != worker2.agent

    # Test case 5: Empty strings
    worker3 = JobProcessor(step_name="", agent="", run_info=run_info,flow_obj=dict)
    assert '_' not in worker3.agent

    # Test case 6: Special characters
    worker4 = JobProcessor(step_name="data_processing", agent="ML-bot", run_info=run_info,flow_obj=dict)
    assert worker4.agent.startswith("data_processing_ML-bot_")


def test_agent_field_validator_consistency(run_info):
    # Test case 7: Consistency of UUID generation
    worker = JobProcessor(step_name="test", agent="bot", run_info=run_info,flow_obj=dict)
    original_agent = worker.agent
    
    # Accessing the agent field multiple times should return the same value
    for _ in range(5):
        assert worker.agent == original_agent

@pytest.mark.parametrize("step_name,agent", [
    ("process", "bot"),
    ("analyze", "ai"),
    ("data_processing", "ML-bot"),
])
def test_agent_field_validator_parameterized(step_name, agent, run_info):
    worker = JobProcessor(step_name=step_name, agent=agent, run_info=run_info,flow_obj=dict)
    expected_prefix = f"{step_name}_{agent}_"
    assert worker.agent.startswith(expected_prefix)
    assert len(worker.agent.split("_")[-1]) == 6