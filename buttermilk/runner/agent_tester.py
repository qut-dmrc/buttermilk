"""
Command-line utility for testing individual agents.

This module provides a CLI for testing individual agents in isolation to detect
potential issues before they appear in full batch runs. It can help diagnose
problems like Pydantic model compatibility, API rate limits, and weave integration.
"""

import argparse
import asyncio
import importlib
import json
import logging
import sys
from pathlib import Path

from buttermilk._core.agent import Agent
from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import AgentInput, ConductorRequest, AgentOutput
from buttermilk._core.types import Record
from buttermilk.bm import bm, logger
from buttermilk.libs.autogen import AutogenAgentAdapter


async def load_agent_class(agent_path):
    """
    Dynamically load an agent class from a path like 'buttermilk.agents.evaluators.scorer.LLMScorer'.
    
    Args:
        agent_path: Fully qualified path to the agent class
        
    Returns:
        The agent class
    """
    try:
        module_path, class_name = agent_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        agent_cls = getattr(module, class_name)
        return agent_cls
    except (ValueError, ImportError, AttributeError) as e:
        logger.error(f"Failed to load agent class {agent_path}: {e}")
        raise


async def create_agent_config(role, name, params=None):
    """
    Create an agent configuration.
    
    Args:
        role: Agent role
        name: Agent name
        params: Dictionary of parameters
        
    Returns:
        AgentConfig object
    """
    return AgentConfig(
        role=role,
        name=name,
        description=f"Test {name}",
        parameters=params or {}
    )


async def run_agent_test(agent_cls, agent_config, input_msg, use_adapter=True):
    """
    Run a test on an individual agent.
    
    Args:
        agent_cls: The agent class
        agent_config: Agent configuration
        input_msg: Input message for the agent
        use_adapter: Whether to use the AutogenAgentAdapter
        
    Returns:
        Agent output
    """
    if use_adapter:
        # Test through the adapter layer (more realistic)
        adapter = AutogenAgentAdapter(
            topic_type="test_topic",
            agent_cls=agent_cls,
            agent_cfg=agent_config,
        )
        await adapter.agent.initialize()
        
        result = await adapter.agent(
            message=input_msg,
            cancellation_token=None,
            public_callback=lambda msg: logger.info(f"Agent published: {type(msg).__name__}"),
            message_callback=lambda msg: logger.info(f"Agent callback: {type(msg).__name__}"),
            source="test_runner"
        )
        return result
    else:
        # Test the agent directly
        agent = agent_cls(**agent_config.model_dump())
        await agent.initialize()
        
        result = await agent(
            message=input_msg,
            cancellation_token=None,
            public_callback=lambda msg: logger.info(f"Agent published: {type(msg).__name__}"),
            message_callback=lambda msg: logger.info(f"Agent callback: {type(msg).__name__}"),
            source="test_runner"
        )
        return result


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        handlers=[logging.StreamHandler()]
    )


async def test_agent_from_cli():
    """Run an agent test from command line arguments."""
    parser = argparse.ArgumentParser(description="Test an individual Buttermilk agent")
    
    parser.add_argument(
        "agent_class", 
        help="Fully qualified agent class (e.g., buttermilk.agents.evaluators.scorer.LLMScorer)"
    )
    
    parser.add_argument(
        "--role", 
        default="TEST", 
        help="Role for the agent"
    )
    
    parser.add_argument(
        "--name", 
        default="TestAgent", 
        help="Name for the agent"
    )
    
    parser.add_argument(
        "--params", 
        type=json.loads, 
        default={}, 
        help="JSON string of parameters for the agent"
    )
    
    parser.add_argument(
        "--input", 
        type=json.loads,
        default={"prompt": "Test input"}, 
        help="JSON string of input for the agent"
    )
    
    parser.add_argument(
        "--input-type",
        choices=["agent_input", "conductor_request"],
        default="agent_input",
        help="Type of input to create"
    )
    
    parser.add_argument(
        "--records", 
        type=json.loads, 
        default=None, 
        help="JSON string of records to include"
    )
    
    parser.add_argument(
        "--no-adapter", 
        action="store_true", 
        help="Test agent directly without the adapter"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load the agent class
        logger.info(f"Loading agent class {args.agent_class}...")
        agent_cls = await load_agent_class(args.agent_class)
        
        # Create agent config
        logger.info(f"Creating agent config with role={args.role}, name={args.name}, params={args.params}...")
        agent_config = await create_agent_config(args.role, args.name, args.params)
        
        # Create input message
        if args.input_type == "agent_input":
            if args.records:
                records = [Record(**r) for r in args.records]
            else:
                records = None
                
            input_msg = AgentInput(
                inputs=args.input,
                prompt=args.input.get("prompt", "Test input"),
                records=records
            )
        else:  # conductor_request
            input_msg = ConductorRequest(
                inputs=args.input,
                prompt=args.input.get("prompt", "Test input"),
                records=[Record(**r) for r in args.records] if args.records else None
            )
        
        logger.info(f"Created input message of type {type(input_msg).__name__}")
        
        # Run the test
        logger.info(f"Running agent test {'directly' if args.no_adapter else 'with adapter'}...")
        result = await run_agent_test(
            agent_cls=agent_cls,
            agent_config=agent_config,
            input_msg=input_msg,
            use_adapter=not args.no_adapter
        )
        
        # Process result
        if result:
            logger.info(f"Agent returned result of type: {type(result).__name__}")
            
            if isinstance(result, AgentOutput):
                logger.info(f"Agent role: {result.role}")
                logger.info(f"Error status: {'Error' if result.is_error else 'Success'}")
                
                if result.is_error:
                    logger.error(f"Error message: {result.error}")
                else:
                    logger.info(f"Output type: {type(result.outputs).__name__}")
                    
                    # Try to show output as JSON if possible
                    try:
                        if hasattr(result.outputs, "model_dump"):
                            output_data = result.outputs.model_dump()
                        elif hasattr(result.outputs, "dict"):
                            output_data = result.outputs.dict()
                        else:
                            output_data = result.outputs
                            
                        json_output = json.dumps(output_data, indent=2, default=str)
                        logger.info(f"Output:\n{json_output}")
                    except Exception as e:
                        logger.info(f"Output: {result.outputs} (could not convert to JSON: {e})")
            else:
                # Try to show as JSON
                try:
                    json_output = json.dumps(result, indent=2, default=str)
                    logger.info(f"Result:\n{json_output}")
                except Exception:
                    logger.info(f"Result: {result}")
                    
        else:
            logger.warning("Agent returned no result")
            
        return 0
        
    except Exception as e:
        logger.exception(f"Error running agent test: {e}")
        return 1


def main():
    """Main entry point."""
    sys.exit(asyncio.run(test_agent_from_cli()))


if __name__ == "__main__":
    main()
