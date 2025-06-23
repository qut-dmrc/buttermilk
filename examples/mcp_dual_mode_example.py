"""Example demonstrating agents callable via both groupchat and MCP.

This example shows how to:
1. Create agents with @tool decorators
2. Use them in groupchat flows via StructuredLLMHostAgent
3. Expose them as MCP tools for external consumption
4. Invoke them directly via MCP adapter

This addresses issue #94 by demonstrating the unified agent interface.
"""

import asyncio
from typing import Any, Literal

from buttermilk._core import Agent, AgentInput, AgentOutput
from buttermilk._core.mcp_decorators import MCPRoute, tool
from buttermilk.mcp.autogen_adapter import MCPHostProvider
from buttermilk.mcp.tool_registry import get_tool_discovery_service


class CalculatorAgent(Agent):
    """Simple calculator agent with tool decorators."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process general requests by routing to appropriate tools."""
        operation = message.inputs.get("operation", "add")
        
        # Route to specific tool based on operation
        if operation == "add":
            result = self.add(
                a=message.inputs.get("a", 0),
                b=message.inputs.get("b", 0)
            )
        elif operation == "multiply":
            result = self.multiply(
                a=message.inputs.get("a", 1),
                b=message.inputs.get("b", 1)
            )
        elif operation == "factorial":
            result = await self.factorial(
                n=message.inputs.get("n", 1)
            )
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )
    
    @tool
    @MCPRoute("/calculator/add", permissions=["math:basic"])
    def add(self, a: float, b: float) -> float:
        """Add two numbers together.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return a + b
    
    @tool
    @MCPRoute("/calculator/multiply", permissions=["math:basic"])
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return a * b
    
    @tool
    @MCPRoute("/calculator/factorial", permissions=["math:advanced"])
    async def factorial(self, n: int) -> int:
        """Calculate factorial of a number.
        
        Args:
            n: Number to calculate factorial for (must be non-negative)
            
        Returns:
            Factorial of n
            
        Raises:
            ValueError: If n is negative
        """
        if n < 0:
            raise ValueError("Factorial is not defined for negative numbers")
        
        result = 1
        for i in range(1, n + 1):
            result *= i
            # Simulate some async work
            await asyncio.sleep(0.001)
        
        return result


class DataAnalysisAgent(Agent):
    """Data analysis agent with multiple tools."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process data analysis requests."""
        operation = message.inputs.get("operation", "summarize")
        
        if operation == "summarize":
            result = await self.summarize_data(
                data=message.inputs.get("data", []),
                metric=message.inputs.get("metric", "mean")
            )
        elif operation == "filter":
            result = self.filter_data(
                data=message.inputs.get("data", []),
                condition=message.inputs.get("condition", "")
            )
        else:
            result = {"error": f"Unknown operation: {operation}"}
        
        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )
    
    @tool
    @MCPRoute("/analysis/summarize", permissions=["data:read"])
    async def summarize_data(
        self, 
        data: list[float], 
        metric: Literal["mean", "median", "sum", "count"] = "mean"
    ) -> dict[str, Any]:
        """Summarize a dataset using specified metric.
        
        Args:
            data: List of numeric values to summarize
            metric: Type of summary metric to calculate
            
        Returns:
            Summary statistics for the data
        """
        if not data:
            return {"error": "No data provided", "metric": metric}
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        if metric == "mean":
            value = sum(data) / len(data)
        elif metric == "median":
            sorted_data = sorted(data)
            n = len(sorted_data)
            value = (sorted_data[n//2] + sorted_data[(n-1)//2]) / 2
        elif metric == "sum":
            value = sum(data)
        elif metric == "count":
            value = len(data)
        else:
            return {"error": f"Unknown metric: {metric}"}
        
        return {
            "metric": metric,
            "value": value,
            "data_points": len(data),
            "data_range": {"min": min(data), "max": max(data)}
        }
    
    @tool
    def filter_data(
        self, 
        data: list[float], 
        condition: str
    ) -> dict[str, Any]:
        """Filter data based on a condition.
        
        Args:
            data: List of numeric values to filter
            condition: Filter condition (e.g., ">5", "<=10", "==0")
            
        Returns:
            Filtered data and filter statistics
        """
        if not data:
            return {"error": "No data provided"}
        
        try:
            # Simple condition parsing (in production, use safer evaluation)
            if condition.startswith(">="):
                threshold = float(condition[2:])
                filtered = [x for x in data if x >= threshold]
            elif condition.startswith("<="):
                threshold = float(condition[2:])
                filtered = [x for x in data if x <= threshold]
            elif condition.startswith(">"):
                threshold = float(condition[1:])
                filtered = [x for x in data if x > threshold]
            elif condition.startswith("<"):
                threshold = float(condition[1:])
                filtered = [x for x in data if x < threshold]
            elif condition.startswith("=="):
                threshold = float(condition[2:])
                filtered = [x for x in data if x == threshold]
            else:
                return {"error": f"Invalid condition format: {condition}"}
        except ValueError as e:
            return {"error": f"Invalid condition: {e}"}
        
        return {
            "original_count": len(data),
            "filtered_count": len(filtered),
            "filtered_data": filtered,
            "condition": condition
        }


async def demo_groupchat_usage():
    """Demonstrate using agents in a groupchat context."""
    print("\n=== Groupchat Usage Demo ===")
    
    # Create agent instances
    calc_agent = CalculatorAgent(
        agent_name="calculator",
        role="calc",
        session_id="demo-groupchat"
    )
    
    data_agent = DataAnalysisAgent(
        agent_name="data_analyzer", 
        role="analyst",
        session_id="demo-groupchat"
    )
    
    print(f"Created agents: {calc_agent.agent_name}, {data_agent.agent_name}")
    
    # Test calculator agent
    calc_input = AgentInput(inputs={"operation": "add", "a": 5, "b": 3})
    calc_result = await calc_agent._process(message=calc_input)
    print(f"Calculator result: {calc_result.outputs}")
    
    # Test data analysis agent
    data_input = AgentInput(inputs={
        "operation": "summarize",
        "data": [1, 2, 3, 4, 5, 10, 15, 20],
        "metric": "mean"
    })
    data_result = await data_agent._process(message=data_input)
    print(f"Data analysis result: {data_result.outputs}")
    
    return calc_agent, data_agent


async def demo_mcp_usage(calc_agent: Agent, data_agent: Agent):
    """Demonstrate using agents via MCP adapter."""
    print("\n=== MCP Usage Demo ===")
    
    # Initialize MCP host provider
    mcp_provider = MCPHostProvider(port=8788)
    
    # Register agents for MCP exposure
    mcp_provider.register_agent(calc_agent)
    mcp_provider.register_agent(data_agent)
    
    # List available tools
    available_tools = mcp_provider.list_available_tools()
    print(f"Available MCP tools: {list(available_tools.keys())}")
    
    # Test calculator tools via MCP
    print("\nTesting calculator tools via MCP:")
    
    try:
        add_result = await mcp_provider.invoke_tool(
            "calculator_calc.add", 
            {"a": 10, "b": 20}
        )
        print(f"MCP add result: {add_result}")
        
        factorial_result = await mcp_provider.invoke_tool(
            "calculator_calc.factorial",
            {"n": 5}
        )
        print(f"MCP factorial result: {factorial_result}")
        
        # Test data analysis tools via MCP
        print("\nTesting data analysis tools via MCP:")
        
        summary_result = await mcp_provider.invoke_tool(
            "data_analyzer_analyst.summarize_data",
            {
                "data": [1, 5, 10, 15, 20, 25, 30],
                "metric": "median"
            }
        )
        print(f"MCP summarize result: {summary_result}")
        
        filter_result = await mcp_provider.invoke_tool(
            "data_analyzer_analyst.filter_data",
            {
                "data": [1, 5, 10, 15, 20, 25, 30],
                "condition": ">10"
            }
        )
        print(f"MCP filter result: {filter_result}")
        
    except Exception as e:
        print(f"Error during MCP tool invocation: {e}")
    
    return mcp_provider


async def demo_tool_discovery():
    """Demonstrate dynamic tool discovery."""
    print("\n=== Tool Discovery Demo ===")
    
    discovery_service = get_tool_discovery_service()
    
    # Create and register agents
    calc_agent = CalculatorAgent(
        agent_name="calc_v2",
        role="calculator",
        session_id="discovery-demo"
    )
    
    discovery_service.register_agent(calc_agent)
    
    # Show discovered tools
    all_tools = discovery_service.get_all_tools()
    print(f"Discovered {len(all_tools)} tools:")
    for tool in all_tools:
        print(f"  - {tool['key']}: {tool['description']}")
    
    # Test direct tool invocation via discovery service
    print("\nTesting direct tool invocation:")
    result = await discovery_service.invoke_tool(
        "calc_v2_calculator.multiply",
        {"a": 6, "b": 7}
    )
    print(f"Direct invocation result: {result}")


async def demo_unified_request_handling():
    """Demonstrate unified request handling."""
    print("\n=== Unified Request Handling Demo ===")
    
    from buttermilk._core.tool_definition import UnifiedRequest
    
    # Create agent
    calc_agent = CalculatorAgent(
        agent_name="unified_calc",
        role="calculator", 
        session_id="unified-demo"
    )
    
    # Test MCP-style unified request
    mcp_request = UnifiedRequest.from_mcp_call(
        tool_name="add",
        parameters={"a": 100, "b": 200},
        agent_name="unified_calc_calculator"
    )
    
    print(f"MCP request metadata: {mcp_request.metadata}")
    print(f"Is MCP request: {mcp_request.is_mcp_request}")
    
    # Handle the unified request
    result = await calc_agent.handle_unified_request(mcp_request)
    print(f"Unified request result: {result}")
    
    # Test groupchat-style unified request
    groupchat_request = UnifiedRequest.from_groupchat_step(
        agent_role="calculator",
        inputs={"a": 50, "b": 75},
        tool_name="multiply"
    )
    
    print(f"Groupchat request metadata: {groupchat_request.metadata}")
    print(f"Is groupchat request: {groupchat_request.is_groupchat_request}")
    
    result = await calc_agent.handle_unified_request(groupchat_request)
    print(f"Unified request result: {result}")


async def main():
    """Main demo function."""
    print("🚀 Buttermilk MCP Dual-Mode Agent Demo")
    print("=====================================")
    
    # Demo 1: Traditional groupchat usage
    calc_agent, data_agent = await demo_groupchat_usage()
    
    # Demo 2: MCP adapter usage
    mcp_provider = await demo_mcp_usage(calc_agent, data_agent)
    
    # Demo 3: Tool discovery
    await demo_tool_discovery()
    
    # Demo 4: Unified request handling
    await demo_unified_request_handling()
    
    print("\n✅ Demo completed successfully!")
    print("\nKey takeaways:")
    print("- Agents can be used in both groupchat and MCP contexts")
    print("- Tool definitions provide structured interfaces")
    print("- Unified request handling works across both modes")
    print("- Dynamic tool discovery enables flexible architectures")
    print("- Same agent code works in multiple deployment scenarios")


if __name__ == "__main__":
    asyncio.run(main())