"""Performance benchmarks for the tool definition system."""

import pytest
import time
import asyncio
from typing import Any
import statistics

from buttermilk._core.agent import Agent
from buttermilk._core.contract import AgentInput, AgentOutput
from buttermilk._core.tool_definition import AgentToolDefinition
from buttermilk._core.mcp_decorators import tool, MCPRoute, extract_tool_definitions
from buttermilk._core.schema_validation import SchemaValidator, validate_tool_input

# Use anyio for async tests
pytestmark = pytest.mark.anyio


class BenchmarkAgent(Agent):
    """Agent with many tools for benchmarking."""
    
    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        return AgentOutput(agent_id=self.agent_id, outputs={"processed": True})
    
    @tool
    def tool_1(self, param: str) -> str:
        return param
    
    @tool
    def tool_2(self, x: int, y: int) -> int:
        return x + y
    
    @tool
    @MCPRoute("/tool3")
    def tool_3(self, data: dict[str, Any]) -> dict[str, Any]:
        return data
    
    @tool
    def tool_4(self, items: list[str]) -> list[str]:
        return items[::-1]
    
    @tool
    @MCPRoute("/tool5", permissions=["read", "write"])
    async def tool_5(self, value: float) -> float:
        await asyncio.sleep(0.001)  # Simulate async work
        return value * 2
    
    # Add more tools to test extraction performance
    @tool
    def tool_6(self, a: str, b: str, c: str) -> str:
        return a + b + c
    
    @tool
    def tool_7(self, enabled: bool) -> bool:
        return not enabled
    
    @tool
    def tool_8(self, config: dict[str, Any]) -> dict[str, Any]:
        return {**config, "processed": True}
    
    @tool
    def tool_9(self, count: int = 10) -> list[int]:
        return list(range(count))
    
    @tool
    @MCPRoute("/tool10")
    def tool_10(self, text: str, repeat: int = 1) -> str:
        return text * repeat


class TestToolExtractionPerformance:
    """Benchmark tool extraction performance."""
    
    def test_tool_extraction_speed(self):
        """Test speed of extracting tool definitions from agent."""
        agent = BenchmarkAgent(agent_name="bench", model_name="test", role="BENCH")
        
        # Warm up
        _ = agent.get_tool_definitions()
        
        # Benchmark
        iterations = 100
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            tools = agent.get_tool_definitions()
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times)
        
        print(f"\nTool extraction performance:")
        print(f"  Average time: {avg_time*1000:.3f}ms")
        print(f"  Std deviation: {std_dev*1000:.3f}ms")
        print(f"  Tools extracted: {len(tools)}")
        
        # Performance assertion - should be fast
        assert avg_time < 0.01  # Less than 10ms on average
        assert len(tools) == 10  # All tools extracted
    
    def test_tool_definition_creation_speed(self):
        """Test speed of creating tool definitions."""
        iterations = 1000
        times = []
        
        for i in range(iterations):
            start = time.perf_counter()
            tool_def = AgentToolDefinition(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                input_schema={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"},
                        "param3": {"type": "boolean"}
                    },
                    "required": ["param1"]
                },
                output_schema={"type": "object"},
                mcp_route=f"/tool{i}",
                permissions=["read", "write"]
            )
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        
        print(f"\nTool definition creation performance:")
        print(f"  Average time: {avg_time*1000:.3f}ms")
        print(f"  Total for {iterations}: {sum(times)*1000:.1f}ms")
        
        # Should be very fast
        assert avg_time < 0.001  # Less than 1ms per definition


class TestSchemaValidationPerformance:
    """Benchmark schema validation performance."""
    
    def test_simple_schema_validation_speed(self):
        """Test validation speed for simple schemas."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            },
            "required": ["name"]
        }
        
        validator = SchemaValidator(schema)
        
        valid_data = {
            "name": "John Doe",
            "age": 30,
            "active": True
        }
        
        # Warm up
        validator.validate(valid_data)
        
        # Benchmark
        iterations = 1000
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            validator.validate(valid_data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        
        print(f"\nSimple schema validation performance:")
        print(f"  Average time: {avg_time*1000:.3f}ms")
        print(f"  Validations per second: {1/avg_time:.0f}")
        
        # Should be very fast
        assert avg_time < 0.001  # Less than 1ms per validation
    
    def test_complex_schema_validation_speed(self):
        """Test validation speed for complex nested schemas."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string", "format": "email"},
                                "tags": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1
                                }
                            },
                            "required": ["name", "email"]
                        },
                        "settings": {
                            "type": "object",
                            "additionalProperties": {"type": "boolean"}
                        }
                    },
                    "required": ["id", "profile"]
                },
                "metadata": {
                    "type": "object",
                    "properties": {
                        "created": {"type": "string"},
                        "updated": {"type": "string"}
                    }
                }
            },
            "required": ["user"]
        }
        
        validator = SchemaValidator(schema)
        
        valid_data = {
            "user": {
                "id": 123,
                "profile": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "tags": ["developer", "python"]
                },
                "settings": {
                    "notifications": True,
                    "dark_mode": False
                }
            },
            "metadata": {
                "created": "2024-01-01",
                "updated": "2024-01-02"
            }
        }
        
        # Benchmark
        iterations = 500
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            validator.validate(valid_data)
            end = time.perf_counter()
            times.append(end - start)
        
        avg_time = statistics.mean(times)
        
        print(f"\nComplex schema validation performance:")
        print(f"  Average time: {avg_time*1000:.3f}ms")
        print(f"  Validations per second: {1/avg_time:.0f}")
        
        # Should still be reasonably fast
        assert avg_time < 0.005  # Less than 5ms per validation





class TestMemoryUsage:
    """Test memory efficiency of tool definitions."""
    
    def test_tool_definition_memory_efficiency(self):
        """Test memory usage of tool definitions."""
        import sys
        
        # Create many tool definitions
        definitions = []
        
        for i in range(1000):
            tool_def = AgentToolDefinition(
                name=f"tool_{i}",
                description=f"This is tool number {i} with a longer description",
                input_schema={
                    "type": "object",
                    "properties": {
                        f"param_{j}": {"type": "string", "description": f"Parameter {j}"}
                        for j in range(5)
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "string"},
                        "metadata": {"type": "object"}
                    }
                },
                mcp_route=f"/api/v1/tools/tool_{i}",
                permissions=["read", "write", "execute"]
            )
            definitions.append(tool_def)
        
        # Estimate memory usage
        total_size = sum(sys.getsizeof(d) for d in definitions)
        avg_size = total_size / len(definitions)
        
        print(f"\nMemory usage for tool definitions:")
        print(f"  Total definitions: {len(definitions)}")
        print(f"  Average size per definition: {avg_size:.0f} bytes")
        print(f"  Total size: {total_size/1024:.1f} KB")
        
        # Should be reasonable
        assert avg_size < 10000  # Less than 10KB per definition


# Run performance tests separately as they may be slow
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])