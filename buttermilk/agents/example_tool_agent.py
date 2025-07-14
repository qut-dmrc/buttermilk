"""Example agent demonstrating the tool definition system.

This module shows how to expose agent capabilities as structured tool definitions.
"""

from typing import Any, Literal

from buttermilk._core import Agent, AgentInput, AgentOutput

# Tool decorators removed with MCP implementation


class DataAnalysisAgent(Agent):
    """Example agent that demonstrates tool definition capabilities."""

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process incoming requests using the appropriate tool."""
        # Extract the requested operation from inputs
        operation = message.inputs.get("operation", "analyze")

        if operation == "analyze":
            result = await self.analyze_dataset(
                dataset=message.inputs.get("dataset", ""),
                query=message.inputs.get("query", ""),
                output_format=message.inputs.get("output_format", "summary")
            )
        elif operation == "visualize":
            result = await self.visualize_data(
                data=message.inputs.get("data", {}),
                chart_type=message.inputs.get("chart_type", "bar")
            )
        else:
            result = {"error": f"Unknown operation: {operation}"}

        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs=result
        )

    async def analyze_dataset(
        self,
        dataset: str,
        query: str,
        output_format: Literal["table", "chart", "summary"] = "summary"
    ) -> dict[str, Any]:
        """Analyze a dataset using natural language queries.
        
        This tool allows you to perform data analysis on various datasets
        using natural language queries. It supports multiple output formats.
        
        Args:
            dataset: The identifier or path to the dataset to analyze
            query: Natural language query describing the analysis to perform
            output_format: Format for the analysis results
            
        Returns:
            Analysis results in the requested format
        """
        # Placeholder implementation
        return {
            "dataset": dataset,
            "query": query,
            "format": output_format,
            "results": {
                "summary": f"Analysis of {dataset} for query '{query}'",
                "row_count": 1000,
                "insights": [
                    "Insight 1 based on the query",
                    "Insight 2 based on the query"
                ]
            }
        }

    async def visualize_data(
        self,
        data: dict[str, list[float]],
        chart_type: Literal["bar", "line", "scatter", "pie"] = "bar"
    ) -> dict[str, Any]:
        """Generate visualizations from data.
        
        Args:
            data: Dictionary mapping series names to data values
            chart_type: Type of chart to generate
            
        Returns:
            Visualization configuration and rendered chart URL
        """
        return {
            "chart_type": chart_type,
            "data": data,
            "config": {
                "title": "Data Visualization",
                "width": 800,
                "height": 600
            },
            "url": f"https://example.com/charts/{chart_type}_chart.png"
        }

    async def health_check(self) -> dict[str, str]:
        """Check the health status of the agent.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy",
            "agent": self.agent_name,
            "version": "1.0.0"
        }


class CalculatorAgent(Agent):
    """Example agent with simple calculator tools."""

    async def _process(self, *, message: AgentInput, **kwargs: Any) -> AgentOutput:
        """Process calculation requests."""
        operation = message.inputs.get("operation")

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
        else:
            result = {"error": "Unknown operation"}

        return AgentOutput(
            source=self.agent_name,
            role=self.role,
            outputs={"result": result}
        )

    def add(self, a: float, b: float) -> float:
        """Add two numbers together.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Sum of a and b
        """
        return a + b

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers.
        
        Args:
            a: First number
            b: Second number
            
        Returns:
            Product of a and b
        """
        return a * b
