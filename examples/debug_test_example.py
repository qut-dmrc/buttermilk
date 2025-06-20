"""Example of using the buttermilk debug infrastructure for automated testing."""

import asyncio
from buttermilk.debug import MCPFlowTester


async def main():
    """Example of testing flows using the debug infrastructure."""
    
    # Create MCP client for testing
    async with MCPFlowTester(base_url="http://localhost:8000") as tester:
        
        # 1. Check API health
        print("1. Checking API health...")
        health = await tester.health_check()
        print(f"   API Reachable: {health.api_reachable}")
        if health.api_reachable:
            print(f"   Available flows: {', '.join(health.available_flows)}")
        print()
        
        # 2. Test a simple query
        if health.api_reachable and "osb" in health.available_flows:
            print("2. Testing OSB flow with a query...")
            result = await tester.test_flow_query(
                flow_name="osb",
                query="What are the community guidelines about hate speech?"
            )
            print(f"   Status: {result.status}")
            print(f"   Response time: {result.response_time:.2f}s")
            if result.trace_id:
                print(f"   Trace ID: {result.trace_id}")
            print()
        
        # 3. Run comprehensive test suite
        if health.api_reachable and "osb" in health.available_flows:
            print("3. Running comprehensive test suite...")
            test_queries = [
                "What is hate speech?",
                "Are there guidelines about harassment?",
                "What content is prohibited?"
            ]
            
            suite_result = await tester.test_flow_comprehensive("osb", test_queries)
            print(f"   Total tests: {suite_result.total_tests}")
            print(f"   Passed: {suite_result.passed_tests}")
            print(f"   Failed: {suite_result.failed_tests}")
            print(f"   Average response time: {suite_result.average_response_time:.2f}s")
            
            if suite_result.recommendations:
                print("   Recommendations:")
                for rec in suite_result.recommendations:
                    print(f"   - {rec}")
            print()
        
        # 4. Generate health report
        print("4. Generating flow health report...")
        report = await tester.generate_flow_health_report("osb")
        print(f"   Overall status: {report.overall_status}")
        print(f"   Timestamp: {report.timestamp}")
        
        if not health.api_reachable:
            print("\n⚠️  Note: API is not running. Start it with:")
            print("   uv run python -m buttermilk.runner.cli +flows=[osb] +run=api +llms=full")


if __name__ == "__main__":
    asyncio.run(main())