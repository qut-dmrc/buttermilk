"""Debug CLI for buttermilk flows."""

import click
import json
import sys
import time
import subprocess
from pathlib import Path

from buttermilk._core.log import logger
from .mcp_client import MCPFlowTester
from .models import StartupTestResult


@click.group()
def debug():
    """Debug utilities for buttermilk flows and components."""
    pass


@debug.command()
@click.option('--flow', multiple=True, help='Flows to test (e.g., osb, trans, tox_allinone)')
@click.option('--timeout', default=60, help='Test timeout in seconds')
@click.option('--output', help='Output file for results (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def test_startup(flow, timeout, output, verbose):
    """Test daemon startup and capture results."""
    flows_str = ','.join(flow) if flow else 'osb'
    
    click.echo(f"🧪 Testing buttermilk daemon startup with flows: {flows_str}")
    click.echo(f"⏱️  Timeout: {timeout}s")
    
    start_time = time.time()
    validation_errors = []
    startup_success = False
    api_reachable = False
    flows_loaded = []
    error_details = None
    
    # Build command
    cmd = [
        'uv', 'run', 'python', '-m', 'buttermilk.runner.cli',
        f'+flows=[{flows_str}]',
        '+run=api',
        '+llms=full'
    ]
    
    if verbose:
        click.echo(f"🚀 Running: {' '.join(cmd)}")
    
    try:
        # Start process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd()
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            if not line:
                break
                
            output_lines.append(line.strip())
            
            if verbose:
                click.echo(f"📝 {line.strip()}")
            
            # Check for validation errors
            if "ValidationError" in line and "validation errors" in line:
                validation_errors.append(line.strip())
            elif "ValidationError" in line:
                validation_errors.append(line.strip())
            
            # Check for startup success
            if "Application startup complete" in line:
                startup_success = True
                if verbose:
                    click.echo("✅ Application startup complete detected")
            
            # Check for server start
            if "Uvicorn running on" in line:
                if verbose:
                    click.echo("🌐 API server running detected")
                # Give a moment for server to be ready
                time.sleep(2)
                
                # Test API health
                client = MCPFlowTester()
                health = client.health_check()
                api_reachable = health.api_reachable
                flows_loaded = health.available_flows
                
                if api_reachable:
                    click.echo("✅ API health check passed")
                    if verbose and flows_loaded:
                        click.echo(f"📊 Available flows: {', '.join(flows_loaded)}")
                else:
                    click.echo("❌ API health check failed")
                
                break
            
            # Check for fatal errors
            if "Error executing job" in line:
                error_details = line.strip()
                break
            
            # Timeout check
            if time.time() - start_time > timeout:
                click.echo(f"⏰ Timeout reached ({timeout}s)")
                break
        
        # Cleanup
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    
    except Exception as e:
        error_details = str(e)
        click.echo(f"❌ Exception during startup test: {e}")
    
    startup_time = time.time() - start_time
    
    # Create result
    result = StartupTestResult(
        startup_time=startup_time,
        startup_success=startup_success,
        validation_errors=validation_errors,
        api_reachable=api_reachable,
        flows_loaded=flows_loaded,
        error_details=error_details
    )
    
    # Display summary
    click.echo("\\n" + "="*60)
    click.echo("📊 STARTUP TEST SUMMARY")
    click.echo("="*60)
    click.echo(f"⏱️  Startup time: {startup_time:.2f}s")
    click.echo(f"🚀 Startup success: {'✅' if startup_success else '❌'}")
    click.echo(f"🌐 API reachable: {'✅' if api_reachable else '❌'}")
    click.echo(f"📋 Validation errors: {len(validation_errors)}")
    click.echo(f"🔄 Flows loaded: {len(flows_loaded)}")
    
    if validation_errors:
        click.echo("\\n❌ VALIDATION ERRORS:")
        for error in validation_errors[:5]:  # Show first 5
            click.echo(f"   • {error}")
        if len(validation_errors) > 5:
            click.echo(f"   ... and {len(validation_errors) - 5} more")
    
    if flows_loaded:
        click.echo(f"\\n✅ LOADED FLOWS: {', '.join(flows_loaded)}")
    
    if error_details:
        click.echo(f"\\n❌ ERROR DETAILS: {error_details}")
    
    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(result.model_dump_json(indent=2))
        click.echo(f"\\n💾 Results saved to: {output_path}")
    
    # Set exit code
    if not startup_success or not api_reachable:
        sys.exit(1)


@debug.command()
@click.option('--flow', required=True, help='Flow name to test (e.g., osb)')
@click.option('--query', required=True, help='Test query to send')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--timeout', default=30, help='Request timeout in seconds')
@click.option('--output', help='Output file for results (JSON)')
def test_mcp_query(flow, query, host, port, timeout, output):
    """Test flow via existing MCP endpoints."""
    base_url = f"http://{host}:{port}"
    client = MCPFlowTester(base_url=base_url, timeout=timeout)
    
    click.echo(f"🧪 Testing {flow} flow with MCP endpoints")
    click.echo(f"🌐 API: {base_url}")
    click.echo(f"❓ Query: {query}")
    
    # Health check first
    click.echo("\\n🏥 Checking API health...")
    health = client.health_check()
    
    if not health.api_reachable:
        click.echo(f"❌ API not reachable: {health.error_details}")
        sys.exit(1)
    
    click.echo(f"✅ API healthy (response time: {health.response_time:.2f}s)")
    if health.available_flows:
        click.echo(f"📊 Available flows: {', '.join(health.available_flows)}")
    
    # Test the specific query
    if flow == "osb":
        click.echo("\\n🔍 Testing OSB vector query...")
        result = client.test_osb_vector_query(query)
    else:
        click.echo(f"\\n🚀 Testing flow start for {flow}...")
        result = client.test_mcp_flow_start(flow, query)
    
    # Display results
    click.echo("\\n" + "="*60)
    click.echo("📊 QUERY TEST RESULTS")
    click.echo("="*60)
    click.echo(f"🎯 Status: {'✅' if result.status == 'success' else '❌'} {result.status}")
    click.echo(f"⏱️  Response time: {result.response_time:.2f}s")
    click.echo(f"🔗 Endpoint: {result.endpoint}")
    
    if result.status == "success" and result.response:
        response = result.response
        if "result" in response:
            click.echo(f"\\n✅ RESPONSE:")
            result_data = response["result"]
            
            # Handle different response types
            if isinstance(result_data, dict):
                if "content" in result_data:
                    click.echo(f"📝 Content: {result_data['content'][:200]}...")
                if "search_results" in result_data:
                    search_results = result_data["search_results"]
                    click.echo(f"🔍 Found {len(search_results)} search results")
                    for i, result in enumerate(search_results[:3]):
                        title = result.get("title", "No title")
                        score = result.get("score", 0)
                        click.echo(f"   {i+1}. {title} (score: {score:.3f})")
            else:
                click.echo(f"📄 Result: {str(result_data)[:200]}...")
        
        if "trace_id" in response:
            click.echo(f"🔍 Trace ID: {response['trace_id']}")
    
    elif result.error_details:
        click.echo(f"\\n❌ ERROR: {result.error_details}")
    
    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(result.model_dump_json(indent=2))
        click.echo(f"\\n💾 Results saved to: {output_path}")
    
    # Set exit code
    if result.status != "success":
        sys.exit(1)


@debug.command()
@click.option('--flow', required=True, help='Flow name to test comprehensively')
@click.option('--queries-file', help='JSON file with test queries')
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--timeout', default=60, help='Request timeout in seconds')
@click.option('--output', help='Output file for results (JSON)')
def test_flow_comprehensive(flow, queries_file, host, port, timeout, output):
    """Run comprehensive test suite for a flow."""
    base_url = f"http://{host}:{port}"
    client = MCPFlowTester(base_url=base_url, timeout=timeout)
    
    # Load test queries
    test_queries = None
    if queries_file:
        queries_path = Path(queries_file)
        if queries_path.exists():
            test_queries = json.loads(queries_path.read_text())
        else:
            click.echo(f"❌ Queries file not found: {queries_file}")
            sys.exit(1)
    
    click.echo(f"🧪 Running comprehensive test suite for {flow} flow")
    click.echo(f"🌐 API: {base_url}")
    
    # Run comprehensive test
    report = client.test_flow_comprehensive(flow, test_queries)
    
    # Display results
    click.echo("\\n" + "="*60)
    click.echo("📊 COMPREHENSIVE TEST REPORT")
    click.echo("="*60)
    click.echo(f"🎯 Flow: {report.flow_name}")
    click.echo(f"🏥 Overall Status: {report.overall_status}")
    click.echo(f"🌐 API Health: {'✅' if report.api_health.api_reachable else '❌'}")
    click.echo(f"🧪 MCP Tests: {len([r for r in report.mcp_test_results if r.status == 'success'])}/{len(report.mcp_test_results)} passed")
    click.echo(f"🤖 Agent Tests: {len([r for r in report.agent_tests if r.status == 'success'])}/{len(report.agent_tests)} passed")
    
    if report.mcp_test_results:
        avg_response_time = sum(r.response_time for r in report.mcp_test_results) / len(report.mcp_test_results)
        click.echo(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
    
    # Show failed tests
    failed_mcp = [r for r in report.mcp_test_results if r.status != "success"]
    if failed_mcp:
        click.echo(f"\\n❌ FAILED MCP TESTS ({len(failed_mcp)}):")
        for test in failed_mcp:
            click.echo(f"   • {test.endpoint}: {test.error_details}")
    
    failed_agents = [r for r in report.agent_tests if r.status != "success"]
    if failed_agents:
        click.echo(f"\\n❌ FAILED AGENT TESTS ({len(failed_agents)}):")
        for test in failed_agents:
            click.echo(f"   • {test.agent_role}: {test.error_details}")
    
    # Show recommendations
    if report.recommendations:
        click.echo(f"\\n💡 RECOMMENDATIONS:")
        for rec in report.recommendations:
            click.echo(f"   • {rec}")
    
    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(report.model_dump_json(indent=2))
        click.echo(f"\\n💾 Results saved to: {output_path}")
    
    # Set exit code based on overall status
    if report.overall_status == "unhealthy":
        sys.exit(1)


if __name__ == '__main__':
    debug()