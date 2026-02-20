"""Debug CLI for buttermilk flows."""

import subprocess
import sys
import time
from pathlib import Path

import click

from .config_validator import validate_configuration
from .error_capture import analyze_type_checking_errors
from .gcp_logs import GCPLogAnalyzer
from .models import StartupTestResult
from .trace_analysis import (
    load_trace_file,
    get_errors,
    get_timeline,
    get_traces_by_agent,
    get_trace,
    summarize,
    get_llm_conversation,
    get_inputs_outputs,
)


@click.group()
def debug():
    """Debug utilities for buttermilk flows and components."""


@debug.command()
@click.option(
    "--flow", multiple=True, help="Flows to test (e.g., osb, trans, tox_allinone)"
)
@click.option("--timeout", default=60, help="Test timeout in seconds")
@click.option("--output", help="Output file for results (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test_startup(flow, timeout, output, verbose):
    """Test daemon startup and capture results."""
    flows_str = ",".join(flow) if flow else "osb"

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
        "uv",
        "run",
        "python",
        "-m",
        "buttermilk.runner.cli",
        f"+flows=[{flows_str}]",
        "run=api",
        "+llms=full",
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
            cwd=Path.cwd(),
        )

        output_lines = []
        for line in iter(process.stdout.readline, ""):
            if not line:
                break

            output_lines.append(line.strip())

            if verbose:
                click.echo(f"📝 {line.strip()}")

            # Check for validation errors
            if (
                "ValidationError" in line and "validation errors" in line
            ) or "ValidationError" in line:
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
                # client = MCPFlowTester()
                # health = client.health_check()
                # api_reachable = health.api_reachable
                # flows_loaded = health.available_flows
                api_reachable = False  # MCP client removed
                flows_loaded = []

                # if api_reachable:
                #     click.echo("✅ API health check passed")
                #     if verbose and flows_loaded:
                #         click.echo(f"📊 Available flows: {', '.join(flows_loaded)}")
                # else:
                #     click.echo("❌ API health check failed")
                click.echo("⚠️  API health check skipped (MCP client removed)")

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
        error_details=error_details,
    )

    # Display summary
    click.echo("\\n" + "=" * 60)
    click.echo("📊 STARTUP TEST SUMMARY")
    click.echo("=" * 60)
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


# @debug.command()
# @click.option('--flow', required=True, help='Flow name to test (e.g., osb)')
# @click.option('--query', required=True, help='Test query to send')
# @click.option('--host', default='localhost', help='API host')
# @click.option('--port', default=8000, help='API port')
# @click.option('--timeout', default=30, help='Request timeout in seconds')
# @click.option('--output', help='Output file for results (JSON)')
# def test_mcp_query(flow, query, host, port, timeout, output):
#     """Test flow via existing MCP endpoints."""
#     base_url = f"http://{host}:{port}"
#     client = MCPFlowTester(base_url=base_url, timeout=timeout)
#
#     click.echo(f"🧪 Testing {flow} flow with MCP endpoints")
#     click.echo(f"🌐 API: {base_url}")
#     click.echo(f"❓ Query: {query}")
#
#     # Health check first
#     click.echo("\\n🏥 Checking API health...")
#     health = client.health_check()
#
#     if not health.api_reachable:
#         click.echo(f"❌ API not reachable: {health.error_details}")
#         sys.exit(1)
#
#     click.echo(f"✅ API healthy (response time: {health.response_time:.2f}s)")
#     if health.available_flows:
#         click.echo(f"📊 Available flows: {', '.join(health.available_flows)}")
#
#     # Test the specific query
#     if flow == "osb":
#         click.echo("\\n🔍 Testing OSB vector query...")
#         result = client.test_osb_vector_query(query)
#     else:
#         click.echo(f"\\n🚀 Testing flow start for {flow}...")
#         result = client.test_mcp_flow_start(flow, query)
#
#     # Display results
#     click.echo("\\n" + "="*60)
#     click.echo("📊 QUERY TEST RESULTS")
#     click.echo("="*60)
#     click.echo(f"🎯 Status: {'✅' if result.status == 'success' else '❌'} {result.status}")
#     click.echo(f"⏱️  Response time: {result.response_time:.2f}s")
#     click.echo(f"🔗 Endpoint: {result.endpoint}")
#
#     if result.status == "success" and result.response:
#         response = result.response
#         if "result" in response:
#             click.echo(f"\\n✅ RESPONSE:")
#             result_data = response["result"]
#
#             # Handle different response types
#             if isinstance(result_data, dict):
#                 if "content" in result_data:
#                     click.echo(f"📝 Content: {result_data['content'][:200]}...")
#                 if "search_results" in result_data:
#                     search_results = result_data["search_results"]
#                     click.echo(f"🔍 Found {len(search_results)} search results")
#                     for i, result in enumerate(search_results[:3]):
#                         title = result.get("title", "No title")
#                         score = result.get("score", 0)
#                         click.echo(f"   {i+1}. {title} (score: {score:.3f})")
#             else:
#                 click.echo(f"📄 Result: {str(result_data)[:200]}...")
#
#         if "trace_id" in response:
#             click.echo(f"🔍 Trace ID: {response['trace_id']}")
#
#     elif result.error_details:
#         click.echo(f"\\n❌ ERROR: {result.error_details}")
#
#     # Save output if requested
#     if output:
#         output_path = Path(output)
#         output_path.write_text(result.model_dump_json(indent=2))
#         click.echo(f"\\n💾 Results saved to: {output_path}")
#
#     # Set exit code
#     if result.status != "success":
#         sys.exit(1)
#

# @debug.command()
# @click.option('--flow', required=True, help='Flow name to test comprehensively')
# @click.option('--queries-file', help='JSON file with test queries')
# @click.option('--host', default='localhost', help='API host')
# @click.option('--port', default=8000, help='API port')
# @click.option('--timeout', default=60, help='Request timeout in seconds')
# @click.option('--output', help='Output file for results (JSON)')
# def test_flow_comprehensive(flow, queries_file, host, port, timeout, output):
#     """Run comprehensive test suite for a flow."""
#     base_url = f"http://{host}:{port}"
#     client = MCPFlowTester(base_url=base_url, timeout=timeout)
#
#     # Load test queries
#     test_queries = None
#     if queries_file:
#         queries_path = Path(queries_file)
#         if queries_path.exists():
#             test_queries = json.loads(queries_path.read_text())
#         else:
#             click.echo(f"❌ Queries file not found: {queries_file}")
#             sys.exit(1)
#
#     click.echo(f"🧪 Running comprehensive test suite for {flow} flow")
#     click.echo(f"🌐 API: {base_url}")
#
#     # Run comprehensive test
#     report = client.test_flow_comprehensive(flow, test_queries)
#
#     # Display results
#     click.echo("\\n" + "="*60)
#     click.echo("📊 COMPREHENSIVE TEST REPORT")
#     click.echo("="*60)
#     click.echo(f"🎯 Flow: {report.flow_name}")
#     click.echo(f"🏥 Overall Status: {report.overall_status}")
#     click.echo(f"🌐 API Health: {'✅' if report.api_health.api_reachable else '❌'}")
#     click.echo(f"🧪 MCP Tests: {len([r for r in report.mcp_test_results if r.status == 'success'])}/{len(report.mcp_test_results)} passed")
#     click.echo(f"🤖 Agent Tests: {len([r for r in report.agent_tests if r.status == 'success'])}/{len(report.agent_tests)} passed")
#
#     if report.mcp_test_results:
#         avg_response_time = sum(r.response_time for r in report.mcp_test_results) / len(report.mcp_test_results)
#         click.echo(f"⏱️  Average Response Time: {avg_response_time:.2f}s")
#
#     # Show failed tests
#     failed_mcp = [r for r in report.mcp_test_results if r.status != "success"]
#     if failed_mcp:
#         click.echo(f"\\n❌ FAILED MCP TESTS ({len(failed_mcp)}):")
#         for test in failed_mcp:
#             click.echo(f"   • {test.endpoint}: {test.error_details}")
#
#     failed_agents = [r for r in report.agent_tests if r.status != "success"]
#     if failed_agents:
#         click.echo(f"\\n❌ FAILED AGENT TESTS ({len(failed_agents)}):")
#         for test in failed_agents:
#             click.echo(f"   • {test.agent_role}: {test.error_details}")
#
#     # Show recommendations
#     if report.recommendations:
#         click.echo(f"\\n💡 RECOMMENDATIONS:")
#         for rec in report.recommendations:
#             click.echo(f"   • {rec}")
#
#     # Save output if requested
#     if output:
#         output_path = Path(output)
#         output_path.write_text(report.model_dump_json(indent=2))
#         click.echo(f"\\n💾 Results saved to: {output_path}")
#
#     # Set exit code based on overall status
#     if report.overall_status == "unhealthy":
#         sys.exit(1)
#


@debug.command()
@click.option(
    "--minutes-back", default=30, help="How many minutes back to analyze logs"
)
@click.option("--project-id", help="GCP project ID (auto-detected if not provided)")
@click.option(
    "--include-warnings", is_flag=True, help="Include warning-level logs in analysis"
)
def analyze_logs(minutes_back, project_id, include_warnings):
    """Analyze GCP logs for Enhanced RAG agent and startup issues."""
    click.echo("🔍 ANALYZING GCP LOGS FOR BUTTERMILK ISSUES")
    click.echo("=" * 50)

    analyzer = GCPLogAnalyzer(project_id=project_id)

    # Check if gcloud is available
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.echo("❌ gcloud CLI not found. Please install Google Cloud SDK.")
        sys.exit(1)

    # Analyze startup issues
    click.echo(f"📊 Analyzing daemon startup (last {minutes_back} minutes)...")
    startup_analysis = analyzer.analyze_daemon_startup(minutes_back=minutes_back)

    click.echo(f"   Total log entries: {startup_analysis.total_entries}")
    click.echo(f"   Errors: {startup_analysis.error_count}")
    click.echo(f"   Warnings: {startup_analysis.warning_count}")

    if startup_analysis.startup_issues:
        click.echo("\n🚨 STARTUP ISSUES:")
        for issue in startup_analysis.startup_issues[:5]:
            click.echo(f"   {issue}")

    if startup_analysis.key_errors:
        click.echo("\n❌ KEY ERRORS:")
        for error in startup_analysis.key_errors[:3]:
            click.echo(f"   {error}")

    # Analyze Enhanced RAG specific issues
    click.echo("\n🎯 Analyzing Enhanced RAG agent issues...")
    rag_analysis = analyzer.analyze_agent_errors("enhanced", minutes_back=minutes_back)

    if rag_analysis.agent_errors:
        click.echo(f"   Found {len(rag_analysis.agent_errors)} agent errors")
        for error in rag_analysis.agent_errors[:3]:
            click.echo(f"   {error}")
    else:
        click.echo("   No Enhanced RAG specific errors found")

    # Show type checking analysis
    type_analysis = analyze_type_checking_errors()
    if type_analysis["total_type_errors"] > 0:
        click.echo(f"\n💻 TYPE CHECKING ERRORS: {type_analysis['total_type_errors']}")
        for rec in type_analysis["recommendations"]:
            click.echo(f"   Issue: {rec['issue']}")
            click.echo(f"   Fix: {rec['fix']}")


@debug.command()
@click.option("--filter", help="Log filter expression for GCP logs")
def stream_logs(filter):
    """Stream GCP logs in real-time for debugging."""
    click.echo("🔍 STREAMING GCP LOGS (Press Ctrl+C to stop)")
    click.echo("=" * 50)

    analyzer = GCPLogAnalyzer()

    # Check if gcloud is available
    try:
        subprocess.run(["gcloud", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        click.echo("❌ gcloud CLI not found. Please install Google Cloud SDK.")
        sys.exit(1)

    try:
        analyzer.stream_logs_realtime(filter_expression=filter)
    except KeyboardInterrupt:
        click.echo("\n✅ Log streaming stopped")


@debug.command()
@click.option("--flow", default="osb", help="Flow to test and debug")
@click.option(
    "--query",
    default="Does hate speech have to be explicit to be prohibited?",
    help="Test query",
)
@click.option("--logs-minutes", default=15, help="Minutes of logs to analyze")
@click.option("--comprehensive", is_flag=True, help="Run full debugging suite")
def diagnose_issue(flow, query, logs_minutes, comprehensive):
    """Complete diagnostic suite for the Enhanced RAG agent issue."""
    click.echo("🩺 COMPLETE ENHANCED RAG AGENT DIAGNOSTIC")
    click.echo("=" * 60)

    # Step 1: Analyze recent logs
    click.echo("1️⃣ Analyzing recent GCP logs...")
    analyzer = GCPLogAnalyzer()
    startup_analysis = analyzer.analyze_daemon_startup(minutes_back=logs_minutes)

    if startup_analysis.startup_issues:
        click.echo(f"   🚨 Found {len(startup_analysis.startup_issues)} startup issues")
        for issue in startup_analysis.startup_issues[:3]:
            click.echo(f"      {issue}")
    else:
        click.echo("   ✅ No startup issues in recent logs")

    # Step 2: Test API health
    click.echo("\n2️⃣ Testing API health...")
    # client = MCPFlowTester()
    # health = client.health_check()
    click.echo("   ⚠️  API health check skipped (MCP client removed)")

    # if health.api_reachable:
    #     click.echo(f"   ✅ API reachable (response time: {health.response_time:.2f}s)")
    #     click.echo(f"   📊 Available flows: {', '.join(health.available_flows)}")
    # else:
    #     click.echo(f"   ❌ API not reachable: {health.error_details}")

    # Try to start daemon if not running
    click.echo("\n🚀 Attempting to start daemon...")
    try:
        start_cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "buttermilk.runner.cli",
            f"+flows=[{flow}]",
            "run=api",
            "+llms=full",
        ]

        click.echo(f"   Running: {' '.join(start_cmd)}")

        # Start in background and check for immediate errors
        process = subprocess.Popen(
            start_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Check first few lines for immediate errors
        for i in range(10):
            line = process.stdout.readline()
            if not line:
                break

            click.echo(f"   📝 {line.strip()}")

            if "Enhanced RAG agent error" in line:
                click.echo(f"   🎯 Found Enhanced RAG error: {line.strip()}")
                break
            if "Subscripted generics" in line:
                click.echo(f"   🎯 Found type checking error: {line.strip()}")
                break
            if "ValidationError" in line:
                click.echo(f"   🎯 Found validation error: {line.strip()}")
                break
            if "Application startup complete" in line:
                click.echo("   ✅ Startup appears successful")
                break

        # Cleanup
        process.terminate()

    except Exception as e:
        click.echo(f"   ❌ Failed to start daemon: {e}")

    # Step 3: Test specific flow if API is running
    # if health.api_reachable and flow in health.available_flows:
    #     click.echo(f"\n3️⃣ Testing {flow} flow with query...")
    #     click.echo(f"   Query: {query}")
    #
    #     if flow == "osb":
    #         result = client.test_osb_vector_query(query)
    #     else:
    #         result = client.test_mcp_flow_start(flow, query)
    #
    #     if result.status == "success":
    #         click.echo(f"   ✅ Flow executed successfully ({result.response_time:.2f}s)")
    #     else:
    #         click.echo(f"   ❌ Flow failed: {result.error_details}")
    click.echo("\n3️⃣ Flow testing skipped (MCP client removed)")

    # Step 4: Analyze type checking errors
    click.echo("\n4️⃣ Analyzing type checking errors...")
    type_analysis = analyze_type_checking_errors()

    if type_analysis["total_type_errors"] > 0:
        click.echo(
            f"   🚨 Found {type_analysis['total_type_errors']} type checking errors"
        )
        for rec in type_analysis["recommendations"]:
            click.echo(f"   💡 {rec['issue']}: {rec['fix']}")
    else:
        click.echo("   ✅ No type checking errors captured")

    # Step 5: Comprehensive testing if requested
    if comprehensive:
        click.echo("\n5️⃣ Running comprehensive flow test...")
        # if health.api_reachable and flow in health.available_flows:
        #     report = client.test_flow_comprehensive(flow, [query])
        #     click.echo(f"   Overall Status: {report.overall_status}")
        #
        #     failed_tests = [r for r in report.mcp_test_results if r.status != "success"]
        #     if failed_tests:
        #         click.echo(f"   ❌ {len(failed_tests)} tests failed")
        #         for test in failed_tests[:3]:
        #             click.echo(f"      {test.endpoint}: {test.error_details}")
        # else:
        #     click.echo("   ⏭️ Skipping comprehensive tests (API not available)")
        click.echo("   ⚠️  Comprehensive tests skipped (MCP client removed)")

    # Summary and recommendations
    click.echo("\n📋 SUMMARY AND RECOMMENDATIONS")
    click.echo("=" * 40)

    if startup_analysis.startup_issues:
        click.echo("🔧 Next steps for startup issues:")
        click.echo("   1. Check Enhanced RAG agent initialization code")
        click.echo("   2. Look for isinstance() calls with subscripted generics")
        click.echo("   3. Add error capture to agent initialization")

    # if not health.api_reachable:
    #     click.echo("🔧 Next steps for API issues:")
    #     click.echo("   1. Check daemon startup logs with --verbose")
    #     click.echo("   2. Verify configuration files are valid")
    #     click.echo("   3. Test with minimal configuration first")

    if type_analysis["total_type_errors"] > 0:
        click.echo("🔧 Next steps for type checking:")
        click.echo(
            "   1. Replace isinstance(obj, List[str]) with isinstance(obj, list)"
        )
        click.echo("   2. Use TYPE_CHECKING guard for typing-only imports")
        click.echo("   3. Test fixes with isolated unit tests")


@debug.command()
@click.option(
    "--config-path",
    default="/workspaces/buttermilk/conf",
    help="Path to configuration directory",
)
@click.option("--output", help="Output file for validation report (JSON)")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed validation results")
def validate_config(config_path, output, verbose):
    """Validate configuration files and dependencies."""
    click.echo("🔧 VALIDATING BUTTERMILK CONFIGURATION")
    click.echo("=" * 50)

    # Run validation
    report = validate_configuration(config_path)

    # Display summary
    click.echo("📊 Validation Summary:")
    click.echo(f"   Files checked: {report.total_files_checked}")
    click.echo(f"   Errors: {len(report.errors)}")
    click.echo(f"   Warnings: {len(report.warnings)}")
    click.echo(f"   Info issues: {len(report.info)}")
    click.echo(f"   Passed checks: {len(report.passed_checks)}")
    click.echo(f"   Dependency issues: {len(report.dependency_issues)}")

    # Show errors
    if report.errors:
        click.echo(f"\n❌ ERRORS ({len(report.errors)}):")
        for error in report.errors:
            click.echo(f"   • {error.component}.{error.field}: {error.message}")
            if error.suggestion and verbose:
                click.echo(f"     💡 {error.suggestion}")
            if error.file_path and verbose:
                click.echo(f"     📁 {error.file_path}")

    # Show warnings if verbose or if there are errors
    if report.warnings and (verbose or report.errors):
        click.echo(f"\n⚠️  WARNINGS ({len(report.warnings)}):")
        for warning in report.warnings[:10]:  # Limit warnings
            click.echo(f"   • {warning.component}.{warning.field}: {warning.message}")
            if warning.suggestion and verbose:
                click.echo(f"     💡 {warning.suggestion}")

    # Show dependency issues
    if report.dependency_issues:
        click.echo(f"\n📦 DEPENDENCY ISSUES ({len(report.dependency_issues)}):")
        for dep_issue in report.dependency_issues:
            click.echo(f"   • {dep_issue}")

    # Show some passed checks if verbose
    if verbose and report.passed_checks:
        click.echo("\n✅ SOME PASSED CHECKS:")
        for check in report.passed_checks[:5]:
            click.echo(f"   • {check}")
        if len(report.passed_checks) > 5:
            click.echo(f"   ... and {len(report.passed_checks) - 5} more")

    # Save output if requested
    if output:
        output_path = Path(output)
        output_path.write_text(report.model_dump_json(indent=2))
        click.echo(f"\n💾 Validation report saved to: {output_path}")

    # Overall status
    if report.is_valid:
        click.echo("\n🎉 CONFIGURATION VALID - No errors found!")
    else:
        click.echo(
            f"\n💥 CONFIGURATION INVALID - {len(report.errors)} errors need fixing"
        )
        click.echo("   Use --verbose for detailed suggestions")

    # Set exit code
    if not report.is_valid:
        sys.exit(1)


@debug.command()
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--summary", is_flag=True, help="Show trace summary")
@click.option("--errors", is_flag=True, help="Show all errors with context")
@click.option("--timeline", is_flag=True, help="Show execution timeline")
@click.option("--agent", help="Filter to specific agent (role or name)")
@click.option("--call-id", help="Show specific trace by call_id")
@click.option("--messages", is_flag=True, help="Show LLM messages for --call-id")
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def trace(path, summary, errors, timeline, agent, call_id, messages, output_json):
    """Analyze execution trace files.

    PATH is the path to a trace file (JSON or JSONL format).

    Examples:

        buttermilk debug trace /tmp/bm_test.jsonl --summary

        buttermilk debug trace /tmp/bm_test.jsonl --errors

        buttermilk debug trace /tmp/bm_test.jsonl --timeline

        buttermilk debug trace /tmp/bm_test.jsonl --agent JUDGE

        buttermilk debug trace /tmp/bm_test.jsonl --call-id abc123
    """
    import json as json_module

    try:
        tf = load_trace_file(path)
    except Exception as e:
        click.echo(f"Error loading trace file: {e}", err=True)
        sys.exit(1)

    # Default to summary if no specific option given
    if not any([summary, errors, timeline, agent, call_id]):
        summary = True

    # Handle specific trace lookup
    if call_id:
        trace_obj = get_trace(tf.traces, call_id)
        if not trace_obj:
            click.echo(f"Trace not found: {call_id}", err=True)
            sys.exit(1)

        if messages:
            msgs = get_llm_conversation(trace_obj)
            if output_json:
                click.echo(json_module.dumps(msgs, indent=2, default=str))
            else:
                for msg in msgs:
                    click.echo(f"[{msg.get('type', 'unknown')}] {str(msg.get('content', ''))[:500]}")
        else:
            io = get_inputs_outputs(trace_obj)
            if output_json:
                click.echo(json_module.dumps(io, indent=2, default=str))
            else:
                click.echo(f"Call ID: {trace_obj.call_id}")
                click.echo(f"Agent: {trace_obj.agent_info.get('component_name')} ({trace_obj.agent_info.get('role')})")
                click.echo(f"Error: {trace_obj.is_error}")
                if trace_obj.is_error and trace_obj.error:
                    click.echo(f"Error details: {trace_obj.error}")
        return

    # Filter by agent if specified
    traces = tf.traces
    if agent:
        traces = get_traces_by_agent(traces, agent)
        click.echo(f"Filtered to {len(traces)} traces for agent: {agent}")

    if summary:
        s = summarize(traces)
        if output_json:
            click.echo(s.model_dump_json(indent=2))
        else:
            click.echo(f"Total traces: {s.total_traces}")
            click.echo(f"Errors: {s.error_count}")
            click.echo(f"Agents: {', '.join(s.agents)}")
            if s.time_range:
                click.echo(f"Time range: {s.time_range[0]} to {s.time_range[1]}")
            click.echo(f"Session: {s.session_id}")

    if errors:
        errs = get_errors(traces)
        if output_json:
            click.echo(json_module.dumps([e.model_dump() for e in errs], indent=2, default=str))
        else:
            if not errs:
                click.echo("No errors found.")
            for e in errs:
                click.echo(f"\n[{e.agent_role}] {e.agent_name}")
                click.echo(f"  Error: {e.error_message}")
                click.echo(f"  Time: {e.timestamp}")
                if e.error_details:
                    click.echo(f"  Details: {e.error_details}")

    if timeline:
        events = get_timeline(traces)
        if output_json:
            click.echo(json_module.dumps([e.model_dump() for e in events], indent=2, default=str))
        else:
            for e in events:
                status = "X" if e.event_type == "error" else "+"
                click.echo(f"[{status}] {e.timestamp.strftime('%H:%M:%S')} {e.agent_role}: {e.summary}")


@debug.command()
@click.option("--host", default="localhost", help="WebSocket server host")
@click.option("--port", default=8000, type=int, help="WebSocket server port")
def websocket(host, port):
    """Interactive WebSocket debug client (no MCP required).

    This provides a standalone interactive CLI for debugging flows via WebSocket.
    It connects directly to the Buttermilk API without requiring MCP.

    Example:
        buttermilk debug websocket --host localhost --port 8000

    """
    click.echo("WebSocket client not yet implemented.")
    click.echo(f"Would connect to {host}:{port}")
    sys.exit(1)


if __name__ == "__main__":
    debug()
