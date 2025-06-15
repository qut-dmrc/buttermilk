#!/usr/bin/env python3
"""Simple test script for MCP endpoints."""

import asyncio
import json
import aiohttp
import sys

async def test_mcp_endpoints():
    """Test the MCP endpoints with simple requests."""
    
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        
        # Test 1: List available tools
        print("🔧 Testing tool listing...")
        try:
            async with session.get(f"{base_url}/mcp/tools") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Tools listed: {len(data['result']['tools'])} tools available")
                    for tool in data['result']['tools']:
                        print(f"   - {tool['name']}: {tool['description']}")
                else:
                    print(f"❌ Tool listing failed: {resp.status}")
        except Exception as e:
            print(f"❌ Tool listing error: {e}")
        
        print()
        
        # Test 2: Judge tool
        print("⚖️  Testing judge tool...")
        judge_data = {
            "text": "This is a test message about climate change policy.",
            "criteria": "toxicity", 
            "model": "gpt4o",
            "flow": "tox"
        }
        
        try:
            async with session.post(
                f"{base_url}/mcp/tools/judge",
                json=judge_data,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['success']:
                        print(f"✅ Judge tool succeeded (trace: {data['trace_id']})")
                        print(f"   Execution time: {data['execution_time_ms']:.2f}ms")
                        print(f"   Result type: {type(data['result'])}")
                    else:
                        print(f"❌ Judge tool failed: {data['error']}")
                else:
                    print(f"❌ Judge tool HTTP error: {resp.status}")
                    text = await resp.text()
                    print(f"   Response: {text}")
        except Exception as e:
            print(f"❌ Judge tool error: {e}")
        
        print()
        
        # Test 3: Synthesize tool
        print("🔄 Testing synthesize tool...")
        synth_data = {
            "text": "The policy has mixed environmental impacts.",
            "criteria": "climate_advocacy",
            "model": "gpt4o", 
            "flow": "tox"
        }
        
        try:
            async with session.post(
                f"{base_url}/mcp/tools/synthesize",
                json=synth_data,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['success']:
                        print(f"✅ Synthesize tool succeeded (trace: {data['trace_id']})")
                        print(f"   Execution time: {data['execution_time_ms']:.2f}ms")
                    else:
                        print(f"❌ Synthesize tool failed: {data['error']}")
                else:
                    print(f"❌ Synthesize tool HTTP error: {resp.status}")
        except Exception as e:
            print(f"❌ Synthesize tool error: {e}")
        
        print()
        
        # Test 4: Differences tool
        print("🔍 Testing differences tool...")
        diff_data = {
            "text1": "Climate change is a serious environmental issue.",
            "text2": "Global warming poses significant ecological challenges.",
            "criteria": "semantic_similarity",
            "model": "gpt4o",
            "flow": "tox"
        }
        
        try:
            async with session.post(
                f"{base_url}/mcp/tools/find_differences",
                json=diff_data,
                headers={'Content-Type': 'application/json'}
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data['success']:
                        print(f"✅ Differences tool succeeded (trace: {data['trace_id']})")
                        print(f"   Execution time: {data['execution_time_ms']:.2f}ms")
                    else:
                        print(f"❌ Differences tool failed: {data['error']}")
                else:
                    print(f"❌ Differences tool HTTP error: {resp.status}")
        except Exception as e:
            print(f"❌ Differences tool error: {e}")
        
        print("\n🏁 MCP endpoint testing complete!")

if __name__ == "__main__":
    print("🧪 Testing MCP endpoints...")
    print("📍 Make sure Buttermilk API is running on localhost:8000")
    print()
    
    try:
        asyncio.run(test_mcp_endpoints())
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed: {e}")
        sys.exit(1)