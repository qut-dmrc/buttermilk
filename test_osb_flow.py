#!/usr/bin/env python3
"""Standalone OSB flow tester that bypasses import issues."""

import time
import requests
import json
import subprocess
import sys
from pathlib import Path


def test_api_health(base_url="http://localhost:8000"):
    """Test if API is healthy and reachable."""
    try:
        response = requests.get(f"{base_url}/monitoring/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health: {data.get('overall_status', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_flows_available(base_url="http://localhost:8000"):
    """Test if flows are available."""
    try:
        response = requests.get(f"{base_url}/api/flows", timeout=10)
        if response.status_code == 200:
            data = response.json()
            flows = data.get("flow_choices", [])
            print(f"✅ Available flows: {flows}")
            return "osb" in flows
        else:
            print(f"❌ Flows check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Flows check failed: {e}")
        return False


def test_osb_vector_query(query, base_url="http://localhost:8000"):
    """Test OSB vector query via MCP."""
    endpoint = f"{base_url}/mcp/osb/vector-query"
    payload = {
        "query": query,
        "max_results": 5,
        "search_strategy": "semantic"
    }
    
    try:
        print(f"🔍 Testing OSB vector query: '{query}'")
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=30)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Vector query success ({response_time:.2f}s)")
            
            if "result" in data and "search_results" in data["result"]:
                results = data["result"]["search_results"]
                print(f"📊 Found {len(results)} search results:")
                for i, result in enumerate(results[:3]):
                    title = result.get("title", "No title")
                    score = result.get("score", 0)
                    print(f"   {i+1}. {title} (score: {score:.3f})")
            
            return True
        else:
            print(f"❌ Vector query failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ Vector query failed: {e}")
        return False


def test_osb_agent(agent_role, query, base_url="http://localhost:8000"):
    """Test individual OSB agent."""
    endpoint = f"{base_url}/mcp/osb/agent-invoke"
    payload = {
        "agent_role": agent_role,
        "query": query,
        "include_context": True
    }
    
    try:
        print(f"🤖 Testing {agent_role} agent: '{query[:50]}...'")
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=60)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ {agent_role} success ({response_time:.2f}s)")
            
            if "result" in data and "content" in data["result"]:
                content = data["result"]["content"]
                print(f"📝 Response: {content[:100]}...")
            
            return True
        else:
            print(f"❌ {agent_role} failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}...")
            return False
            
    except Exception as e:
        print(f"❌ {agent_role} failed: {e}")
        return False


def test_osb_flow_start(query, base_url="http://localhost:8000"):
    """Test starting OSB flow."""
    endpoint = f"{base_url}/mcp/flows/start"
    payload = {
        "flow_name": "osb",
        "initial_message": query,
        "session_id": f"debug-{int(time.time())}"
    }
    
    try:
        print(f"🚀 Testing OSB flow start: '{query}'")
        start_time = time.time()
        response = requests.post(endpoint, json=payload, timeout=90)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Flow start success ({response_time:.2f}s)")
            
            if "result" in data:
                result = data["result"]
                if isinstance(result, dict) and "content" in result:
                    content = result["content"]
                    print(f"📝 Flow response: {content[:150]}...")
                elif isinstance(result, str):
                    print(f"📝 Flow response: {result[:150]}...")
            
            return True
        else:
            print(f"❌ Flow start failed: HTTP {response.status_code}")
            print(f"   Response: {response.text[:300]}...")
            return False
            
    except Exception as e:
        print(f"❌ Flow start failed: {e}")
        return False


def main():
    """Run comprehensive OSB flow test."""
    print("🧪 Starting OSB Flow End-to-End Test")
    print("="*60)
    
    base_url = "http://localhost:8000"
    test_query = "does hate speech have to be explicit to be prohibited?"
    
    # Test API health
    print("\\n🏥 Testing API Health...")
    if not test_api_health(base_url):
        print("❌ API health check failed - is daemon running?")
        return False
    
    # Test flows availability
    print("\\n📊 Testing Flow Availability...")
    if not test_flows_available(base_url):
        print("❌ OSB flow not available")
        return False
    
    # Test vector query
    print("\\n🔍 Testing Vector Store...")
    vector_success = test_osb_vector_query(test_query, base_url)
    
    # Test individual agents
    print("\\n🤖 Testing Individual Agents...")
    agent_results = {}
    for agent in ["researcher", "policy_analyst", "fact_checker", "explorer"]:
        agent_results[agent] = test_osb_agent(agent, test_query, base_url)
    
    # Test flow start
    print("\\n🚀 Testing Flow Orchestration...")
    flow_success = test_osb_flow_start(test_query, base_url)
    
    # Summary
    print("\\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"🔍 Vector Store: {'✅' if vector_success else '❌'}")
    
    successful_agents = sum(1 for success in agent_results.values() if success)
    print(f"🤖 Agent Tests: {successful_agents}/4 passed")
    for agent, success in agent_results.items():
        print(f"   • {agent}: {'✅' if success else '❌'}")
    
    print(f"🚀 Flow Start: {'✅' if flow_success else '❌'}")
    
    overall_success = vector_success and all(agent_results.values()) and flow_success
    print(f"\\n🎯 OVERALL: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)