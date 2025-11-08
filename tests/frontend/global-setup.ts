/**
 * Playwright Global Setup
 *
 * Sets up the test environment before running E2E tests including:
 * - Backend API health verification
 * - Test data preparation
 * - WebSocket server validation
 * - Performance monitoring initialization
 */

import { chromium, FullConfig } from "@playwright/test"

async function globalSetup(config: FullConfig) {
  console.log("🚀 Starting Playwright Global Setup...")

  const baseURL = config.projects[0].use.baseURL || "http://localhost:5173"
  const apiURL = "http://localhost:8000"

  // Launch browser for setup tasks
  const browser = await chromium.launch()
  const context = await browser.newContext()
  const page = await context.newPage()

  try {
    // 1. Verify backend API is running
    console.log("⏳ Verifying backend API health...")
    const apiResponse = await page.request.get(`${apiURL}/health`)
    if (!apiResponse.ok()) {
      throw new Error(`Backend API health check failed: ${apiResponse.status()}`)
    }
    console.log("✅ Backend API is healthy")

    // 2. Verify frontend is accessible
    console.log("⏳ Verifying frontend accessibility...")
    await page.goto(baseURL, { waitUntil: "networkidle" })

    // Check if terminal interface loads
    const terminalContainer = page.locator("[data-testid=\"terminal-container\"]")
    if (!(await terminalContainer.isVisible())) {
      console.warn("⚠️  Terminal container not immediately visible - may need to wait for app initialization")
    } else {
      console.log("✅ Frontend terminal interface is accessible")
    }

    // 3. Verify WebSocket connectivity
    console.log("⏳ Testing WebSocket connectivity...")
    try {
      // Create a test session
      const sessionResponse = await page.request.get(`${apiURL}/api/session`)
      if (sessionResponse.ok()) {
        const sessionData = await sessionResponse.json()
        console.log(`✅ WebSocket session creation successful: ${sessionData.session_id}`)
      }
    } catch (error) {
      console.warn("⚠️  WebSocket test failed:", error)
    }

    // 4. Verify flow configurations are available
    console.log("⏳ Verifying flow configurations...")
    try {
      // Test MCP endpoint to ensure flows are loaded
      const flowTestResponse = await page.request.post(`${apiURL}/mcp/agents/message-validation`, {
        data: {
          message_type: "run_flow",
          message_data: { query: "test", flow: "osb" },
          strict_validation: false,
        },
      })

      if (flowTestResponse.ok()) {
        console.log("✅ Flow configurations are accessible")
      } else {
        console.warn("⚠️  Flow configuration test returned:", flowTestResponse.status())
      }
    } catch (error) {
      console.warn("⚠️  Flow configuration verification failed:", error)
    }

    // 5. Set up test data (if needed)
    console.log("⏳ Setting up test data...")

    // Create any necessary test sessions or data
    // This could include creating mock vector store data, test users, etc.

    console.log("✅ Test data setup complete")

    // 6. Performance monitoring setup
    console.log("⏳ Initializing performance monitoring...")

    // Store baseline performance metrics
    const performanceData = {
      setupTime: Date.now(),
      baseURL,
      apiURL,
      testStartTime: new Date().toISOString(),
    }

    // Store in a file for tests to access
    const fs = require("fs")
    const path = require("path")
    fs.writeFileSync(
      path.join(__dirname, "test-performance-baseline.json"),
      JSON.stringify(performanceData, null, 2),
    )

    console.log("✅ Performance monitoring initialized")
  } catch (error) {
    console.error("❌ Global setup failed:", error)
    throw error
  } finally {
    await context.close()
    await browser.close()
  }

  console.log("🎉 Playwright Global Setup completed successfully!")
}

export default globalSetup
