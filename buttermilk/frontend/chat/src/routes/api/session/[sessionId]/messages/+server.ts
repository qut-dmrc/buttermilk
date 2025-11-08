import { env } from "$env/dynamic/private"
import { checkBackendHealth, getSessionsDir, logBackendStatus } from "$lib/utils/backendUtils"
import { json } from "@sveltejs/kit"
import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ params, url, fetch }) => {
  const { sessionId } = params

  // Get backend URL from environment
  const backendUrl = env.BACKEND_API_URL || "http://localhost:8000"

  // Check if backend is available using centralized health check
  const isBackendHealthy = await checkBackendHealth(true, fetch)
  logBackendStatus("API messages endpoint", isBackendHealthy)

  if (isBackendHealthy) {
    try {
      // Backend is available, try to get the actual messages
      const backendResponse = await fetch(`${backendUrl}/api/session/${sessionId}/messages`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
        signal: AbortSignal.timeout(3000), // 3 second timeout
      })

      if (backendResponse.ok) {
        const data = await backendResponse.json()
        return json(data)
      }

      console.log(`Backend unavailable (${backendResponse.status}), falling back to file system`)
    } catch (error) {
      // Log backend connection errors concisely - these are expected in development
      if (error instanceof Error && error.message.includes("fetch failed")) {
        console.log("Backend unavailable, falling back to file system")
      } else {
        console.log("Backend unavailable, falling back to file system:", error)
      }
    }
  }

  // Fallback: Read directly from file system (demo mode)
  try {
    // Get configured sessions directory from environment variable
    const configuredSessionsDir = getSessionsDir()
    // Sessions directory is relative to project root
    const sessionsDir = join(process.cwd(), configuredSessionsDir)
    const sessionFile = join(sessionsDir, `${sessionId}.json`)

    const fileContent = await readFile(sessionFile, "utf-8")
    const sessionData = JSON.parse(fileContent)

    // Transform to match backend API format
    const response = {
      messages: sessionData.messages || [],
      session_metadata: {
        flow_status: sessionData.flow_status || "completed",
        is_stale: false,
        is_resumable: false, // Always false for demo mode
        message_count: sessionData.messages ? sessionData.messages.length : 0,
        parameters: sessionData.parameters || {},
      },
    }

    return json(response)
  } catch (fileError) {
    // Log file errors concisely - ENOENT is expected for non-existent sessions
    if (fileError instanceof Error && "code" in fileError && fileError.code === "ENOENT") {
      console.log(`Session file not found: ${sessionId}.json`)
    } else {
      console.error("Error reading session file:", fileError)
    }
    return json(
      { error: "Session not found" },
      { status: 404 },
    )
  }
}
