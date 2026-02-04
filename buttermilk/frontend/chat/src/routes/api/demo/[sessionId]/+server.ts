import { json } from "@sveltejs/kit"
import { readFile } from "node:fs/promises"
import { join } from "node:path"
import type { RequestHandler } from "./$types"

export const GET: RequestHandler = async ({ params }) => {
  const { sessionId } = params

  try {
    // Demo sessions are stored in data/demos directory
    const demosDir = join(process.cwd(), "data/demos")
    const demoFile = join(demosDir, `${sessionId}.json`)

    const fileContent = await readFile(demoFile, "utf-8")
    const demoData = JSON.parse(fileContent)

    // Transform to match the session API format expected by ChatTerminal
    const response = {
      messages: demoData.messages || [],
      session_metadata: {
        flow_status: "demo",
        is_stale: false,
        is_resumable: false,
        message_count: demoData.messages ? demoData.messages.length : 0,
        parameters: demoData.parameters || {},
        demo_metadata: demoData.demo_metadata || {},
      },
    }

    return json(response)
  } catch (fileError) {
    // Log file errors concisely
    if (fileError instanceof Error && "code" in fileError && fileError.code === "ENOENT") {
      console.log(`Demo file not found: ${sessionId}.json`)
    } else {
      console.error("Error reading demo file:", fileError)
    }
    return json(
      { error: "Demo not found" },
      { status: 404 },
    )
  }
}
