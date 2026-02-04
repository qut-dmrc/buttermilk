import { json } from "@sveltejs/kit"
import { readdir, readFile } from "node:fs/promises"
import { join } from "node:path"

// Demo metadata interface
interface DemoMetadata {
  session_id: string
  title: string
  description: string
  duration: string
  agents: string[]
}

export async function GET() {
  try {
    // Demo sessions are stored in data/demos directory
    const demosDir = join(process.cwd(), "data/demos")

    // Read all JSON files from the demos directory
    let files: string[] = []
    try {
      files = await readdir(demosDir)
    } catch (err) {
      // Directory doesn't exist yet - return empty list
      console.log("Demos directory not found, returning empty list")
      return json({ demos: [], total: 0 })
    }

    const demoFiles = files.filter((file: string) => file.endsWith(".json"))

    const demos: DemoMetadata[] = []

    // Read each demo file to extract metadata
    for (const file of demoFiles) {
      try {
        const filePath = join(demosDir, file)
        const fileContent = await readFile(filePath, "utf-8")
        const demoData = JSON.parse(fileContent)

        // Extract metadata from demo file
        const demoInfo: DemoMetadata = {
          session_id: demoData.session_id || file.replace(".json", ""),
          title: demoData.demo_metadata?.title || demoData.session_id || "Untitled Demo",
          description: demoData.demo_metadata?.description || "Multi-agent groupchat session",
          duration: demoData.demo_metadata?.duration || calculateDuration(demoData.messages || []),
          agents: extractAgents(demoData.messages || []),
        }

        demos.push(demoInfo)
      } catch (fileError) {
        console.warn(`Failed to read demo file ${file}:`, fileError)
        // Continue with other files
      }
    }

    // Sort demos by title
    demos.sort((a, b) => a.title.localeCompare(b.title))

    return json({
      demos,
      total: demos.length,
    })
  } catch (error) {
    console.error("Error reading demos directory:", error)
    return json(
      { error: "Failed to read demos directory" },
      { status: 500 },
    )
  }
}

// Calculate duration from message timestamps
function calculateDuration(messages: any[]): string {
  if (messages.length < 2) return "< 1 min"

  try {
    const first = new Date(messages[0].timestamp)
    const last = new Date(messages[messages.length - 1].timestamp)
    const diffMs = last.getTime() - first.getTime()
    const diffMins = Math.round(diffMs / 60000)

    if (diffMins < 1) return "< 1 min"
    if (diffMins === 1) return "1 min"
    return `${diffMins} mins`
  } catch (e) {
    return "unknown"
  }
}

// Extract unique agent names from messages
function extractAgents(messages: any[]): string[] {
  const agents = new Set<string>()

  for (const msg of messages) {
    if (msg.agent_info?.role) {
      agents.add(msg.agent_info.role)
    }
  }

  return Array.from(agents)
}
