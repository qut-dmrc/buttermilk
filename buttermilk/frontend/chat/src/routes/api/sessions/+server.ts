import { json } from '@sveltejs/kit';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

export async function GET() {
	try {
		// Path to the sessions directory
		const sessionsDir = join(process.cwd(), '../../../data/sessions');
		
		// Read all JSON files from the sessions directory
		const files = await readdir(sessionsDir);
		const sessionFiles = files.filter((file: string) => file.endsWith('.json'));
		
		const sessions = [];
		
		// Read each session file to extract session info and parameters
		for (const file of sessionFiles) {
			try {
				const filePath = join(sessionsDir, file);
				const fileContent = await readFile(filePath, 'utf-8');
				const sessionData = JSON.parse(fileContent);
				
				// Extract session metadata and parameters
				const sessionInfo = {
					session_id: sessionData.session_id || file.replace('.json', ''),
					created_at: sessionData.created_at,
					last_updated: sessionData.last_updated,
					flow_status: sessionData.flow_status,
					parameters: sessionData.parameters || {},
					message_count: sessionData.messages ? sessionData.messages.length : 0
				};
				
				sessions.push(sessionInfo);
			} catch (fileError) {
				console.warn(`Failed to read session file ${file}:`, fileError);
				// Continue with other files
			}
		}
		
		// Sort sessions by last_updated, most recent first
		sessions.sort((a, b) => {
			const dateA = new Date(a.last_updated || a.created_at);
			const dateB = new Date(b.last_updated || b.created_at);
			return dateB.getTime() - dateA.getTime();
		});
		
		return json({
			sessions,
			total: sessions.length
		});
		
	} catch (error) {
		console.error('Error reading sessions directory:', error);
		return json(
			{ error: 'Failed to read sessions directory' },
			{ status: 500 }
		);
	}
}