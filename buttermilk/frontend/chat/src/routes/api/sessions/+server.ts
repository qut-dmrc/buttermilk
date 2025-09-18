import { json } from '@sveltejs/kit';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { getSessionsDir, checkBackendHealth, logBackendStatus } from '$lib/utils/backendUtils';
import { env } from '$env/dynamic/private';

export async function GET({ fetch }) {
	// Get backend URL from environment
	const backendUrl = env.BACKEND_API_URL || 'http://localhost:8000';
	
	// Check if backend is available
	const isBackendHealthy = await checkBackendHealth(true, fetch);
	logBackendStatus('API sessions endpoint', isBackendHealthy);
	
	// Try to get sessions from backend first
	if (isBackendHealthy) {
		try {
			const backendResponse = await fetch(`${backendUrl}/api/sessions`, {
				method: 'GET',
				headers: {
					'Content-Type': 'application/json',
				},
				signal: AbortSignal.timeout(3000) // 3 second timeout
			});
			
			if (backendResponse.ok) {
				const data = await backendResponse.json();
				return json(data);
			}
			
			console.log(`Backend unavailable (${backendResponse.status}), falling back to file system`);
		} catch (error) {
			console.log('Backend unavailable, falling back to file system:', error);
		}
	}
	
	// Fallback: Read directly from file system (demo mode)
	try {
		// Get configured sessions directory from environment variable
		const configuredSessionsDir = getSessionsDir();
		// Sessions directory is relative to project root
		const sessionsDir = join(process.cwd(), configuredSessionsDir);
		
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