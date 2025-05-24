import { json } from '@sveltejs/kit';
import { normalizeWebSocketMessage } from '$lib/utils/messageUtils';

// This endpoint is for testing purposes only
export async function POST({ request }) {
  try {
    const data = await request.json();
    const normalized = normalizeWebSocketMessage(data);
    return json({ success: true, result: normalized });
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    return json({ success: false, error: errorMessage }, { status: 500 });
  }
}
