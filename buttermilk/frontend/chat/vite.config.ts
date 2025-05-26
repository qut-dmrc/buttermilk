import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		// Add proper CORS handling
		cors: true,
		allowedHosts: ['localhost', 'nichome.stoat-musical.ts.net', 'services.stoat-musical.ts.net', 'nichome', 'wsl','services'],
		// Configure proxy for backend websocket connections
		proxy: {
			'/ws': {
				target: 'ws://localhost:8000',
				ws: true,
				changeOrigin: true
			},
			// Proxy API requests that should go to backend
			'/api/backend': {
				target: 'http://localhost:8000',
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/api\/backend/, '/api')
			}
		},
		// Ensure we can resolve modules properly
		fs: {
			strict: false
		}
	},
// Optimize dependencies to avoid dynamic import issues
optimizeDeps: {
	include: [
		'@sveltejs/kit', 
		'bootstrap', 
		'bootstrap/dist/js/bootstrap.bundle.min.js'
	],
	exclude: []
},
// Help with module resolution
resolve: {
	dedupe: ['svelte', '@sveltejs/kit'],
	preserveSymlinks: true
}
});
