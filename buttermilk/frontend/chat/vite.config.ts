import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
	const env = loadEnv(mode, process.cwd(), '');
	const backendApiUrl = env.BACKEND_API_URL || 'http://localhost:8000';
	
	return {
		plugins: [sveltekit()],
		server: {
			// Add proper CORS handling
			cors: true,
			allowedHosts: ['localhost', 'nichome.stoat-musical.ts.net', 'services.stoat-musical.ts.net', 'nichome', 'wsl','services', 'nicdev', 'nicdev.stoat-musical.ts.net', "127.0.0.1"],
			// Configure proxy for backend websocket connections
			proxy: {
				'/ws': {
					target: backendApiUrl.replace('http', 'ws'),
					ws: true,
					changeOrigin: true
				},
			// Proxy API requests that should go to backend
			'/api/backend': {
				target: backendApiUrl,
				changeOrigin: true,
				rewrite: (path) => path.replace(/^\/api\/backend/, '/api')
			},
			// Proxy all other API requests to backend (except SvelteKit API routes)
			'^/api/(flows|flowinfo|records)': {
				target: backendApiUrl,
				changeOrigin: true
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
	};
});
