import adapter from "@sveltejs/adapter-cloudflare"
import { vitePreprocess } from "@sveltejs/vite-plugin-svelte"
import { mdsvex } from "mdsvex"

const config = {
  preprocess: [vitePreprocess(), mdsvex()],
  kit: { adapter: adapter() },
  extensions: [".svelte", ".svx"],
}

export default config
