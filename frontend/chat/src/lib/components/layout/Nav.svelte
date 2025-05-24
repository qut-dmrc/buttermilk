<script lang="ts">
	import { page } from '$app/stores';
</script>

<nav class="navbar navbar-expand-lg navbar-light bg-light mb-3">
	<!-- Navbar Toggler (for smaller screens) -->
	<button
		class="navbar-toggler"
		type="button"
		data-bs-toggle="collapse"
		data-bs-target="#navbarNav"
		aria-controls="navbarNav"
		aria-expanded="false"
		aria-label="Toggle navigation"
	>
		<span class="navbar-toggler-icon" />
	</button>

	<!-- Navbar Links -->
	<div class="collapse navbar-collapse" id="navbarNav">
		<ul class="navbar-nav me-auto mb-2 mb-lg-0">
			<li class="nav-item glitch">
				<a
					class="nav-link"
					class:active={$page.url.pathname === '/'}
					aria-current={$page.url.pathname === '/' ? 'page' : undefined}
					href="/"
				>
					ChatGPT vs the Oversight Board
				</a>
			</li>
			<li class="nav-item">
				<a
					class="nav-link"
					class:active={$page.url.pathname === '/results'}
					aria-current={$page.url.pathname === '/results' ? 'page' : undefined}
					href="/results"
				>
					Results
				</a>
			</li>
			<li class="nav-item">
				<a
					class="nav-link"
					class:active={$page.url.pathname === '/terminal'}
					aria-current={$page.url.pathname === '/terminal' ? 'page' : undefined}
					href="/terminal"
				>
					Terminal
				</a>
			</li>
			<li class="nav-item">
				<a
					class="nav-link"
					class:active={$page.url.pathname === '/faq'}
					aria-current={$page.url.pathname === '/faq' ? 'page' : undefined}
					href="/faq"
				>
					Contact
				</a>
			</li>
		</ul>
	</div>
</nav>
