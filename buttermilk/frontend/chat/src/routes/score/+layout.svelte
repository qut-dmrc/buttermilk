<!-- Score layout is now simplified - sidebar logic moved to UnifiedSidebar -->

<div class="score-layout">
	<!-- Content will be in the main layout's content area, sidebar is unified -->
	<div class="score-content">
		<slot></slot>
	</div>
</div>

<style>
	.score-layout {
		
	}

	.score-content {
		padding: 1rem;
	}
</style>
