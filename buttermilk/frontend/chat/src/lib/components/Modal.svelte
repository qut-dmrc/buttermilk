<script lang="ts">
	export let show = false;
	export let title = '';
</script>

{#if show}
	<div class="modal fade show" style="display: block;" tabindex="-1">
		<div class="modal-dialog modal-lg">
			<div class="modal-content">
				<div class="modal-header">
					<h5 class="modal-title">{title}</h5>
					<button type="button" class="btn-close" on:click={() => (show = false)}></button>
				</div>
				<div class="modal-body">
					<slot name="body"></slot>
				</div>
				<div class="modal-footer">
					<slot name="footer">
						<button type="button" class="btn btn-secondary" on:click={() => (show = false)}
							>Close</button
						>
					</slot>
				</div>
			</div>
		</div>
	</div>
	<div class="modal-backdrop fade show"></div>
{/if}
