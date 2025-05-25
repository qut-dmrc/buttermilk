<!-- Content based on demo/templates/index.html -->
<script lang="ts">
	// Any specific script needed for this page can go here
	const title = 'ChatGPT vs the Oversight Board'; // Hardcoded title
</script>

<div class="row align-items-top">
	<div class="col-sm-12">
		<h1>{title}</h1>
		<p class="text-muted text-lowercase">
			A project of the <a href="https://www.admscentre.org.au/">ARC Centre of Excellence for Automated Decision-Making and Society</a>. Developed by <a href="#">Nicolas Suzor</a> and <a href="https://lucinda-nelson.com/">Lucinda Nelson</a>.
		</p>
	</div>
</div>

<div class="row align-items-top">
	<div class="col-sm-12">
		<p class="text-light">
			This experimental tool uses large multimodal machine learning models to understand implicit and
			covert hateful speech.
		</p>
		<div class="alert alert-primary" role="alert">
			<p>
				OK, this isn't really ChatGPT <em>versus</em> the Oversight Board. We're not trying to
				recreate the important deliberation and norm-setting role of Board. Instead, we're trying to
				learn from the Board's decisions and see how we can improve the <em>application</em> of
				standards to new contexts.
			</p>
		</div>
		<p class="text-light">
			We use the <a href="https://oversightboard.com">Oversight Board</a> decisions on hate speech
			to calibrate against some of the toughest challenges in content moderation. On this initial
			small sample, we have seen surprisingly good results -- we have been able to approximate
			Oversight Board results in 13 of the 15 hate speech decisions released up to late 2023.
		</p>
	</div>
</div>

<pre>
|:----------------------------------|------------:|------------:|--------:|----------:|
|     accuracy - our prompts        | Claude 3.5  |   Gemini    | GPT 4o  | Llama 3.1 |
|                                   |   Sonnet    |  1.5 pro    |         |   405bn   |
|:----------------------------------|------------:|------------:|--------:|----------:|
| Drag Queens (zero-shot CoT)       |        0.96 |        0.84 |    0.91 |      0.98 |
| Drag Queens (mix of 8 experts)    |        0.92 |        0.87 |    0.91 |      0.92 |
| Oversight Board (zero-shot CoT)   |        0.88 |        0.72 |    0.78 |      0.75 |
| Oversight Board (mix of 8 experts)|        0.78 |        0.80 |    0.82 |      0.79 |
|:----------------------------------|------------:|------------:|--------:|----------:|
</pre>

<div class="col-sm-12">
	<h1>(anti-)tone police experiments</h1>
	<p class="text-light">
		Most attempts to build machine learning models that can detect 'toxicity', hate, and abuse
		don't work very well. Usually, they end up detecting strongly-worded statements -- wrongly
		classifying counterspeech, in-group jokes, and reappropriated slurs as 'toxic', while missing
		whole ranges of hateful messages that are politely expressed, use coded language, or are
		disguised as humour. Predictably, these systems tend to disproportionately impact already
		marginalised groups.
	</p>
	<p class="text-light">
		This is bad news for societies that are increasingly reliant on machine classification and
		content generation. Safety standards and AI 'guardrails' are urgently needed, but so far we
		have few examples of tools that detect implicit or covert hate without over-policing the
		speech of marginalised groups.
	</p>
	<p class="text-light">
		We are working to <strong>build guardrails that do not tone police</strong>. We are chaining
		together large multimodal models with a range of information retrieval and classification
		techniques to better take context into account. So far, our focus is on gender and sex --
		areas where we have extensive subject-matter expertise. Our goal is to develop evaluation
		criteria based on media standards that have been developed by community groups and experts. We
		hope to make these criteria available in the form of tools and test sets that can be used by
		developers to evaluate their own systems.
	</p>
</div>

<div class="col-sm-12">
	<p class="text-light">
		In the next step of this work-in-progress, we are exploring a variety of techniques to
		iteratively refine the quality of the analysis -- focusing particularly on the difficult
		challenge of understanding meaning from ambiguous content that typically requires knowledge of
		context and human assessment of the poster's intent and the audience's likely interpretation.
	</p>
	<p class="text-light">
		Here you can find some examples of our experiments so far and see where we're going. You might
		also be able to test our tools on your own examples and see how they perform. We're still in
		the early stages of this work, so please be patient with us!
	</p>
</div>

<style>
	/* Add any specific styles for the index page if needed */
	pre {
		/* Ensure preformatted text wraps or scrolls if needed */
		white-space: pre-wrap; /* Or pre-scroll */
		word-wrap: break-word;
		background-color: rgba(0,0,0,0.1); /* Slight background for visibility */
		padding: 0.5rem;
		border-radius: var(--bs-border-radius);
	}
</style>
