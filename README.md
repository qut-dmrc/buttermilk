# buttermilk: opinionated data tools for HASS scholars

**AI and data tools for HASS researchers, putting culture first.**

Developed by and for @QUT-DMRC scholars, this repo aims to provide standard **flows** that help scholars make their own **pipelines** to collect data and use machine learning, generative AI, and computational techniques as part of rich analysis and experimentation that is driven by theory and deep understanding of cultural context. We try to:

* Provide a set of research-backed analysis tools that help scholars bring cultural expertise to computational methods.
* Help HASS scholars with easy, well-documented, and proven tools for data collection and analysis.
* Make ScholarOps™ easier with opinionated defaults that take care of logging and archiving in standard formats.
* Create a space for collaboration, experimentation, and evaluation of computational methods for HASS scholars.

```md
Q: Why 'buttermilk'??
A: It's cultured and flows...
```

```md
Q: What's MLOps?
A: A general term for standardised approaches to machine learning workflows that 
helps you organize your project, collaborate, iteratively improve your analysis 
and track versioned changes, monitor onging performance, reproduce experiments, 
and verify and compare results. 
```

The "pipeline" we are building is documented and versioned. We're aiming to make it easy for HASS scholars to use AI tools in a way that is understandable, traceable, and reproducible.

## Flows and jobs
We think of predictions not as individual runs of links on chains,
but instead as inherently stochastic insights into problems that
are not always well-defined, by agents with varying capabilities.

Accordingly, Buttermilk's Flows provide a methodology for repeatedly
querying agents about a dataset from multiple perspectives over
multiple time periods. We sample from prior answers to inform
current predictions, adjusting for randomness and specificity.

* A **Flow** is a predefined pipeline consisting of one or more steps
to process data and pass it on.

* A **Job** is the basic unit of work containing all information required
to run one record through one step of processing. Jobs are designed to 
include all the information necessary to run and trace a single
atomic step of work. They can be run basically anywhere, and saving a job
to a database or to disk provides all information required to understand
and reproduce a single result.

* A **Dataset** contains many **Record** objects, each with a unique 
`record_id`. Records are immutable and may include text and binary data 
as well as adequate metadata.


## Usage

So far, we have a few standard pieces that set sensible defaults to make it easy for HASS scholars to use, store, assess, compare, and reproduce complicated AI workflows in their research.

Working right now:

* Multimodal support for current-generation foundation models (Gemini, Claude, Llama, GPT) and plug-in support for basically any other analysis tool API.
* A prompt templating system that allows you to evaluate, improve, and reuse components. These will ideally become a collection of versioned prompts that have been carefully evaluated, and come with test cases to monitor if they start to break.
* Standard cloud logging, flexible data storage, secure project and individual credentials (Azure KeyVault or Google Secrets), built-in database storage (BigQuery), tracing (Promptflow or Langchain).
* An API and CLI that integrates each modular component and orchestrates complex workflows, including parallel runs and multi-step workflows.
* Use the same code to run locally, on a remote GPU, or in Azure/Google Compute [and AWS llambda, I guess, but not yet].

Next:

* Some tutorial workbooks showing a full walkthrough of an entire pipeline research run.
* A pub/sub distributed queue system and an interface to add batch runs
* A web interface and notebooks with good examples of how to assess, track, and compare performance on individual examples and aggregated runs.

## Very early stages!

We would love your help! Contact [nic](mailto:n.suzor@qut.edu.au) to discuss what you'd like to see, help us plan, or how to contribute!

## Installation

Create a new environment and install using uv:

```shell
pip install uv
uv install
```

Authenticate to cloud providers, where your relevant secrets are stored.

```shell
GOOGLE_CLOUD_PROJECT=<project>
gcloud auth login --update-adc --enable-gdrive-access --project ${GOOGLE_CLOUD_PROJECT} --billing-project ${GOOGLE_CLOUD_PROJECT}
gcloud auth application-default set-quota-project ${GOOGLE_CLOUD_PROJECT}
gcloud config set project ${GOOGLE_CLOUD_PROJECT}
```

Configurations are stored as YAML files in `conf/`. You can select options at runtime using [hydra](https://hydra.cc).
