### Buttermilk & Automod: System Overview

This document provides a high-level overview of the Buttermilk/Automod ecosystem.

#### The Big Picture: What is Buttermilk?

Buttermilk is the foundational engine for our research. Think of it as an "MLOps for HASS scholars" platform.

*   **Core Mission:** It’s a Python-based framework designed to make computational research methods accessible and rigorous for Humanities, Arts, and Social Sciences (HASS) scholars. It prioritizes usability and reproducibility over raw performance.
*   **How it Works:** The system is built around **Agents** (specialized Python objects that do one thing well, like calling an LLM) that are connected into **Flows** (research pipelines). All of the settings for these flows are defined in simple **YAML files**, which means our experiments are easy to reproduce.

#### The `automod.cc` Interface

The `automod.cc` website is the primary interface for running and inspecting these research flows.

*   **Core Concept:** It's designed to look and feel like a retro **IRC-style group chat**. The researcher acts as the supervisor of the chat, and the AI agents are the participants. This makes it easy to see how the different agents are interacting to produce the final result.

The frontend web interface (buttermilk/frontend/chat) and the backend api server are deployed from a Docker image (deploy/Dockerfile) that is rebuilt with github CI/CD.

#### The Core Evaluation Pattern: `JUDGE` -> `SYNTH`

A primary goal of this research is to evaluate how well different AI models can apply a set of complex, human-readable guidelines to a piece of text. Our core workflow for this is a two-step "groupchat" pattern:

1.  **The `JUDGE` Round:**
    *   First, we show the same piece of text (e.g., a news article) to several different `JUDGE` agents simultaneously.
    *   Each `JUDGE` is powered by a different Large Language Model (e.g., one uses Google's Gemini, another uses Anthropic's Claude 3 Opus, another uses OpenAI's GPT-4).
    *   Each `JUDGE` is a "zero-shot" agent, meaning it gets no prior examples. It is simply given the text and the guidelines and asked to make a decision.

2.  **The `SYNTH` Round:**
    *   Next, a `SYNTH` agent is shown the original text and guidelines, *plus all of the answers from the `JUDGE` round*.
    *   The `SYNTH` agent's job is to act as a meta-analyst. It reviews the arguments made by the different `JUDGE` agents and synthesizes them to produce a final, more robust answer.

This pattern allows us to not only test the individual performance of different models but also to see if a "wisdom of the crowd" approach can lead to better results.

#### The Scoring System

To evaluate the performance of this system, we use a multi-layered process:

*   **The Golden Set:** For each project (like TJA), we have a "golden set" of examples that have been hand-coded with detailed, criterion-referenced assessments. This is our ground truth.
*   **The `SCORER` Agents:** Every `JUDGE` and `SYNTH` answer is evaluated by a team of `SCORER` agents. These `SCORER`s are also powered by different LLMs. They are given the agent's answer and the "golden set" answer and are asked to score the agent's reasoning on several qualitative points.
*   **Stochastic Testing:** Because LLMs can be random, we run every evaluation at least 10 times to ensure our results are stable and not just a fluke.

#### The Data Workflow

*   **Raw Data:** The datasets for each project (e.g., the 28 TJA articles) are stored in **Google Cloud Storage (GCS)**.
*   **Results:** All results are stored in **BigQuery** as `AgentTrace` records and are also sent to **Weights and Biases** for experiment tracking.
*   **Analysis:** We use SQL to query the BigQuery database for analysis. Currently, we are using **Google Sheets** to visualize this aggregate data.


#### Workflow configurations and experiments

The system is designed so that you can easily run new experiments without needing to change any code. Flows are defined in the `/conf` directory, and they reference template files. To create an experiment, you can simply create a new YAML file (e.g., `my-new-experiment.yaml`), inherit from a base configuration and override the specific parameter you want to test. To test a new prompt, the file would look like this:
    ```yaml
    defaults:
      - tja # Inherits all the settings from the base tja.yaml flow

    agents:
      judge:
        template: my-new-prompt.jinja2 # Overrides just the prompt template
    ```
