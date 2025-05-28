# Getting Started with Buttermilk Group Chats

Welcome to the Buttermilk Group Chat feature! This guide will walk you through setting up, configuring, running, and customizing multi-agent conversations. You'll learn how to:

*   Define the structure and participants of a group chat.
*   Create detailed YAML configuration files for your chat flows.
*   Customize specialized agents, like Judge agents, using Jinja2 templates.
*   Run your group chats using different modes (console, API) via the Buttermilk CLI.
*   Leverage Hydra for flexible configuration overrides.

## Prerequisites

Before you begin, please ensure you have:

*   An active Buttermilk account (if applicable for cloud features).
*   The Buttermilk CLI installed and configured on your system.
*   A basic understanding of Buttermilk concepts like flows and agents. If you're new to Buttermilk, consider reviewing the general documentation first.

## Understanding Group Chat Configuration

A group chat in Buttermilk is defined by a collection of agents that interact with each other, managed by an orchestrator, and configured through YAML files. While a very basic conceptual setup might look like this:

```yaml
# conceptual_group_chat_example.yaml
# This is a simplified example to illustrate the core idea.
# Practical configurations are more detailed.
name: MySimpleConceptualChat
agents:
  - name: UserProxy
    type: UserProxyAgent
    description: Handles user interaction.
  - name: Assistant
    type: AssistantAgent
    description: Provides responses.
  - name: Specialist
    type: SomeSpecializedAgent
    description: Performs a specific task.
```
...a typical operational group chat requires a more structured **flow configuration file**. These files detail aspects like datasets, the orchestrator, and the specific behaviors and LLM configurations for each agent.

## Defining a Custom Group Chat Flow

A full Buttermilk group chat flow involves detailed configuration, typically saved as a YAML file within the `conf/flows/` directory (e.g., `conf/flows/my_custom_chat.yaml`). This structure allows Buttermilk to discover and run your flow using a command like `buttermilk run flow=my_custom_chat`.

A flow configuration file for a group chat generally includes these key sections:
*   **`name`**: A unique identifier for your flow.
*   **`description`**: A short description of what the flow does.
*   **`dataset`** (optional): Specifies the data source for the flow. This could be records for agents to process, topics for discussion, or initial prompts.
*   **`orchestrator`**: Defines the group chat manager, including its type, the LLM it uses, and its main instructions.
*   **`agents`**: A list defining each participating agent, including their roles, types, LLM configurations, and system messages.

### Key Configuration Sections:

1.  **Dataset Configuration:**
    If your group chat needs to process items from a dataset (e.g., questions to answer, documents to analyze) or start with predefined prompts:
    ```yaml
    # Example Dataset Configuration
    dataset:
      name: "my_discussion_topics"
      # For BigQuery source:
      # project: "my-gcp-project"
      # table_name: "my_bq_table"
      # For local files (e.g., JSONL, CSV):
      # path: "data/my_topics.jsonl" # Relative to a data directory or an absolute path
      # format: "jsonl" # or "csv", etc.
      # For simple, in-configuration prompts (useful for fixed scenarios):
      prompts:
        - "Discuss the future of AI in education."
        - "What are the ethical implications of autonomous vehicles?"
    ```
    The exact fields and their usage (e.g., how `prompts` are injected into the chat) will depend on your specific data loader and orchestrator setup in Buttermilk.

2.  **Orchestrator Configuration:**
    The orchestrator is responsible for managing the conversation flow, deciding which agent speaks next, and enforcing rules like maximum turns.
    ```yaml
    # Example Orchestrator Configuration
    orchestrator:
      type: "groupchat" # Specifies the GroupChatOrchestrator (actual type name may vary)
      max_turns: 20      # Maximum number of turns in the conversation
      # LLM configuration for the orchestrator itself (if it uses an LLM for decision-making,
      # summarization, or other management tasks). This might not always be directly used
      # by all orchestrator types but can serve as a default or for specific features.
      llm:
        provider: "openai" # or "anthropic", "vertexai", etc.
        model: "gpt-4-turbo-preview"
        temperature: 0.7
      # General instructions or system message for the entire group chat.
      # Individual agents will also have their own specific system messages.
      system_message: |
        This is a collaborative discussion. Agents should work together
        to address the user's query or explore the given topic comprehensively.
        Be respectful and build upon each other's contributions.
    ```

3.  **Agent Configuration:**
    This is a list where you define each agent participating in the chat.
    ```yaml
    # Example Agent Definitions
    agents:
      - name: "ChatInitiator" # Often a UserProxyAgent to represent the human or start the chat
        type: "UserProxyAgent" # Actual agent type name may vary
        description: "The agent that starts the conversation, often representing the human user or an initial prompt."
        # UserProxyAgents typically relay user input in interactive modes or can be configured
        # to inject prompts from a dataset. They usually don't have their own LLM for generation
        # but might have settings for how they handle input/output.
        # human_input_mode: "TERMINATE" # Example: defines when to ask for human input.
                                      # Options: ALWAYS, TERMINATE, NEVER.

      - name: "ResearchSpecialist"
        type: "AssistantAgent" # A generic or custom LLM-backed assistant agent
        description: "Specializes in finding and presenting research information."
        llm:
          provider: "openai"
          model: "gpt-4-turbo-preview"
          temperature: 0.5
        system_message: |
          You are a research specialist. Your primary goal is to find and provide accurate,
          well-sourced information relevant to the ongoing discussion or user's query.
          If you have access to tools (like web search), use them effectively.
        # You might also define specific tools available to an agent:
        # tools:
        #   - type: "web_search_tool" # Actual tool type/name may vary
        #   - type: "code_interpreter_tool"

      - name: "CriticalThinker"
        type: "AssistantAgent" # Another assistant agent with a different persona
        description: "Analyzes information, asks clarifying questions, and ensures depth of discussion."
        llm:
          provider: "anthropic"
          model: "claude-3-opus-20240229"
          temperature: 0.8
        system_message: |
          You are a critical thinker. Your role is to analyze the information provided,
          identify assumptions, and ask pertinent questions to ensure clarity and accuracy.
          Challenge statements respectfully and encourage deeper thought.
    ```

### Complete Example: `conf/flows/my_custom_chat.yaml`

Here's how these pieces might come together in a file named `conf/flows/my_custom_chat.yaml`:

```yaml
name: "MyCustomDiscussionFlow"
description: "A group chat flow for discussing various topics with specialized agents."

# Optional: If the chat is seeded by predefined topics/prompts
dataset:
  name: "discussion_starters"
  prompts:
    - "What are the key challenges and opportunities in renewable energy adoption?"
    - "Explore the impact of social media on political discourse."

orchestrator:
  type: "groupchat"
  max_turns: 15
  system_message: "Facilitate a productive discussion among the agents to explore the given topic."
  # LLM for the orchestrator itself (e.g., for turn management or summarization if applicable)
  llm:
    provider: "openai"
    model: "gpt-3.5-turbo"

agents:
  - name: "UserProxy"
    type: "UserProxyAgent" # Represents the user or initiates with a dataset prompt
    description: "Initiates the discussion with a topic and relays user input if interactive."
    # For non-interactive mode driven by dataset prompts, human_input_mode might be "NEVER",
    # or the agent could be configured to automatically process prompts from the dataset.
    # If interactive, it might wait for user input via the console.

  - name: "ProArgumentAgent"
    type: "AssistantAgent"
    description: "Develops and presents arguments in favor of a position or explores a positive aspect of the topic."
    llm:
      provider: "openai"
      model: "gpt-4-turbo-preview"
      temperature: 0.7
    system_message: |
      You are the ProArgumentAgent. Your task is to explore and articulate the
      positive aspects, benefits, and supporting arguments related to the current
      discussion topic. Be constructive, provide well-reasoned points, and cite sources if possible.

  - name: "ContraArgumentAgent"
    type: "AssistantAgent"
    description: "Develops and presents counter-arguments, challenges, or alternative perspectives."
    llm:
      provider: "anthropic"
      model: "claude-3-sonnet-20240229" # Example of using a different LLM provider/model
      temperature: 0.75 # Slightly higher temperature for more varied responses
    system_message: |
      You are the ContraArgumentAgent. Your task is to critically examine the topic
      and any arguments presented. Raise potential challenges, drawbacks,
      counter-arguments, and alternative perspectives. Ensure your points are
      well-reasoned, respectful, and contribute to a balanced and thorough discussion.

  - name: "SummarizerAgent"
    type: "AssistantAgent"
    description: "Periodically summarizes the discussion or provides a final summary."
    llm:
      provider: "openai"
      model: "gpt-3.5-turbo" # A smaller model might be sufficient for summarization
      temperature: 0.5
    system_message: |
      You are SummarizerAgent. Your role is to periodically provide concise
      summaries of the ongoing discussion, highlighting key arguments, agreements,
      and disagreements. You may be asked to summarize at the end of the conversation.
```

By saving this YAML as `conf/flows/my_custom_chat.yaml`, you can then execute it using:
```bash
buttermilk run flow=my_custom_chat
```
You can also combine this with other Hydra overrides as discussed in the next section (e.g., `buttermilk run flow=my_custom_chat run.mode=api` or `buttermilk run flow=my_custom_chat agents.ProArgumentAgent.llm.model=gpt-3.5-turbo`).

## Customizing Judge Agents with Prompts and Criteria

Judge agents (or Evaluator agents) play a crucial role in assessing the output of other agents or the overall conversation quality. Their behavior is heavily guided by prompt templates and criteria templates, which are typically written in Jinja2 format and stored in the `templates/prompt/` and `templates/criteria/` directories respectively.

### 1. Writing Jinja2 Prompt Templates for Judges

A prompt template for a Judge agent dynamically constructs the instructions given to the LLM that powers the Judge. It can include placeholders for various pieces of information, such as the text to be evaluated, specific guidelines, or examples.

**Example: `templates/prompt/my_custom_judge_prompt.jinja2`**

Let's say you want a Judge agent to evaluate a piece of text for clarity and conciseness.

```jinja2
{# templates/prompt/my_custom_judge_prompt.jinja2 #}
You are an expert evaluator. Your task is to assess the following text for clarity and conciseness based on the provided criteria.

**Text to Evaluate:**
{{ submission_text }}

**Evaluation Guidelines:**
{{ critique_guidelines }}

Provide your assessment in the format specified by the criteria.
```

*   `{{ submission_text }}`: This variable would be replaced with the actual text the Judge needs to evaluate.
*   `{{ critique_guidelines }}`: This could be a general set of instructions or specific points to focus on, passed from the agent's configuration.

Save this file as `my_custom_judge_prompt.jinja2` inside your `templates/prompt/` directory.

### 2. Writing Jinja2 Criteria Templates for Judges

Criteria templates define the structure and specific points the Judge agent should use for its evaluation. This can be free-form text, or it can be structured (e.g., to request JSON output) for easier parsing of the Judge's feedback.

**Example: `templates/criteria/my_custom_criteria.jinja2`**

This template asks the Judge to provide scores and reasoning in a structured way.

```jinja2
{# templates/criteria/my_custom_criteria.jinja2 #}
Please provide your evaluation based on the following aspects:

1.  **Clarity Score (1-5):** How clear and easy to understand is the text?
    *   Score: {{'{'}} "clarity_score": <score_value> {{'}'}}
    *   Reasoning: {{'{'}} "clarity_reasoning": "<your_reasoning>" {{'}'}}

2.  **Conciseness Score (1-5):** Is the text to the point and free of unnecessary jargon or verbosity?
    *   Score: {{'{'}} "conciseness_score": <score_value> {{'}'}}
    *   Reasoning: {{'{'}} "conciseness_reasoning": "<your_reasoning>" {{'}'}}

Return your response as a single JSON object containing these scores and reasons.
Example:
{
  "clarity_score": 4,
  "clarity_reasoning": "The main points were clear, but some sentences could be simpler.",
  "conciseness_score": 3,
  "conciseness_reasoning": "The text was a bit repetitive in places."
}
```
*   Note the use of `{{'{'}}` and `{{'}'}}` to output literal curly braces for the JSON structure, preventing Jinja2 from interpreting them.

Save this file as `my_custom_criteria.jinja2` inside your `templates/criteria/` directory.

### 3. Configuring the Judge Agent in YAML

Once you have your custom Jinja2 templates, you need to tell your Judge agent to use them. This is done in the flow configuration YAML file where you define your agents.

**Example: Part of `conf/flows/my_judging_flow.yaml`**

```yaml
# ... (flow name, description, other agents, etc.)

agents:
  # ... (other agents in your group chat or flow)

  - name: "QualityAssessorAgent"
    type: "JudgeAgent" # Or your specific Judge agent type
    description: "Assesses submissions for clarity and conciseness using custom templates."
    llm:
      provider: "openai"
      model: "gpt-4-turbo-preview" # Judges often benefit from powerful models
      temperature: 0.3
    
    # Configuration for the Judge agent's behavior
    judge_config: # This key (`judge_config`) is illustrative; the actual key for configuring
                  # prompt/criteria templates and related variables might vary based on your
                  # specific JudgeAgent or EvaluatorAgent implementation in Buttermilk.
      prompt_template: "prompt/my_custom_judge_prompt.jinja2" # Path relative to project's templates/ directory
      criteria_template: "criteria/my_custom_criteria.jinja2" # Path relative to project's templates/ directory
      
      # Variables to be passed into the prompt and/or criteria templates
      template_vars:
        critique_guidelines: |
          Focus on whether the text is understandable by a general audience
          and if it conveys its message without unnecessary length.
          Scores should be integers between 1 (poor) and 5 (excellent).
          Provide constructive feedback.

# ... (other parts of your flow configuration, like the orchestrator)
```

**Key Points for YAML Configuration:**

*   **Agent Type**: Ensure you're using the correct `type` for your evaluating agent (e.g., `JudgeAgent`, `EvaluatorAgent`, or a custom type).
*   **Template Paths**: The `prompt_template` and `criteria_template` values are typically paths relative to a base `templates/` directory in your Buttermilk project (e.g., `templates/prompt/your_template.jinja2`).
*   **`template_vars`**: This section (the key name might vary, e.g., `prompt_vars`, `params`) is where you define static variables that your Jinja2 templates expect. Dynamic variables like `submission_text` (the content to be judged) are usually passed by the Buttermilk framework or orchestrator when an evaluation is triggered.

By setting up your Judge agents with custom Jinja2 templates, you can tailor their evaluation process precisely to your needs, ensuring consistent, structured, and relevant feedback for your group chat interactions or other agent outputs.

## Running Your Group Chat: Configuration and Execution

Buttermilk's command-line interface (CLI), primarily managed by `buttermilk/runner/cli.py`, uses the [Hydra](https://hydra.cc/) library for powerful and flexible configuration management. Hydra allows you to define your application's settings (including group chat flows, agent parameters, LLM choices, etc.) in YAML files, typically located in the `conf/` directory. You can then easily override any part of this configuration directly from the command line.

The primary way to specify how Buttermilk should execute (e.g., run a group chat in the console, or expose it via an API) is through the `run.mode` configuration, which can be set in your main YAML config or overridden via the CLI.

### Main Configuration File (`conf/config.yaml` or a custom one):

```yaml
# conf/config.yaml (or a custom config file)
defaults:
  - run: console # Default run mode
  # ... other defaults

run:
  mode: console # Can be console, api, batch, etc.
  # ... other run configurations

# Flow specific configuration for your groupchat
flow: your_groupchat_flow_name
# ... other parameters like record_id, prompt, etc.

# Agent configurations would be part of your specific flow's config
# or loaded dynamically by the groupchat orchestrator.
```

### Overriding Configuration with Hydra

One of the powerful features of Hydra is the ability to override any part of your configuration directly from the command line, without modifying the YAML files. This is extremely useful for experiments, testing different settings, or running specific configurations on the fly.

You can override parameters by providing them as `key=value` arguments after the `buttermilk run` command. For nested parameters, use dot notation (e.g., `parent.child=value`).

**Common Override Examples:**

1.  **Selecting a Specific Flow:**
    If you have multiple flow configurations (e.g., `conf/flows/default.yaml`, `conf/flows/judger.yaml`, `conf/flows/trans.yaml`), you can switch between them. The `flow` parameter in `conf/config.yaml` (or your main config) usually points to a default flow. To run the `judger` flow:
    ```bash
    buttermilk run flow=judger
    ```
    This tells Buttermilk to load the configuration from `conf/flows/judger.yaml` (or however `judger` is defined in your flow configurations).

2.  **Changing LLM Configuration:**
    Buttermilk might have different LLM configurations (e.g., for different models, API keys, or settings). If these are structured as a Hydra config group (e.g., in `conf/llms/`), you could switch to a configuration named `full` (defined in `conf/llms/full.yaml`) like this:
    ```bash
    buttermilk run llms=full
    ```
    This would make the settings from `conf/llms/full.yaml` active for the `llms` part of the configuration. If the LLM settings are nested, for example under `bm` (Buttermilk global settings), the override might look like `bm.llms=full`. The exact path depends on your `config.yaml` structure.

**Combining Overrides with Run Modes:**

These overrides can be combined with the `run.mode` setting:

*   **Console Mode with specific flow and LLM config:**
    ```bash
    buttermilk run run.mode=console flow=judger llms=full
    ```
    This command starts Buttermilk in console mode, runs the `judger` flow, and uses the `full` LLM configuration.

*   **API Mode with a specific flow:**
    ```bash
    buttermilk run run.mode=api flow=your_api_optimized_flow
    ```
    This starts the API server, configured to serve `your_api_optimized_flow`. You can also add other overrides like `bm.llm.default_model=claude-3-opus` if your configuration supports such specific overrides:
    ```bash
    buttermilk run run.mode=api flow=my_flow bm.llm.default_model=claude-3-opus
    ```

Remember that the available override keys (`flow`, `llms`, `bm.llm.default_model`, etc.) depend on how your Hydra configuration files (in `conf/`) are structured.

### 1. Console Mode

This mode allows you to run and interact with a group chat directly in your terminal. It's useful for development, testing, and direct interaction.

**How it works:**

The `buttermilk/runner/cli.py` script, when `run.mode` is set to `console` (or if it's the default and not overridden), executes the specified flow.
As seen in `cli.py`:
```python
# buttermilk/runner/cli.py snippet (illustrative)
# ...
        case "console":
            ui = CLIUserAgent() # Handles terminal input/output
            # Prepare the RunRequest with command-line parameters or loaded flow config
            run_request = RunRequest(
                ui_type=conf.ui, # From configuration
                flow=conf.get("flow"), # Name of the flow to run
                # ... other parameters like record_id, prompt, uri
                callback_to_ui=ui.callback_to_ui, # Function for agents to send messages to UI
            )

            # Run the flow synchronously
            logger.info(f"Running flow '{conf.flow}' in console mode...")
            asyncio.run(flow_runner.run_flow(run_request=run_request, wait_for_completion=True))
            logger.info(f"Flow '{run_request.flow}' finished.")
# ...
```
A `CLIUserAgent` (or similar) is instantiated to handle input from your keyboard and display output from the agents to your terminal. If your group chat flow is designed for interaction (e.g., includes a `UserProxyAgent` that asks for input), this is where you'll type your messages.

**Running in Console Mode:**

Assuming your main configuration (`conf/config.yaml`) defaults to `run.mode=console`, or you have a specific flow configuration (e.g., `conf/flows/my_custom_chat.yaml`) that you want to run:
```bash
# Run the flow specified in your default config (e.g., conf/config.yaml's 'flow' variable)
buttermilk run

# Run a specific flow (e.g., 'my_custom_chat' which refers to conf/flows/my_custom_chat.yaml)
buttermilk run flow=my_custom_chat

# Explicitly set console mode and run a specific flow
buttermilk run run.mode=console flow=my_custom_chat
```
Replace `my_custom_chat` with the actual name of your group chat flow file (without the `.yaml` extension).

### 2. API Mode

This mode starts a FastAPI web server, exposing your Buttermilk flows (including group chats, if designed accordingly) via an HTTP API. This is useful for integrating Buttermilk with other applications or services.

**How it works:**

When `run.mode` is set to `api`, `buttermilk/runner/cli.py` initializes and starts a FastAPI application using Uvicorn.
From `cli.py`:
```python
# buttermilk/runner/cli.py snippet (illustrative)
# ...
        case "api":
            # Starts a FastAPI web server.
            logger.info("Starting FastAPI API server...")
            fastapi_app = create_fastapi_app( # From buttermilk.api.flow
                bm=bm_instance,  # Pass the global BM instance
                flows=flow_runner,  # Pass the FlowRunner, which knows about configured flows
            )
            # ... (Uvicorn server configuration)
            uvicorn_config = uvicorn.Config(
                app=fastapi_app,
                host=str(conf.get("host", "0.0.0.0")), # Host from config or default
                port=int(conf.get("port", 8000)),    # Port from config or default
                reload=bool(conf.get("reload", False)), # Hot reloading (dev only)
                # ... other uvicorn settings
            )
            api_server = uvicorn.Server(config=uvicorn_config)
            logger.info(f"FastAPI server starting on http://{uvicorn_config.host}:{uvicorn_config.port}")
            api_server.run() # This is a blocking call
# ...
```
The `create_fastapi_app` function (likely found in `buttermilk.api.flow` or a similar module) sets up the necessary API endpoints. These endpoints would allow you to start new chat sessions, send messages, retrieve responses, etc., typically using HTTP requests (e.g., POST, GET).

**Running in API Mode:**

```bash
# Run Buttermilk in API mode, using flow configurations defined
# (e.g., the default flow or allowing flow selection via API calls)
buttermilk run run.mode=api

# You can also specify host and port if your config supports it and you want to override
buttermilk run run.mode=api host=127.0.0.1 port=8080
```
This will start the Uvicorn server (defaulting to a host and port like `http://0.0.0.0:8000` unless configured otherwise). You would then interact with the group chat by sending HTTP requests to the API endpoints defined by the FastAPI application. Consult the Buttermilk API documentation or the relevant source code (e.g., `buttermilk.api.routes`) for details on available endpoints and request/response formats.

## Interacting with the Group Chat (Console Mode)

When running in `console` mode, your interaction with the group chat depends on how your agents, particularly the `UserProxyAgent` (or equivalent), are configured.

*   **Initiating Conversation**: If your flow doesn't start with a predefined prompt from a dataset, the `UserProxyAgent` will typically wait for your first message. Simply type your query or starting statement and press Enter.
*   **Agent Responses**: Agents will respond in sequence according to the orchestrator's logic. Their messages will be printed to the console.
*   **Providing Input**: If the `UserProxyAgent` is configured to ask for human input (e.g., `human_input_mode: ALWAYS` or `human_input_mode: TERMINATE` and the termination condition is met), you'll be prompted to enter your response.
*   **Ending Conversation**: To end the chat, you might type a specific keyword like "exit", "quit", or "stop" (this depends on the `UserProxyAgent`'s configuration or the orchestrator's termination conditions). Pressing `Ctrl+C` will also terminate the Buttermilk application.

The specifics of interaction (e.g., special commands, how agents are addressed) will be determined by the design of your group chat flow and the capabilities of the agents involved.

## Further Customization and Advanced Topics

This guide covers the fundamentals of getting started with group chats in Buttermilk. There are many avenues for further customization and more advanced usage:

*   **Agent Behavior**: Beyond system messages and LLM choices, you might explore:
    *   Developing custom agent types with unique Python code.
    *   Equipping agents with specialized tools (e.g., web search, code execution, database lookups).
    *   Implementing more complex conversational patterns and memory management.
*   **Orchestrator Logic**: For highly custom group interactions, you might need to modify or create new orchestrator logic to control the flow of conversation.
*   **Data Handling**: Integrating diverse data sources or implementing more sophisticated data processing pipelines for your agents.
*   **LLM Fine-tuning**: Fine-tuning LLMs for specific agent roles or tasks within your group chat.
*   **Monitoring and Logging**: Buttermilk likely provides logging mechanisms. Integrating with monitoring tools can help you track agent performance and conversation quality.
*   **Batch Processing**: Running group chat flows over many records or prompts in a batch manner (e.g., using `run.mode=batch` or `run.mode=batch_run`).
*   **Error Handling and Resilience**: Implementing robust error handling within your agent and flow configurations.

## Troubleshooting Tips

*   **Configuration Errors**: Carefully check your YAML syntax. Hydra can be strict about indentation and structure. Error messages from Hydra usually point to the problematic part of the configuration.
*   **Template Not Found**: If using Jinja2 templates, ensure the paths in your YAML configuration correctly point to the template files relative to the `templates/` directory.
*   **API Issues**: When using API mode, check that the server has started correctly and that you are sending requests to the correct endpoints with the correct payload structure. Use tools like `curl` or Postman to test API interactions.
*   **Agent Not Responding**:
    *   Check the agent's LLM configuration (API keys, model availability).
    *   Look at the console logs for error messages from the agent or the LLM provider.
    *   Ensure the agent's system message is clear and that the agent is capable of handling the current task or input.
*   **Permissions**: If reading/writing files or accessing cloud services, ensure Buttermilk has the necessary permissions.

---

For more detailed information on specific components, agent types, or advanced features, please refer to the main Buttermilk documentation and the source code of relevant modules. Happy chatting!
