# Getting Started with Buttermilk Group Chats

Welcome to the Buttermilk Group Chat feature! This guide will walk you through setting up, configuring, running, and customizing multi-agent conversations. You'll learn how to:

*   Define the structure and participants of a group chat.
*   Create detailed YAML configuration files for your chat flows.
*   Customize specialized agents, like Judge agents, using Jinja2 templates.
*   Run your group chats using different modes (console, API) via the Buttermilk CLI.
*   Leverage Hydra for flexible configuration overrides.


## Defining a Custom Group Chat Flow

A group chat in Buttermilk is defined by a collection of agents that interact with each other, managed by an orchestrator, and configured through YAML files. 

A typical operational group chat requires a structured **flow configuration** **file** or **files**. The main flow file is typically saved as a YAML file within the `conf/flows/` directory (e.g., `conf/flows/my_custom_chat.yaml`). 


```yaml
name: MySimpleConceptualChat
job: InteractiveChat


flows:

  # Define a named flow
  vaw:
    # Use the autogen orchestrator
    orchestrator: buttermilk.orchestrators.groupchat.AutogenOrchestrator
    description: 

    # Define the agents that will be available to the Orchestrator
    agents:
      judge:
        role: judge
        description: Expert analysts, particularly suited to assess content with subject matter expertise.
        agent_obj: Judge
        name_components: ["⚖️", "role", "model", "criteria", "unique_identifier"]
        num_runs: 1
        # Set parameters common to all agents of this type
        parameters:
          template: judge

        # Variants are parameters that will be multiplied to run unique agents
        variants:
          model: ${llms.judgers}

        # Inputs specify what information the agent will extract from the group chat
        # This is expressed in JMESPath format: https://jmespath.org/specification.html
        inputs:
          records: "FETCH.outputs||*.records[]"

    # Observers are agents that will participate in the group chat but are not
    # activated by the orchestrator. They are not a 'step' in the 'flow'.
    observers:
      fetch:
        role: FETCH
        name: "🔍 fetch record"
        agent_obj: FetchAgent
        description: A TOOL to retrieve the content of URLs and stored RECORDS required for the discussion.

        # This agent will get records from a specified dataset: 
        data: {}
        inputs:
          prompt: "prompt||content"

      host:
        role: HOST
        name: "🎯 Assistant"
        description: You are the HOST, an assistant to the MANAGER. Help them with their tasks.
        # Choose a type of host. Standard HostAgent just iterates through each step. LLMHostAgent
        # will use a template to determine the next step. 
        agent_obj: LLMHostAgent
        
        # determine whether the host should seek confirmation from the user (MANAGER)
        human_in_loop: true
        parameters:
          template: researchassistant
          model: ${llms.host}
          task: Assign an agent to the task. Repeat.
        inputs: {}

```

## Customizing LLM Agents with Prompts and Criteria

LLM Agents rely on a main Jinja2 template with additional variables. Variables can come from data collected from the outputs of other agents or from other templates. Core templates are stored in the `templates/prompt/` directory. 


## Running Your Group Chat: Configuration and Execution

Buttermilk's command-line interface (CLI), primarily managed by `buttermilk/runner/cli.py`, uses the [Hydra](https://hydra.cc/) library for powerful and flexible configuration management. Hydra allows you to define your application's settings (including group chat flows, agent parameters, LLM choices, etc.) in YAML files, typically located in the `conf/` directory. You can then easily override any part of this configuration directly from the command line.

The primary way to specify how Buttermilk should execute (e.g., run a group chat in the console, or expose it via an API) is through the `run.mode` configuration, which can be set in your main YAML config or overridden via the CLI.


### Overriding Configuration with Hydra

One of the powerful features of Hydra is the ability to override any part of your configuration directly from the command line, without modifying the YAML files. This is extremely useful for experiments, testing different settings, or running specific configurations on the fly.

You can override parameters by providing them as `key=value` arguments. For nested parameters, use dot notation (e.g., `parent.child=value`).

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
    buttermilk run run.mode=console flows=[judger] flow=judger llms=full
    ```
    This command starts Buttermilk in console mode, runs the `judger` flow, and uses the `full` LLM configuration.

*   **API Mode with a specific flow:**
    ```bash
    buttermilk run run.mode=api flows=[your_api_optimized_flow]
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
buttermilk run run.mode=api host=127.0.0.1 port=8000
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
