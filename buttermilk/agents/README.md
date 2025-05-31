# Buttermilk Architecture Overview

Buttermilk provides a framework for building and running multi-agent workflows. It defines core abstractions for agents and orchestrators and includes an optional integration with the Autogen runtime for robust message passing and agent management.

## Core Concepts

These components form the foundation of Buttermilk and are designed to be runtime-agnostic.

1.  **`Agent` (`buttermilk._core.agent.Agent`)**
    *   **Purpose:** The fundamental processing unit in a Buttermilk flow. Each agent encapsulates specific logic, potentially interacting with LLMs, tools, or data sources.
    *   **Interface:**
        *   Inherits from `AgentConfig` for configuration loading (YAML).
        *   `_process(input_data: AgentInput, cancellation_token: CancellationToken | None) -> AsyncGenerator[AgentTrace | None, None]`: The core abstract method. It receives input (`AgentInput`), performs its logic, and asynchronously yields zero or more `AgentTrace` messages.
        *   `__call__(...)`: Makes the agent callable, typically invoking `_process`. Used by orchestrators.
        *   `initialize(**kwargs)`: Optional async method for setup (e.g., loading resources, models). Called once when the agent is created.
        *   `on_reset(...)`: Optional async method to clear internal state.
        *   `handle_control_message(...)`: Optional method to handle non-standard messages (e.g., from UI agents or orchestrators).
    *   **State:** Agents are intended to be stateful. They manage their own internal state or receive necessary context via the `context` field in `AgentInput`.
    *   **Configuration (`AgentConfig`):** Defined in YAML files, specifying `id`, `role`, `description`, the Python class (`agent_obj`), `parameters`, `tools`, `inputs`, etc.

2.  **`Orchestrator` (`buttermilk._core.orchestrator.Orchestrator`)**
    *   **Purpose:** Manages the overall execution of a flow. It coordinates interactions between agents, routes messages, manages shared data/context, and determines the sequence of operations.
    *   **Interface:**
        *   `run(request: Any = None)`: The main abstract method. Subclasses implement the specific flow logic here (e.g., sequential execution, conditional branching, parallel calls).
        *   `_prepare_step(step: StepRequest) -> AgentInput`: Helper method to construct the `AgentInput` message for an agent based on the flow state and step definition.
    *   **Data Flow:** Manages shared flow data (`_flow_data`, `_records`, `_context`) accessible during `_prepare_step`.

3.  **`Contract` (`buttermilk._core.contract`)**
    *   **Purpose:** Defines the Pydantic models for messages passed between components.
    *   **Key Models:**
        *   `FlowMessage`: Base class for all messages.
        *   `AgentInput`: Standard input to an agent's `_process` method. Contains `content`, `context`, `inputs`, `records`, etc.
        *   `AgentTrace`: Standard output yielded by an agent's `_process` method. Contains `content`, `outputs`, `metadata`, `error`, etc.
        *   `UIMessage`, `ManagerMessage`, `ConductorRequest`: Control messages for UI interaction or specific orchestration patterns.
        *   `StepRequest`: Used by some orchestrators (`Selector`) to define the next agent to run and its parameters.

## Autogen Integration (Optional Runtime)

Buttermilk can leverage the Autogen library for its runtime capabilities, providing features like robust message routing, agent lifecycle management, and background processing.

1.  **`AutogenAgentAdapter` (`buttermilk.libs.autogen.AutogenAgentAdapter`)**
    *   **Purpose:** Acts as a bridge or wrapper. It makes a standard Buttermilk `Agent` compatible with Autogen's `RoutedAgent` interface.
    *   **Mechanism:**
        *   Wraps a Buttermilk `Agent` instance.
        *   Implements Autogen message handlers (`@message_handler`).
        *   `handle_request(message: AgentInput, ...)`: Receives Autogen messages, translates them if necessary, and calls the wrapped Buttermilk agent's `__call__` method via `_process_request`.
        *   `_process_request(...)`: Iterates through the `AgentTrace` messages yielded by the Buttermilk agent and uses `publish_message` to send them back into the Autogen runtime.
        *   `handle_oob(...)`: Routes control messages to the Buttermilk agent's `handle_control_message`.
        *   `handle_input()`: Provides a callback mechanism for UI agents to publish user input into the Autogen runtime.

2.  **`AutogenOrchestrator` (`buttermilk.runner.groupchat.AutogenOrchestrator`)**
    *   **Purpose:** An implementation of the Buttermilk `Orchestrator` interface that uses the Autogen `SingleThreadedAgentRuntime`.
    *   **Mechanism:**
        *   `_setup_runtime()`: Initializes the Autogen runtime.
        *   `_register_agents()`: Iterates through the flow's agent configurations, wraps each Buttermilk agent using `AutogenAgentAdapter`, and registers the adapter with the Autogen runtime. Sets up topic subscriptions for message routing.
        *   `_register_collectors()` / `_register_human_in_the_loop()`: Registers special Autogen `ClosureAgent`s to handle message collection (`CLOSURE`) and user confirmation (`CONFIRM`).
        *   `run()`: Starts the Autogen runtime and potentially publishes an initial message. The actual flow logic might be delegated to subclasses or driven by messages within the Autogen runtime.
        *   `_execute_step()`: Publishes an `AgentInput` message to a specific topic within the Autogen runtime, targeting the agent(s) subscribed to that topic (step role).
        *   `_send_ui_message(...)`: Publishes messages specifically for UI agents.

3.  **`Selector` (`buttermilk.runner.selector.Selector`)**
    *   These are concrete orchestrators inheriting from `AutogenOrchestrator`.
    *   They implement `_get_next_step()` as an async generator yielding `StepRequest` objects.
    *   Their `run()` methods typically loop through `_get_next_step()`, potentially ask for user confirmation (`_user_confirmation` queue), and then call `_execute_step()` to trigger the next agent via the Autogen runtime.
    *   `Selector` uses a `CONDUCTOR` agent to decide the next step dynamically.

## How Components Interact (with Autogen Runtime)

1.  **Initialization:** `cli.py` uses Hydra to load configuration and instantiate the chosen `Orchestrator` (e.g., `Selector`).
2.  **Runtime Setup:** The `AutogenOrchestrator`'s `run()` method calls `_setup_runtime()`.
3.  **Agent Registration:** `_register_agents()` creates `AutogenAgentAdapter` instances for each Buttermilk agent defined in the config and registers them with the Autogen runtime. UI agents (`UIAgent` subclasses) are also registered.
4.  **Flow Start:** The orchestrator might prompt the user via `_send_ui_message()`. The UI agent (e.g., `Slack `, `CLIUserAgent`) displays the prompt.
5.  **User Input:** The user provides input (text, button click). The UI agent's callback (`callback_to_groupchat` provided by `AutogenAgentAdapter.handle_input()`) publishes   `ManagerMessage` message into the Autogen runtime via the adapter.
6.  **Orchestration Logic:**
    *   The orchestrator (e.g., `Selector`) might receive the user input (if subscribed) or proceed based on its internal logic (`_get_next_step`).
    *   It determines the next `StepRequest`.
    *   It calls `_prepare_step()` to create an `AgentInput` message.
    *   It calls `_execute_step(step)` which publishes the `AgentInput` to the topic corresponding to the target agent's role (e.g., "RESEARCHER").
7.  **Agent Execution:**
    *   The Autogen runtime routes the `AgentInput` message to the appropriate `AutogenAgentAdapter` subscribed to that topic.
    *   The adapter's `handle_request` calls `_process_request`.
    *   `_process_request` calls the wrapped Buttermilk `Agent`'s `__call__` method (which invokes `_process`).
    *   The Buttermilk `Agent` executes its logic, potentially calling LLMs or tools, and `yield`s `AgentTrace` messages.
8.  **Output Handling:**
    *   `_process_request` receives the yielded `AgentTrace`.
    *   It publishes the `AgentTrace` back into the Autogen runtime on the main topic (`self._topic`).
9.  **Message Collection/Display:**
    *   The `CLOSURE` agent receives the `AgentTrace` and updates the orchestrator's internal state (`_flow_data`, `_context`, `_records`).
    *   UI agents subscribed to the main topic receive the `AgentTrace` via their adapter's `handle_request` (if they implement it) or potentially a dedicated handler, and display it to the user.
10. **Loop/Termination:** The orchestrator continues the loop (steps 6-9) based on its logic until a termination condition is met.
11. **Cleanup:** `_cleanup()` stops the Autogen runtime.

## Extending Buttermilk

### Creating Custom Agents

1.  **Create a Python Class:**
    *   Define a new class that inherits from `buttermilk._core.agent.Agent`.
    *   ```python
        # filepath: /path/to/your/custom_agent.py
        from typing import AsyncGenerator
        from autogen_core import CancellationToken
        from buttermilk._core.agent import Agent, AgentInput, AgentTrace
        from buttermilk import logger

        class MyCustomAgent(Agent):
            async def initialize(self,session_id: str,  **kwargs) -> None:
                logger.info(f"Initializing {self.id}...")
                # Load models, resources, etc.
                # Access config via self.parameters, self.tools, etc.
                pass


        async def _process(self, *, message: AgentInput, 
                cancellation_token: CancellationToken = None, **kwargs
                ) -> AgentTrace | ToolOutput | None:
                logger.info(f"{self.id} received content: {input_data.content}")
                # Access context: input_data.context
                # Access specific inputs: input_data.inputs.get("my_param")
                # Access records: input_data.records

                # Perform agent logic...
                result_content = f"Processed: {input_data.content}"
                result_outputs = {"processed_length": len(input_data.content)}

                # Yield one or more outputs
                yield AgentTrace(
                    source=self.id,session_id=self.session_id,
                    content=result_content,
                    outputs=result_outputs,
                    records=input_data.records, # Pass through records if needed
                    metadata=self.parameters, # Include parameters in metadata
                )

            async def on_reset(self, cancellation_token: CancellationToken | None = None) -> None:
                logger.info(f"Resetting {self.id} state...")
                # Clear internal state if necessary
                pass
        ```
2.  **Implement `_process`:** This is the core logic. It must be an `async def` that returns an `AsyncGenerator[AgentTrace | None, None]`. Use `yield` to return results.
3.  **Implement `initialize` (Optional):** Perform setup tasks here.
4.  **Implement `on_reset` (Optional):** Clear any internal state if the agent needs to be reset.
5.  **Configure in YAML:**
    *   In your flow configuration (e.g., `conf/flows/my_flow.yaml`), define the agent:
    *   ```yaml
        # filepath: conf/flows/my_flow.yaml
        # ... other flow config ...
        agents:
          my_step_name: # Logical name for this step in the flow
            id: MY_AGENT_ID # Unique ID for the agent instance
            role: "My Custom Role"
            description: "This agent does custom processing."
            agent_obj: MyCustomAgent # The name of your Python class
            # Optional: Define parameters, tools, input mappings
            parameters:
              my_config_param: "value"
            inputs:
              my_param: "some_static_value" # Or map from other agents: ${other_agent.output_key}
            tools: []
            variants: {} # Add variants if needed
        # ... rest of agents ...
        ```
    *   Ensure the Python file containing your agent class is importable.

### Creating Custom Orchestrators

1.  **Choose Base Class:**
    *   **Inherit from `buttermilk._core.orchestrator.Orchestrator`:** If you want to build a runtime-agnostic orchestrator or integrate with a different runtime. You will need to implement agent instantiation, message passing, and state management yourself.
    *   **Inherit from `buttermilk.runner.groupchat.AutogenOrchestrator`:** If you want to leverage the Autogen runtime. This provides agent registration, message passing via topics, and collector/confirm agents out-of-the-box.
2.  **Create a Python Class:**
    *   ```python
        # filepath: /path/to/your/custom_orchestrator.py
        import asyncio
        from buttermilk._core.orchestrator import Orchestrator # Or AutogenOrchestrator
        from buttermilk._core.contract import AgentInput  # etc.
        
        
        

        class MyCustomOrchestrator(Orchestrator): # Or AutogenOrchestrator

            async def run(self, request: Any = None) -> None:
                logger.info(f"Starting custom flow: {self.name}")

                # --- If inheriting from AutogenOrchestrator ---
                await self._setup_runtime()
                # Publish initial message, prompt user, etc.
                # Example: Get initial user input
                initial_input = None
                if isinstance(request, UserInstructions):
                    initial_input = request
                elif isinstance(request, str):
                     initial_input = UserInstructions(source="user", role="user", content=request)
                # else: prompt user via _send_ui_message and _user_confirmation queue

                if not initial_input:
                     logger.warning("No initial input provided.")
                     await self._cleanup()
                     return

                # --- Custom Flow Logic ---
                # Example: Send input to AgentA, then AgentB
                try:
                    # Step 1: Agent A
                    step_a_request = StepRequest(role="AGENT_A_ROLE", content=initial_input.content) # Get role from self.agents keys
                    await self._execute_step(step_a_request) # Uses Autogen publish

                    # Need a way to wait for and get AgentA's output
                    # This often involves the collector agent updating self._flow_data
                    # or using asyncio events/queues if not using Autogen runtime fully.
                    await asyncio.sleep(5) # Simplistic wait, better mechanisms needed

                    agent_a_output = self._flow_data.get_latest("AGENT_A_ROLE") # Example access
                    if not agent_a_output:
                         raise Exception("Agent A did not respond")

                    # Step 2: Agent B
                    step_b_request = StepRequest(role="AGENT_B_ROLE", prompt=agent_a_output.content)
                    await self._execute_step(step_b_request)

                    # ... wait for Agent B ...

                    logger.info("Custom flow finished.")

                except Exception as e:
                    logger.error(f"Error in custom flow: {e}")
                finally:
                    await self._cleanup() # Important for AutogenOrchestrator

            # --- If inheriting from core Orchestrator ---
            # async def run(self, request: Any = None) -> None:
            #     # 1. Instantiate agents (e.g., self.agents["MY_STEP_NAME"].get_configs()[0]...)
            #     # 2. Manage message loop (e.g., asyncio tasks, queues)
            #     # 3. Call agent.__call__(agent_input) directly
            #     # 4. Handle agent output (yielded AgentTrace)
            #     # 5. Manage state and context
            #     pass
        ```
3.  **Implement `run`:** Define the sequence of agent interactions, data flow, and termination conditions. If using `AutogenOrchestrator`, leverage `_execute_step`, `_send_ui_message`, `_user_confirmation`, and potentially `_flow_data`.
4.  **Configure in YAML:**
    *   Set the `orchestrator` key in your flow config:
    *   ```yaml
        # filepath: conf/flows/my_flow.yaml
        name: my_custom_flow
        description: A flow run by a custom orchestrator
        orchestrator: MyCustomOrchestrator # Name of your orchestrator class
        agents:
          # ... agent definitions ...
        ```
5.  **Update `cli.py` (if needed):** Ensure your orchestrator class is imported and potentially added to the list or logic in `runner/cli.py` if it's not automatically picked up by the YAML `orchestrator` key.

## Configuration

*   **Hydra:** Buttermilk uses Hydra (`hydra.main`) for configuration management.
*   **YAML Files:** Configurations for flows, agents, models, and credentials reside in the `conf/` directory.
*   **Structure:** Configuration is hierarchical, allowing overrides and composition (e.g., `conf/config.yaml` defines defaults and selects the active flow).
*   **Resolvers:** OmegaConf resolvers (like `${llms.general}`) allow dynamic referencing of other configuration values.

## Running a Flow

*   The primary entry point for console-based execution is `buttermilk.runner.cli.py`.
*   Run using Python: `python -m buttermilk.runner.cli flow=<your_flow_name>`
*   Hydra allows overriding configuration parameters from the command line (e.g., `python -m buttermilk.runner.cli flow=my_flow agents.my_step_name.parameters.my_config_param=new_value`).