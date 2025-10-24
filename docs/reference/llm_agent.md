# LLM Agent Call Stack Overview

Here is a high-level overview of the call stack, from the initial agent invocation to the final response processing:
```mermaid
sequenceDiagram
    participant User
    participant Agent.invoke()
    participant LLMAgent._process()
    participant AutoGenWrapper.call_chat(max_tool_iterations=1)
    participant AutoGenWrapper.create()
    participant LLM_API
    participant AutoGenWrapper._execute_tools()

    User->>Agent.invoke(): Start with AgentInput
    Agent.invoke()->>LLMAgent._process(): Calls subclass logic
    LLMAgent._process()->>AutoGenWrapper.call_chat(): Hands off to LLM wrapper
    
    Note over AutoGenWrapper.call_chat(): Manages the conversation turn
    AutoGenWrapper.call_chat()->>AutoGenWrapper.create(): Makes the first LLM call
    AutoGenWrapper.create()->>LLM_API: Sends messages
    LLM_API-->>AutoGenWrapper.create(): Returns raw response
    
    alt LLM requests tool calls
        AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns FunctionCall(s)
        AutoGenWrapper.call_chat()->>AutoGenWrapper._execute_tools(): Executes tools
        AutoGenWrapper._execute_tools()-->>AutoGenWrapper.call_chat(): Returns tool results
        
        Note over AutoGenWrapper.call_chat(): Sends tool results back to LLM
        AutoGenWrapper.call_chat()->>AutoGenWrapper.create(): Makes second LLM call
        AutoGenWrapper.create()->>LLM_API: Sends messages + tool results
        LLM_API-->>AutoGenWrapper.create(): Returns final text response
        AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns CreateResult/ModelOutput
    else LLM returns text/JSON directly
        AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns CreateResult/ModelOutput
    end

    alt Error during processing (fail-fast)
         AutoGenWrapper.create()--x AutoGenWrapper.call_chat(): raises ProcessingError
         AutoGenWrapper.call_chat()--x LLMAgent._process(): raises ProcessingError
         LLMAgent._process()--x Agent.invoke(): raises ProcessingError
    end

    AutoGenWrapper.call_chat()-->>LLMAgent._process(): Returns final result
    LLMAgent._process()-->>Agent.invoke(): Returns AgentOutput
    Agent.invoke()-->>User: Returns ExecutionTrace
```

   Detailed Execution Flow

  Here’s a step-by-step breakdown of what happens in each method.

  1. Agent.invoke() - The Entry Point
   * File: buttermilk/_core/agent.py
   * Purpose: This is the main, public-facing method to run any agent.
   * Flow:
       1. It receives an AgentInput (or StepRequest).
       2. It prepares the final input by merging the agent's internal state (like conversation history) with the message's data via
          _add_state_to_input().
       3. It calls trace_and_execute(), which in turn calls the abstract _process() method. This is where control is handed to the specific
          agent implementation (in this case, LLMAgent).
       4. After _process() returns, it wraps the result in an ExecutionTrace object for logging and returns it.

  2. LLMAgent._process() - The Agent's Core Logic (fail-fast)
   * File: buttermilk/agents/llm.py
   * Purpose: Implements the specific logic for an LLM-based agent.
   * Flow:
       1. Template Filling: It calls _fill_template() to render the prompt, creating a list of LLMMessage objects ready for the API.
       2. LLM Call: It calls _call_llm(), which is a helper that gets the correct AutoGenWrapper client and passes the request to it.
     3. Response Handling: It receives a result from the LLM call (CreateResult or ModelOutput). Any error is raised as ProcessingError.
       * If the result is a ModelOutput, it extracts the parsed_object.
       * If it's a plain CreateResult, it uses the content.
       4. Output Creation: It packages the final content (string, Pydantic object, or error) into an AgentOutput object and returns it up
          the chain.

  3. LLMAgent._call_llm() -> AutoGenWrapper.call_chat() - Orchestrating the Chat
   * Files: buttermilk/agents/llm.py -> buttermilk/_core/llms.py
   * Purpose: call_chat() manages a full "turn" of conversation, which might involve multiple back-and-forths with the LLM if tools are
     used.
   * Flow:
  1. It makes an initial call to self.create() with the user's messages and the list of available tools. call_chat enforces a max_tool_iterations guard (default 1) to avoid infinite loops.
       2. Scenario A: The LLM returns tool calls.
           * The create() method returns a CreateResult where the content is a list of FunctionCall objects.
           * call_chat() then calls _execute_tools() to run the functions.
           * The tool results are added to the message history.
           * call_chat() calls self.create() a second time with the updated history to get a final, synthesized response from the LLM.
           * If a schema was provided, the final response is parsed against it to ensure structured output.
  3. Scenario B: The LLM returns text or JSON directly.
           * create() returns a CreateResult or ModelOutput. call_chat() simply passes this result back up to LLMAgent.
     4. Scenario C: An error occurs.
       * If create() or _execute_tools() runs into a problem, ProcessingError is raised. Agents do not attempt recovery.

  4. AutoGenWrapper.create() - The Atomic LLM Call (normalized outputs)
   * File: buttermilk/_core/llms.py
   * Purpose: This method handles a single API call to the LLM, including structured output and error handling. It supports both tools and schema simultaneously.
   * Flow:
       1. Argument Preparation: It prepares the arguments for the underlying client. If you provide a Pydantic schema:
          - For models with native structured output: Uses json_output parameter regardless of tools
          - For models without native structured output and no real tools: Creates a "fake tool" to force structured output
          - For models without native structured output but with real tools: Allows tools to execute, then parses final response against schema
       2. API Call: It executes the call using _execute_with_retry(). API-level errors (like connection issues) are handled here.
     3. Result Processing: After a successful API call, it processes the response.
       * If the response is invalid (e.g., empty), it returns ModelOutput with error details instead of raising ProcessingError
       * Tool calls are handled intelligently - real tools are returned for execution, fake schema tools are parsed immediately
       * If a schema was requested, the response is always parsed against it, returning ModelOutput with parsed_object set
       * Content is normalized to string format while preserving parsed objects
       * Errors are captured in ModelOutput rather than raised, allowing tracing to complete

  5. AutoGenWrapper._call_tool() & _execute_tools() - Function Execution
   * File: buttermilk/_core/llms.py
   * Purpose: These methods handle the execution of tools requested by the LLM.
   * Flow:
       1. _execute_tools() receives the list of FunctionCall objects.
       2. It iterates through them and calls _call_tool() for each one, running them concurrently.
       3. _call_tool() finds the corresponding Tool object, executes its run_json method, and wraps the output in a FunctionExecutionResult
          object, which is what the LLM expects to see in the message history.

  How ModelOutput Fits In

  * `ModelOutput` is a success-path object, created in create() when a schema is used. content is a normalized JSON string, and parsed_object holds the hydrated Pydantic model.
  * Errors at any stage result in a raised ProcessingError. Agents do not attempt recovery; failures are surfaced to the caller and tracing.
