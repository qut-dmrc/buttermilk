# LLM Agent Call Stack Overview

Here is a high-level overview of the call stack, from the initial agent invocation to the final response processing:

    1 sequenceDiagram
    2     participant User
    3     participant Agent.invoke()
    4     participant LLMAgent._process()
    5     participant AutoGenWrapper.call_chat()
    6     participant AutoGenWrapper.create()
    7     participant LLM_API
    8     participant AutoGenWrapper._execute_tools()
    9 
   10     User->>Agent.invoke(): Start with AgentInput
   11     Agent.invoke()->>LLMAgent._process(): Calls subclass logic
   12     LLMAgent._process()->>AutoGenWrapper.call_chat(): Hands off to LLM wrapper
   13     
   14     Note over AutoGenWrapper.call_chat(): Manages the conversation turn
   15     AutoGenWrapper.call_chat()->>AutoGenWrapper.create(): Makes the first LLM call
   16     AutoGenWrapper.create()->>LLM_API: Sends messages
   17     LLM_API-->>AutoGenWrapper.create(): Returns raw response
   18     
   19     alt LLM requests tool calls
   20         AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns FunctionCall(s)
   21         AutoGenWrapper.call_chat()->>AutoGenWrapper._execute_tools(): Executes tools
   22         AutoGenWrapper._execute_tools()-->>AutoGenWrapper.call_chat(): Returns tool results
   23         
   24         Note over AutoGenWrapper.call_chat(): Sends tool results back to LLM
   25         AutoGenWrapper.call_chat()->>AutoGenWrapper.create(): Makes second LLM call
   26         AutoGenWrapper.create()->>LLM_API: Sends messages + tool results
   27         LLM_API-->>AutoGenWrapper.create(): Returns final text response
   28         AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns CreateResult/ModelOutput
   29     else LLM returns text/JSON directly
   30         AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns CreateResult/ModelOutput
   31     end
   32 
   33     alt Error during processing
   34          AutoGenWrapper.create()-->>AutoGenWrapper.call_chat(): Returns ErrorResult
   35     end
   36 
   37     AutoGenWrapper.call_chat()-->>LLMAgent._process(): Returns final result
   38     LLMAgent._process()-->>Agent.invoke(): Returns AgentOutput
   39     Agent.invoke()-->>User: Returns AgentTrace


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
       4. After _process() returns, it wraps the result in an AgentTrace object for logging and returns it.

  2. LLMAgent._process() - The Agent's Core Logic
   * File: buttermilk/agents/llm.py
   * Purpose: Implements the specific logic for an LLM-based agent.
   * Flow:
       1. Template Filling: It calls _fill_template() to render the prompt, creating a list of LLMMessage objects ready for the API.
       2. LLM Call: It calls _call_llm(), which is a helper that gets the correct AutoGenWrapper client and passes the request to it.
       3. Response Handling: It receives a result from the LLM call (CreateResult, ModelOutput, or ErrorResult).
           * If the result is a ModelOutput, it extracts the parsed_object.
           * If it's a plain CreateResult, it uses the content.
           * If it's an ErrorResult, it logs the error.
       4. Output Creation: It packages the final content (string, Pydantic object, or error) into an AgentOutput object and returns it up
          the chain.

  3. LLMAgent._call_llm() -> AutoGenWrapper.call_chat() - Orchestrating the Chat
   * Files: buttermilk/agents/llm.py -> buttermilk/_core/llms.py
   * Purpose: call_chat() manages a full "turn" of conversation, which might involve multiple back-and-forths with the LLM if tools are
     used.
   * Flow:
       1. It makes an initial call to self.create() with the user's messages and the list of available tools.
       2. Scenario A: The LLM returns tool calls.
           * The create() method returns a CreateResult where the content is a list of FunctionCall objects.
           * call_chat() then calls _execute_tools() to run the functions.
           * The tool results are added to the message history.
           * call_chat() calls self.create() a second time with the updated history to get a final, synthesized response from the LLM.
       3. Scenario B: The LLM returns text or JSON directly.
           * create() returns a CreateResult or ModelOutput. call_chat() simply passes this result back up to LLMAgent.
       4. Scenario C: An error occurs.
           * If create() or _execute_tools() runs into a problem, it returns an ErrorResult, which call_chat() passes up.

  4. AutoGenWrapper.create() - The Atomic LLM Call
   * File: buttermilk/_core/llms.py
   * Purpose: This method handles a single API call to the LLM, including structured output and error handling.
   * Flow:
       1. Argument Preparation: It prepares the arguments for the underlying client. If you provide a Pydantic schema, it will either
          configure the json_output parameter (for models that support it) or create a "fake tool" to force the model to return the desired
          structure.
       2. API Call: It executes the call using _execute_with_retry(). API-level errors (like connection issues) are handled here.
       3. Result Processing: After a successful API call, it processes the response.
           * If the response is invalid (e.g., empty), it returns an ErrorResult.
           * If a schema was requested, it calls _parse_structured_output() to parse the LLM's string response into a Pydantic object. If
             parsing succeeds, it returns a ModelOutput. If parsing fails, it returns an ErrorResult.
           * Otherwise, it returns the standard CreateResult.

  5. AutoGenWrapper._call_tool() & _execute_tools() - Function Execution
   * File: buttermilk/_core/llms.py
   * Purpose: These methods handle the execution of tools requested by the LLM.
   * Flow:
       1. _execute_tools() receives the list of FunctionCall objects.
       2. It iterates through them and calls _call_tool() for each one, running them concurrently.
       3. _call_tool() finds the corresponding Tool object, executes its run_json method, and wraps the output in a FunctionExecutionResult
          object, which is what the LLM expects to see in the message history.

  How ErrorResult and ModelOutput Fit In

  Your introduction of these custom result types is handled correctly within this flow:

   * `ModelOutput` is a success-path object. It's created at the lowest level (create()) and passed all the way up to LLMAgent._process(),
     which knows how to unwrap the parsed_object. This is a clean way to pass structured data.
   * `ErrorResult` is a failure-path object. It's created in create() or call_chat() when an error occurs after a successful API call
     (e.g., parsing fails, a tool fails). By returning this object instead of raising an exception, the agent can gracefully handle the
     failure, log it, and report it in the final AgentOutput without crashing the entire process.

  The flow is complex, but it correctly separates concerns and handles the different success and failure paths that can occur during an
  LLM interaction with tools.