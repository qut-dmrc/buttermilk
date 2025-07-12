# Core Buttermilk Concepts

Buttermilk is designed to help Humanities, Arts, and Social Sciences (HASS) scholars use complex data and AI tools in their research. To get started, it's helpful to understand a few core concepts. This document explains the key building blocks of Buttermilk.

## Flows

**What are Flows?**

In Buttermilk, a **Flow** represents a complete research or data processing pipeline. Think of it as a recipe that outlines a series of steps to achieve a specific outcome, like collecting data from a particular source, analyzing it with an AI model, and then saving the results. Flows are designed to be repeatable and versionable, allowing you to track exactly how your results were generated.

**How are Flows Defined?**

Flows are typically defined as a sequence of interconnected **Jobs** (explained below). Each step in a Flow might involve fetching data, transforming it, applying a specific analytical tool (like an AI model or a statistical analysis), and then passing the processed data to the next step.

The structure of a Flow can be simple, like a linear chain of operations, or more complex, involving branching logic or parallel processing.

**Lifecycle of a Flow:**

1.  **Definition**: You define the sequence of operations and the configurations for each step.
2.  **Execution**: You run the Flow, providing it with initial data or parameters.
3.  **Monitoring**: Buttermilk helps you track the progress of the Flow and each Job within it.
4.  **Completion**: The Flow finishes, producing outputs, logs, and potentially new datasets.
5.  **Archiving**: The definition of the Flow, its configuration, and its results can be archived for reproducibility and future reference.

*Example*: A Flow might be designed to:
1.  Fetch a collection of social media posts (using a specific **Agent**).
2.  Clean and preprocess the text data.
3.  Analyze the sentiment of each post using an AI language model (another **Agent**).
4.  Store the posts along with their sentiment scores in a database.


## Records

**What are Records?**

A **Record** in Buttermilk represents a single, indivisible piece of data that your Flow will process. This could be a social media post, a paragraph from a historical document, an image, an audio clip, or any other piece of information relevant to your research.

**Key characteristics of Records:**

*   **Immutability**: Once a Record is created, its core data content should not change. If you need to modify data, you typically create a new Record that represents the transformed version. This ensures data integrity and helps with reproducibility.
*   **Unique Identifier**: Each Record has a unique `record_id` that allows Buttermilk to track it throughout the system and across different Flows and Jobs.
*   **Metadata**: Besides the primary data content, Records can store rich metadata. This could include information about the data's source, creation date, copyright information, or any other contextual details that are important for your analysis.
    *   *Example*: A Record representing an image might contain the image file itself, along with metadata like the date it was taken, the photographer, and the resolution. (See `buttermilk._core.types.MediaObj` for how media is handled).
*   **Data Types**: Records can hold various types of data, including text, numbers, and references to binary data like images or audio files.

A collection of Records forms a **Dataset**, which is often the input or output of a Flow.

## Agents

**What are Agents?**

**Agents** are the workhorses of Buttermilk. They are specialized components responsible for performing specific tasks on your data. Each Agent is designed to interact with a particular tool, API, or model.

**Types of Agents:**

Buttermilk can support a variety of Agents, for example:
*   **LLMAgents**: Interact with Large Language Models (LLMs) like GPT, Claude, or Gemini to perform tasks like text generation, classification, summarization, or question answering. (See `buttermilk._core.llm.LLMAgent`).
*   **DataCollectionAgents**: Fetch data from various sources, such as social media platforms, web APIs, or databases.
*   **PreprocessingAgents**: Clean, transform, or reformat data to prepare it for analysis.
*   **AnalysisAgents**: Perform specific analytical tasks, such as statistical calculations, network analysis, or image recognition.

**How are Agents Configured?**

Each Agent has its own set of configuration options that you can set when defining a Job or a Flow. This allows you to customize the Agent's behavior for your specific needs. For example, when using an LLMAgent, you might configure:
*   The specific model to use (e.g., "gpt-4-turbo" or "claude-3-opus").
*   Parameters influencing the model's output, like `temperature` or `max_tokens`.
*   The prompt template to be used for the task.

(See `buttermilk._core.config.AgentConfig` for examples of agent configuration structures).

## Orchestrators

**What are Orchestrators?**

**Orchestrators** are responsible for managing and executing Flows. They take a defined Flow, a dataset of Records, and then coordinate the execution of Jobs by the appropriate Agents.

**Role of Orchestrators:**

*   **Flow Execution**: They manage the sequence of operations in a Flow, ensuring that Jobs are run in the correct order.
*   **Job Management**: They create and dispatch Jobs to Agents.
*   **Data Flow**: They handle the passing of data (Records and results) between different steps in a Flow.
*   **Error Handling**: They can manage retries or alternative paths if a Job fails.
*   **Logging and Tracing**: They often work with logging systems to keep a detailed record of the Flow's execution.

(See `buttermilk._core.orchestrator.Orchestrator` for an example of an orchestrator implementation).

Essentially, the Orchestrator is the conductor of your research pipeline, making sure all the different parts work together smoothly.

## Configuration with Hydra

Buttermilk uses a powerful library called [Hydra](https://hydra.cc/) for managing configurations. This is particularly helpful for HASS scholars because it allows for flexible and organized experimentation.

**How Hydra is used:**

*   **YAML Files**: Configurations are typically defined in human-readable YAML files (found in the `conf/` directory). This makes it easy to see and change settings without modifying code.
*   **Modularity**: Hydra allows you to break down configurations into smaller, reusable pieces. For example, you might have separate configuration files for different datasets, AI models, or output settings.
*   **Command-Line Overrides**: You can easily change configuration parameters from the command line when you run a Flow. This is great for trying out different settings (e.g., testing different AI models or prompt variations) without editing files each time.
*   **Reproducibility**: Hydra helps ensure that your experiments are reproducible by keeping a clear record of the configurations used for each run.

Understanding how to view and modify these configuration files will be key to tailoring Buttermilk to your specific research questions.

## Data Handling

**How is data passed between Agents and steps?**

Data in Buttermilk typically flows in the form of **Records**.
1.  An Orchestrator takes a **Dataset** (a collection of Records) as input for a Flow.
2.  For each step in the Flow, the Orchestrator creates **Jobs**. Each Job is usually associated with a single Record.
3.  The Job (containing the Record and task configuration) is sent to an **Agent**.
4.  The Agent processes the Record's data and produces an output.
5.  This output can then be:
    *   Stored back into an updated version of the Record (or a new field within it).
    *   Used to create a new Record.
    *   Passed on to the next Agent in the Flow.

Buttermilk aims to make this data flow traceable, so you can always see how a piece of data was transformed at each step. Message contracts (like `buttermilk._core.contract.FlowMessage`) define the structure of data as it moves through the system.

## Extensibility

Buttermilk is designed to be extensible. While it provides core components, researchers can add new functionalities:

*   **New Agents**: If you need to integrate a new tool, API, or model, you can create a new Agent. This involves writing Python code that conforms to Buttermilk's Agent structure (see `buttermilk._core.agent.Agent`).
*   **New Data Types**: If your research involves specialized data formats, you can define new data types.
*   **Custom Flows**: You can combine existing Agents in novel ways to create new Flows tailored to your research questions.

This extensibility allows Buttermilk to adapt to the diverse and evolving needs of HASS research.
