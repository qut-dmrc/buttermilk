#!/usr/bin/env python3
"""Mock Orchestrator for testing Buttermilk frontend components.

This extends the base Orchestrator class and simulates agent behavior by generating 
fake messages of all types. It can be used for frontend testing without requiring
actual agent execution.
"""

import asyncio
import random
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import shortuuid
from pydantic import Field, PrivateAttr

from buttermilk._core.config import AgentConfig
from buttermilk._core.contract import (
    AgentInput,
    AgentTrace,
    ErrorEvent,
    ManagerMessage,
    TaskProgressUpdate,
    ToolOutput,
    UIMessage,
)
from buttermilk._core.log import logger
from buttermilk._core.exceptions import FatalError
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import RunRequest
from buttermilk.agents.differences import Differences
from buttermilk.agents.evaluators.scorer import QualScoreCRA
from buttermilk.agents.ui.console import QualResults
from buttermilk.api.services.message_service import MessageService


from buttermilk._core.contract import FlowEvent, TaskProcessingStarted, TaskProcessingComplete
from buttermilk._core.types import Record
from buttermilk.agents.judge import JudgeReasons
from buttermilk.agents.differences import Differences, Divergence, Position, Expert

class MockTerminationHandler:
    """Simplified termination handler that allows simulation control"""

    def __init__(self) -> None:
        self._should_terminate = False
        self._termination_value = None

    def request_termination(self):
        """Signal that the simulation should terminate"""
        self._should_terminate = True

    @property
    def has_terminated(self) -> bool:
        return self._termination_value is not None or self._should_terminate


class MockOrchestrator(Orchestrator):
    """Mock version of Orchestrator that generates simulated agent messages.
    
    This orchestrator bypasses the actual agent processing and instead generates
    fake messages that mimic a real orchestrator run. It's useful for frontend
    testing without needing the actual LLM calls or data processing.
    """

    # Flag to control whether to generate random messages or follow a fixed flow
    random_generation: bool = Field(default=True, description="Generate random messages if True, follow fixed flow if False")

    # Controls frequency of message generation
    message_interval: float = Field(default=2.0, description="Seconds between random message generation")

    # Private state
    _termination_handler: MockTerminationHandler | None = PrivateAttr(default_factory=lambda: None)
    _simulation_task: asyncio.Task | None = PrivateAttr(default_factory=lambda: None)
    _client_websocket: Any = PrivateAttr(default_factory=lambda: None)
    _topic_type: str = PrivateAttr(default_factory=lambda: f"mock-flow-{shortuuid.uuid()[:8]}")

    _agent_ids: list[str] = PrivateAttr(default_factory=lambda: [f"agent-{i+1}" for i in range(5)]) # Use simpler, consistent IDs

    async def _setup(self, request: RunRequest) -> None:
        """Setup the mock orchestrator environment"""
        msg = f"Setting up MockOrchestrator for topic: {self._topic_type}"
        logger.info(msg)

        # Create our custom termination handler
        self._termination_handler = MockTerminationHandler()

        # Set up simplified participants dict
        self._participants = {
            "CONDUCTOR": "Orchestrates the workflow and decides the sequence of steps",
            "ASSISTANT": "Provides primary responses and analysis",
            "RESEARCHER": "Collects and analyzes relevant information",
            "ANALYST": "Performs quantitative analysis of data",
            "CRITIC": "Reviews and evaluates outputs from other agents",
            "MANAGER": "User interface for interaction and feedback",
        }

        # Send initial welcome message
        await self._publish_message(
            ManagerMessage(content=msg),
        )

        # Handle any initial data if provided
        if request:
            await self._fetch_initial_records(request)

    async def _cleanup(self) -> None:
        """Clean up resources"""
        logger.debug("Cleaning up MockOrchestrator...")

        await asyncio.sleep(0.1)  # Small delay for cleanup

    async def _run(self, request: RunRequest | None = None, **ignored_tracing_kwargs) -> None:  # noqa
        """Override run to generate fake messages instead of using real agents"""
        try:
            # Setup the mock environment
            await self._setup(request or RunRequest(flow="mock_flow", ui_type="web"))

            # Set callback on request if provided
            if request and hasattr(request, "callback_to_ui") and request.callback_to_ui:
                self._client_websocket = request.callback_to_ui

            # Start generating messages
            if self.random_generation:
                # Random, continuous message generation
                self._simulation_task = asyncio.create_task(self._generate_random_messages())
            else:
                # Structured flow simulation
                self._simulation_task = asyncio.create_task(self._simulate_flow())

            # Wait for termination signal
            while True:
                if self._termination_handler and self._termination_handler.has_terminated:
                    logger.info("Termination message received.")
                    break
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Flow terminated by user.")
        except FatalError as e:
            logger.error(f"Fatal error: {e}")
        except Exception as e:
            logger.exception(f"Unhandled exception: {e}")
        finally:
            # Cancel simulation if running
            if self._simulation_task and not self._simulation_task.done():
                self._simulation_task.cancel()
                try:
                    await self._simulation_task
                except asyncio.CancelledError:
                    pass

            await self._cleanup()

    # --- Message Generation Methods ---

    async def _generate_random_messages(self):
        """Generate random messages of various types periodically"""
        try:
            while True:
                # Choose a message type randomly with weights
                message_generators = [
                    (self._generate_agent_trace, 3),       # Higher weight for agent traces
                    (self._generate_progress_update, 2),   # Medium weight for progress updates
                    (self._generate_error_event, 1),       # Lower weight for errors
                    (self._generate_ui_message, 2),        # Medium weight for UI messages
                    (self._generate_tool_output, 2),       # Medium weight for tool outputs
                    (self._generate_record, 3),            # Higher weight for records
                ]
                
                # Weighted random selection
                generators, weights = zip(*message_generators)
                message_generator = random.choices(generators, weights=weights, k=1)[0]

                # Generate and publish the message
                message = message_generator()
                await self._publish_message(message)

                # Wait a bit before sending the next message
                await asyncio.sleep(random.uniform(1.0, self.message_interval))
        except asyncio.CancelledError:
            logger.debug("Random message generation cancelled")
            raise

    async def _simulate_flow(self):
        """Simulate a structured flow with sequential messages"""
        try:
            # Step 1: Start the flow with a FlowEvent
            start_event = FlowEvent(
                content="Flow xyz started"
            )
            await self._publish_message(start_event)
            await asyncio.sleep(0.5)
            
            # Publish a TaskProcessingStarted event
            task_start = TaskProcessingStarted(
                agent_id=random.choice(self._agent_ids),
                role="JUDGE"
            )
            await self._publish_message(task_start)
            await asyncio.sleep(0.5)
            
            # Continue with typical progress update
            start_msg = self._generate_progress_update(
                source="ORCHESTRATOR",
                role="CONDUCTOR",
                step_name="flow_initialization",
                status="started",
                message="Starting workflow execution",
                total_steps=5,
                current_step=0,
            )
            await self._publish_message(start_msg)
            await asyncio.sleep(1)

            # Step 2: Research phase
            for i in range(3):
                progress = self._generate_progress_update(
                    source="RESEARCHER",
                    role="RESEARCHER",
                    step_name="data_collection",
                    status="in_progress",
                    message=f"Collecting data sources ({i + 1}/3)",
                    total_steps=5,
                    current_step=1,
                )
                await self._publish_message(progress)

                # Add a sample record for each research iteration
                record = self._generate_record(
                    record_id=f"research-data-{i + 1}",
                    content=f"Research data sample #{i + 1}: Analysis of trending topics in AI research",
                    metadata={
                        "source": "academic_database",
                        "relevance_score": round(random.uniform(0.65, 0.95), 2),
                        "publication_date": "2025-03-15",
                        "category": "research_paper",
                    },
                )
                await self._publish_message(record)
                await asyncio.sleep(0.8)

            research_result = self._generate_agent_trace(
                agent_id=random.choice(self._agent_ids),
                outputs="Collected 5 research papers and 3 dataset references",
                metadata={"data_sources": 8, "processing_time": "2.3s"},
            )
            await self._publish_message(research_result)
            await asyncio.sleep(1)

            # Step 3: Analysis phase
            progress = self._generate_progress_update(
                source="ANALYST",
                role="ANALYST",
                step_name="data_analysis",
                status="in_progress",
                message="Analyzing research data",
                total_steps=5,
                current_step=2,
            )
            await self._publish_message(progress)
            await asyncio.sleep(1.5)

            tool_output = self._generate_tool_output(
                function_name="analyze_dataset",
                content="Analysis complete: found 3 key trends in the data",
            )
            await self._publish_message(tool_output)
            await asyncio.sleep(0.8)

            # Use a QualResults instance in the analyst output
            qual_results = QualResults(
                assessments=[
                    QualScoreCRA(correct=True, feedback="Document contains comprehensive research with strong methodological approach"),
                    QualScoreCRA(correct=True, feedback="Data visualization effectively communicates key trends"),
                    QualScoreCRA(correct=True, feedback="Statistical analysis is thorough and appropriate for the dataset")
                ],
                assessed_agent_id=random.choice(self._agent_ids), # Use agent_ids list
                assessed_call_id=str(uuid.uuid4()), # Keep mock call ID
            )
            
            analysis_result = self._generate_agent_trace(
                agent_id=random.choice(self._agent_ids), # Use agent_ids list
                outputs=qual_results,
                metadata={"confidence": 0.87, "model": "claude-3"},
            )
            await self._publish_message(analysis_result)
            await asyncio.sleep(1)

            # Add critic analysis with Differences output
            # Generate random experts using the new method
            experts_pos1 = [self._generate_expert() for _ in range(random.randint(1, 3))]
            experts_pos2 = [self._generate_expert() for _ in range(random.randint(1, 3))]
            experts_pos3 = [self._generate_expert() for _ in range(random.randint(1, 3))]

            differences = Differences(
                conclusion="Revise the document to address these points before finalization",
                divergences=[Divergence(topic="Methodology",positions=[
                    Position(experts=experts_pos1, position="The conclusion section could be strengthened with additional examples"),
                    Position(experts=experts_pos2, position="Some statistical methods might benefit from more detailed explanation"),
                    Position(experts=experts_pos3, position="Consider addressing alternative interpretations of the findings")
                ])],
            )
            
            critic_result = self._generate_agent_trace(
                agent_id=random.choice(self._agent_ids), # Use agent_ids list
                outputs=differences,
                metadata={"analysis_depth": "detailed", "model": "gpt-4"},
            )
            await self._publish_message(critic_result)
            await asyncio.sleep(1)

            # Add judge evaluation with JudgeReasons
            judge_reasons = JudgeReasons(
                conclusion="The content adheres to guidelines with minor concerns about citation formatting.",
                prediction=random.choice([True, False]), # Randomize prediction
                uncertainty=random.choice(["low", "medium", "high"]), # Randomize uncertainty
                reasons=[
                    "Content is factually accurate and well-supported",
                    "No harmful, misleading, or inappropriate material detected",
                    "Citation style is inconsistent but all sources are properly credited"
                ]
            )
            
            judge_result = self._generate_agent_trace(
                agent_id=random.choice(self._agent_ids), # Use agent_ids list
                outputs=judge_reasons,
                metadata={"evaluation_criteria": "academic_standards", "model": "claude-3"},
            )
            await self._publish_message(judge_result)
            await asyncio.sleep(1)

            # Step 4: User interaction with UIMessage
            user_request = UIMessage(
                content="How should I proceed with this analysis?",
                options=["Detailed analysis", "Generate summary", "Request revisions"],
            )
            await self._publish_message(user_request)
            await asyncio.sleep(2)

            # Step 5: Final response
            progress = self._generate_progress_update(
                source="ASSISTANT",
                role="ASSISTANT",
                step_name="response_generation",
                status="in_progress",
                message="Generating final response",
                total_steps=5,
                current_step=4,
            )
            await self._publish_message(progress)
            await asyncio.sleep(1.5)

            final_result = self._generate_agent_trace(
                agent_id="ASSISTANT",
                outputs="Based on the analysis, here are the key recommendations: 1) Implement strategy X, 2) Modify approach Y, 3) Consider alternative Z",
                metadata={"tokens": 856, "model": "gpt-4"},
            )
            await self._publish_message(final_result)
            await asyncio.sleep(0.8)

            # Add a TaskProcessingComplete event
            task_complete = TaskProcessingComplete(
                agent_id=random.choice(self._agent_ids), role="ASSISTANT",
            )
            await self._publish_message(task_complete)
            await asyncio.sleep(0.5)


            # Signal termination after simulation is complete
            if self._termination_handler:
                self._termination_handler.request_termination()

        except asyncio.CancelledError:
            logger.debug("Flow simulation cancelled")
            raise

    # --- Mock Message Generator Methods ---

    def _generate_expert(self) -> Expert:
        """Generate a fake Expert object using a consistent agent_id."""
        # Use the aliased ExpertType from buttermilk._core.types
        agent_id = random.choice(self._agent_ids)
        return Expert(name=f"Expert {agent_id}", answer_id=agent_id) # Use agent_id for answer_id for consistency

    def _generate_agent_trace(self, agent_id=None, outputs=None, metadata=None, parent_call_id=None, tool_code=None) -> AgentTrace:
        """Generate a fake agent trace"""
        from buttermilk._core.log import logger  # Import logger locally if needed
        from buttermilk._core.contract import AgentInput, AgentTrace, AgentConfig
        from buttermilk._core.types import Record # Import Record if used in outputs
        from buttermilk.agents.judge import JudgeReasons # Import JudgeReasons if used in outputs
        from buttermilk.agents.evaluators.scorer import QualResults, QualScoreCRA # Corrected import and added QualScoreCRA
        from buttermilk.agents.differences import Differences, Divergence, Position, Expert # Corrected import and added nested models

        if agent_id is None:
            agent_types = ["ASSISTANT", "RESEARCHER", "JUDGE", "ANALYST", "CRITIC"]
            agent_id = random.choice(agent_types)

        if outputs is None:
            # Use actual structured message types for outputs
            if agent_id == "JUDGE":
                outputs = JudgeReasons(
                    conclusion="The content appears to comply with policies with minor concerns.",
                    prediction=random.choice([True, False]),
                    uncertainty=random.choice(["low", "medium", "high"]),
                    reasons=[
                        "The content doesn't contain explicit harmful instructions",
                        "No obvious hate speech or discriminatory language was detected",
                        "The text remains within appropriate boundaries for general audiences"
                    ]
                )
            elif agent_id == "ANALYST":
                # QualResults requires assessed_agent_id, assessed_call_id, and assessments (list of QualScoreCRA)
                outputs = QualResults(
                    assessed_agent_id=random.choice(self._agent_ids), # Use agent_ids list
                    assessed_call_id=str(uuid.uuid4()), # Mock the call ID being assessed
                    assessments=[
                        QualScoreCRA(correct=random.choice([True, False]), feedback="Criterion 1 assessment."),
                        QualScoreCRA(correct=random.choice([True, False]), feedback="Criterion 2 assessment."),
                        QualScoreCRA(correct=random.choice([True, False]), feedback="Criterion 3 assessment."),
                    ],
                    # score and summary are computed properties, no need to set here
                )
            elif agent_id == "CRITIC":
                # Differences requires conclusion and divergences (list of Divergence)
                # Divergence requires topic and positions (list of Position)
                # Position requires experts (list of Expert) and position (string)
                # Expert requires name and answer_id
                # Generate random experts using the new method
                experts_pos1 = [self._generate_expert() for _ in range(random.randint(1, 3))]
                experts_pos2 = [self._generate_expert() for _ in range(random.randint(1, 3))]
                experts_pos3 = [self._generate_expert() for _ in range(random.randint(1, 3))]

                outputs = Differences(
                    conclusion="Overall, there are some notable differences in the expert opinions.",
                    divergences=[
                        Divergence(
                            topic="Key Findings Interpretation",
                            positions=[
                                Position(
                                    experts=experts_pos1,
                                    position="Interpretation focuses on positive trends."
                                ),
                                Position(
                                    experts=experts_pos2,
                                    position="Interpretation highlights potential risks."
                                ),
                            ]
                        ),
                        Divergence(
                            topic="Methodology Validity",
                            positions=[
                                Position(
                                    experts=experts_pos3,
                                    position="Methodology is considered sound."
                                ),
                                Position(
                                    experts=[self._generate_expert()], # Single expert position
                                    position="Concerns raised about sample size."
                                ),
                            ]
                        ),
                    ]
                )
            else:
                outputs_list = [
                    "I've analyzed the data and found several key insights...",
                    "The model performance indicates a 87% accuracy rate...",
                    "Based on the document analysis, the main themes are...",
                    "The search results show 15 relevant articles on this topic...",
                    "I've generated a comprehensive response to the query...",
                ]
                outputs = random.choice(outputs_list)

        if metadata is None:
            metadata = {
                "duration_ms": random.randint(500, 3000),
                "tokens": random.randint(100, 1000),
                "model": random.choice(["gpt-4", "claude-3", "gemini-pro", "llama-3"]),
            }

        # Create an empty AgentInput for the trace (or populate with mock data if needed)
        agent_input = AgentInput()

        # Create a proper AgentConfig instance
        agent_info = AgentConfig(
            agent_id=agent_id,
            description=f"{agent_id} agent for processing tasks",
            parameters={}, # Add mock parameters if needed
        )

        # Optionally generate parent_call_id and tool_code
        generated_parent_call_id = parent_call_id if parent_call_id is not None else (str(uuid.uuid4()) if random.random() < 0.3 else None) # 30% chance of having a parent
        generated_tool_code = tool_code if tool_code is not None else ("print('hello world')" if random.random() < 0.1 else None) # 10% chance of having tool code

        return AgentTrace(
            agent_id=agent_id,
            call_id=str(uuid.uuid4()),
            outputs=outputs,
            metadata=metadata,
            timestamp=datetime.now(UTC),
            agent_info=agent_info,
            session_id=self.trace_id,
            inputs=agent_input,
            parent_call_id=generated_parent_call_id,
        )

    def _generate_progress_update(self, source=None, role=None, step_name=None,
                                status=None, message=None, total_steps=None,
                                current_step=None) -> TaskProgressUpdate | FlowEvent | TaskProcessingComplete |TaskProcessingStarted:
        """Generate a fake progress update or flow event"""
        from buttermilk._core.contract import FlowEvent, TaskProcessingComplete, TaskProcessingStarted, TaskProgressUpdate

        # Sometimes generate flow events instead of progress updates
        if random.choice([True, False, False]):  # 1/3 chance for flow events
            event_types = [TaskProcessingStarted, TaskProcessingComplete, FlowEvent]
            event_class = random.choice(event_types)

            if event_class == TaskProcessingStarted:
                # TaskProcessingStarted requires agent_id, role, task_index
                return TaskProcessingStarted(
                    agent_id=source if source else random.choice(self._agent_ids), # Use agent_ids list
                    role=role if role else random.choice(["ASSISTANT", "RESEARCHER", "ANALYST"]),
                    task_index=random.randint(0, 5), # Mock task index
                    # task_id and flow_id are not part of TaskProcessingStarted based on contract.py
                    # inputs and timestamp are not part of TaskProcessingStarted based on contract.py
                )
            elif event_class == TaskProcessingComplete:
                # TaskProcessingComplete requires agent_id, role, task_index, more_tasks_remain, is_error
                return TaskProcessingComplete(
                    agent_id=source if source else random.choice(self._agent_ids), # Use agent_ids list
                    role=role if role else random.choice(["ASSISTANT", "RESEARCHER", "ANALYST"]),
                    task_index=random.randint(0, 5), # Mock task index
                    more_tasks_remain=random.choice([True, False]),
                    is_error=random.choice([True, False]),
                    # task_id, flow_id, result, and timestamp are not part of TaskProcessingComplete based on contract.py
                )
            else:  # FlowEvent
                # FlowEvent requires source and content
                event_type = random.choice(["flow_started", "flow_completed", "agent_selected", "error_occurred"])
                
                generated_source = source if source else "ORCHESTRATOR" # Default source for FlowEvent
                generated_content = f"Flow event: {event_type}" # Default content

                details = {} # Details are not a required parameter for FlowEvent

                # Adding details to content for better mock representation
                if event_type == "flow_started":
                    details.update({
                        "flow_name": "mock_flow",
                        "parameters": {"initial_param": random.choice(["A", "B", "C"])}
                    })
                    generated_content = f"Flow started: {details.get('flow_name')}"
                elif event_type == "flow_completed":
                    details.update({
                        "summary": "Mock flow completed successfully.",
                        "duration_ms": random.randint(1000, 10000)
                    })
                    generated_content = f"Flow completed. Summary: {details.get('summary')}"
                elif event_type == "agent_selected":
                    details.update({
                        "agent_id": random.choice(self._agent_ids), # Use agent_ids list
                        "task_description": "Processing a mock task."
                    })
                    generated_content = f"Agent selected: {details.get('agent_id')} for task: {details.get('task_description')}"
                elif event_type == "error_occurred":
                    details.update({
                        "error_message": "A simulated error occurred.",
                        "error_type": random.choice(["ValueError", "RuntimeError", "TimeoutError"])
                    })
                    generated_content = f"Error occurred: {details.get('error_type')} - {details.get('error_message')}"

                # Note: FlowEvent in contract.py only has 'source' and 'content'.
                # The 'details' dictionary was used in the previous attempt but is not part of the model.
                # I will include the details information within the 'content' string for better mock data.
                
                return FlowEvent(
                    source=generated_source,
                    content=generated_content,
                    # timestamp is not part of FlowEvent based on contract.py
                )

        # Otherwise generate a regular progress update
        if source is None:
            roles = ["ASSISTANT", "RESEARCHER", "JUDGE", "ANALYST", "CRITIC"]
            source = random.choice(roles)

        if role is None:
            role = source

        if step_name is None:
            steps = ["preparing", "analyzing", "generating", "validating", "completing"]
            step_name = random.choice(steps)

        if status is None:
            statuses = ["started", "in_progress", "completed", "error"]
            status = random.choice(statuses)

        if total_steps is None:
            total_steps = random.randint(5, 10)

        if current_step is None:
            current_step = random.randint(0, total_steps)

        if message is None:
            messages = [
                f"Processing step {current_step} of {total_steps}...",
                f"Analyzing data for {random.choice(['sentiment', 'key entities', 'topics'])}...",
                f"Generating response using {random.choice(['GPT-4', 'Claude', 'Gemini'])}...",
                "Evaluating response quality...",
                "Applying formatting to results...",
            ]
            message = random.choice(messages)

        # TaskProgressUpdate requires source, role, step_name, status, message, total_steps, current_step, timestamp, waiting_on
        return TaskProgressUpdate(
            source=source,
            role=role,
            step_name=step_name,
            status=status,
            message=message,
            total_steps=total_steps,
            current_step=current_step,
            timestamp=datetime.now(UTC),
            waiting_on={}, # Mock empty waiting_on for simplicity
        )

    def _generate_ui_message(self, content=None, options=None) -> UIMessage:
        """Generate a fake UI message"""
        from buttermilk._core.contract import UIMessage

        if content is None:
            questions = [
                "How should I proceed with this analysis?",
                "Which of these options would you prefer?",
                "Should I continue with the current approach?",
                "Do you want more detailed results on this topic?",
                "Would you like to see alternative approaches?",
            ]
            content = random.choice(questions)

        if options is None:
            has_options = random.choice([True, False])
            if has_options:
                num_options = random.randint(2, 4)
                options = [f"Option {i + 1}" for i in range(num_options)]

        # UIMessage requires content, options, show_continue, allow_free_text, message_id
        # response and response_timestamp are not part of UIMessage based on contract.py

        return UIMessage(
            content=content,
            options=options,
        )

    def _generate_record(self, record_id=None, content=None, metadata=None) -> Any:
        """Generate a fake record message"""
        # Import Record locally to avoid circular imports
        from buttermilk._core.types import Record

        if record_id is None:
            record_id = f"record-{shortuuid.uuid()[:8]}"

        if content is None:
            content_options = [
                "This is a sample document about machine learning applications in healthcare.",
                "The quarterly financial report shows a 15% increase in revenue compared to last year.",
                "User feedback survey indicates high satisfaction with the new interface design.",
                "Research paper: 'Advances in Natural Language Processing for Document Analysis'",
                "Customer support transcript regarding product installation issues.",
            ]
            content = random.choice(content_options)

        if metadata is None:
            metadata = {
                "source": random.choice(["database", "api", "user_upload", "web_scrape"]),
                "timestamp": datetime.now(UTC).isoformat(),
                "category": random.choice(["document", "report", "article", "data", "message"]),
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "word_count": random.randint(100, 5000),
            }

        # Occasionally add a title to the metadata
        if random.choice([True, False]):
            titles = [
                "Quarterly Report Q1 2025",
                "User Feedback Analysis",
                "ML Applications in Healthcare",
                "Customer Support Summary",
                "Research Findings",
            ]
            metadata["title"] = random.choice(titles)

        # Record requires record_id, content, metadata, mime, update_type, session_id
        # parent_record_id, tool_code, agent_id, and call_id are not part of Record based on contract.py

        return Record(
            record_id=record_id,
            content=content,
            metadata=metadata,
            mime=random.choice(["text/plain", "text/markdown", "text/html", "application/json"]),
        )


    def _generate_error_event(self, error_type=None, message=None, details=None) -> ErrorEvent:
        """Generate a fake error event message"""
        from buttermilk._core.contract import ErrorEvent

        if error_type is None:
            error_type = random.choice(["ValueError", "RuntimeError", "TimeoutError", "APIError"])

        if message is None:
            messages = [
                f"An error occurred during {random.choice(['data processing', 'API call', 'analysis'])}.",
                f"Failed to complete task due to {error_type}.",
                "Unexpected error encountered.",
            ]
            message = random.choice(messages)

        # ErrorEvent requires source and content
        # error_type, details, timestamp, agent_id, call_id are not part of ErrorEvent based on contract.py

        generated_source = random.choice(["ASSISTANT", "RESEARCHER", "JUDGE", "ANALYST", "CRITIC", "ORCHESTRATOR"])
        generated_content = f"ERROR: {error_type} - {message}" # Combine type and message into content

        return ErrorEvent(
            source=generated_source,
            content=generated_content,
            # error_type=error_type, # Removed
            # details=details, # Removed
            # timestamp=datetime.now(UTC), # Removed
            # agent_id=generated_agent_id, # Removed
            # call_id=generated_call_id, # Removed
        )

    async def _fetch_initial_records(self, request: RunRequest):
        """Simulate fetching initial records based on the run request."""
        if request and request.record_id:
            logger.info(f"Simulating fetching initial record: {request.record_id}")
            # Generate a mock record based on the requested ID
            mock_record = self._generate_record(
                record_id=request.record_id,
                content=f"Mock content for record {request.record_id}. This is a simulated document.",
                metadata={
                    "source": "simulated_fetch",
                    "request_params": request.parameters,
                    "flow_name": request.flow,
                }
            )
            await self._publish_message(mock_record)
            await asyncio.sleep(0.5) # Simulate fetch delay

        if request and request.record_id:
            logger.info(f"Simulating fetching initial records: {request.record_id}")
            mock_record = self._generate_record(
                record_id=request.record_id,
                content=f"Mock content for record {request.record_id}.",
                metadata={
                    "source": "simulated_fetch_list",
                    "request_params": request.parameters,
                    "flow_name": request.flow,
                }
            )
            await self._publish_message(mock_record)
            await asyncio.sleep(0.3) # Simulate fetch delay for each record

    def make_publish_callback(self) -> Callable:
        """Creates an asynchronous callback function for the UI to use.

        Returns:
            An async callback function that takes a message and publishes it.

        """
        async def publish_callback(message) -> None:
            # Just forward to our _publish_message method
            await self._publish_message(message)

        return publish_callback

    async def _publish_message(self, message):
        """Publish a message directly to the websocket if available"""
        # Log the message
        formatted_message = MessageService.format_message_for_client(message)
        if formatted_message is None:
            logger.warning(f"Message not formatted: {message}")
            return
        log_content = str(formatted_message.preview)
        if len(log_content) > 100:
            log_content = log_content[:97] + "..."
        logger.debug(f"Mock message: {message.__class__.__name__} - {log_content}")

        # Send directly to websocket if we have a client_websocket
        if self._client_websocket:
            try:
                await self._client_websocket(formatted_message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
