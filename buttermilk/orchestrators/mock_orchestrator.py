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
    ManagerRequest,
    TaskProgressUpdate,
    ToolOutput,
)
from buttermilk._core.exceptions import FatalError
from buttermilk._core.orchestrator import Orchestrator
from buttermilk._core.types import RunRequest
from buttermilk.api.services.message_service import MessageService
from buttermilk.bm import logger


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
            await self._setup(request or RunRequest(flow="mock_flow"))

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
        except Exception as e:  # noqa: BLE001
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
                # Choose a message type randomly
                message_generator = random.choice([
                    self._generate_agent_trace,
                    self._generate_progress_update,
                    self._generate_error_event,
                    self._generate_manager_request,
                    self._generate_tool_output,
                    self._generate_record,
                ])

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
            # Step 1: Start the flow
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
                agent_id="RESEARCHER",
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

            # Add an analysis record with processed data
            analysis_record = self._generate_record(
                record_id="data-analysis-summary",
                content="## Analysis Results\n\n- Market growth trend: 12.5% increase over 6 months\n- User engagement: 35% higher on new platform\n- Conversion rate: Improved from 3.2% to 4.8%\n\nThese metrics suggest the new strategy is performing above expectations.",
                metadata={
                    "source": "data_processing_engine",
                    "analysis_id": f"analysis-{shortuuid.uuid()[:6]}",
                    "confidence": 0.92,
                    "charts_available": True,
                    "categories": ["market_analysis", "user_metrics", "performance"],
                },
            )
            await self._publish_message(analysis_record)
            await asyncio.sleep(0.8)

            analysis_result = self._generate_agent_trace(
                agent_id="ANALYST",
                outputs="Data analysis reveals strong correlation between variables A and B",
                metadata={"confidence": 0.87, "model": "claude-3"},
            )
            await self._publish_message(analysis_result)
            await asyncio.sleep(1)

            # Step 4: User interaction
            user_request = self._generate_manager_request(
                content="Should I proceed with the detailed analysis or generate a summary?",
                options=["Detailed analysis", "Generate summary"],
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

            # Add a final summary record
            final_record = self._generate_record(
                record_id="final-recommendations",
                content="# Executive Summary\n\nBased on our comprehensive analysis, we recommend the following strategic actions:\n\n1. **Implement Strategy X** - Focus on expanding market reach through targeted campaigns\n2. **Modify Approach Y** - Revise the user onboarding process to improve retention\n3. **Consider Alternative Z** - Explore new partnership opportunities in emerging markets\n\nExpected outcomes include 18% growth in Q2 and improved user satisfaction metrics.",
                metadata={
                    "source": "final_analysis",
                    "report_type": "executive_summary",
                    "priority": "high",
                    "implementation_timeline": "Q2 2025",
                    "stakeholders": ["executive_team", "product_management", "marketing"],
                },
            )
            await self._publish_message(final_record)
            await asyncio.sleep(0.8)

            # Step 6: Completion
            completion = self._generate_progress_update(
                source="ORCHESTRATOR",
                role="CONDUCTOR",
                step_name="flow_completion",
                status="completed",
                message="Workflow completed successfully",
                total_steps=5,
                current_step=5,
            )
            await self._publish_message(completion)

            # Signal termination after simulation is complete
            if self._termination_handler:
                self._termination_handler.request_termination()

        except asyncio.CancelledError:
            logger.debug("Flow simulation cancelled")
            raise

    def _make_publish_callback(self) -> Callable:
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

    # --- Mock Message Generator Methods ---

    def _generate_agent_trace(self, agent_id=None, outputs=None, metadata=None) -> AgentTrace:
        """Generate a fake agent trace"""
        if agent_id is None:
            agent_types = ["ASSISTANT", "RESEARCHER", "JUDGE", "ANALYST", "CRITIC"]
            agent_id = random.choice(agent_types)

        if outputs is None:
            outputs_list = [
                "I've analyzed the data and found several key insights...",
                "The model performance indicates a 87% accuracy rate...",
                "Based on the document analysis, the main themes are...",
                "The search results show 15 relevant articles on this topic...",
                "I've generated a comprehensive response to the query...",
            ]
            outputs = [random.choice(outputs_list)]

        if metadata is None:
            metadata = {
                "duration_ms": random.randint(500, 3000),
                "tokens": random.randint(100, 1000),
                "model": random.choice(["gpt-4", "claude-3", "gemini-pro", "llama-3"]),
            }

        # Create an empty AgentInput for the trace
        agent_input = AgentInput()

        # Create a proper AgentConfig instance
        agent_info = AgentConfig(
            agent_id=agent_id, name=agent_id,
            description=f"{agent_id} agent for processing tasks",
            parameters={},
        )

        return AgentTrace(
            agent_id=agent_id,
            call_id=str(uuid.uuid4()),
            outputs=outputs,
            metadata=metadata,
            timestamp=datetime.now(UTC),
            agent_info=agent_info,
            session_id=self.session_id,
            inputs=agent_input,
        )

    def _generate_progress_update(self, source=None, role=None, step_name=None,
                                status=None, message=None, total_steps=None,
                                current_step=None) -> TaskProgressUpdate:
        """Generate a fake progress update"""
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

        return TaskProgressUpdate(
            source=source,
            role=role,
            step_name=step_name,
            status=status,
            message=message,
            total_steps=total_steps,
            current_step=current_step,
            timestamp=datetime.now(UTC),
        )

    def _generate_error_event(self, source=None, content=None) -> ErrorEvent:
        """Generate a fake error event"""
        if source is None:
            sources = ["orchestrator", "agent", "llm_service", "database", "file_system"]
            source = random.choice(sources)

        if content is None:
            errors = [
                "Connection to LLM service timed out",
                "Failed to parse JSON response",
                "Agent execution exceeded time limit",
                "Invalid configuration detected",
                "Resource limit exceeded",
            ]
            content = random.choice(errors)

        return ErrorEvent(
            source=source,
            content=content,
        )

    def _generate_manager_request(self, content=None, options=None) -> ManagerRequest:
        """Generate a fake manager request"""
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

        return ManagerRequest(
            content=content,
            options=options,
        )

    def _generate_tool_output(self, function_name=None, content=None) -> ToolOutput:
        """Generate a fake tool output"""
        if function_name is None:
            tools = ["search_web", "analyze_document", "fetch_data", "generate_image", "summarize_text"]
            function_name = random.choice(tools)

        if content is None:
            outputs = [
                "Retrieved 5 relevant documents from the search",
                "Analysis complete. Found 12 key entities and 3 main topics.",
                "Data fetched successfully: 256 records processed",
                "Image generated with parameters: style=realistic, subject=landscape",
                "Summary created with 150 words covering the main points",
            ]
            content = random.choice(outputs)

        return ToolOutput(
            name=function_name,
            content=content,
            call_id=str(uuid.uuid4()),
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

        return Record(
            record_id=record_id,
            content=content,
            metadata=metadata,
            mime="text/plain",
        )


async def start_mock_orchestrator(flow_name="mock_flow", record_id="sample-record"):
    """Start a mock orchestrator for testing"""
    from buttermilk._core.types import RunRequest

    # Create a basic run request
    request = RunRequest(
        flow=flow_name,
        record_id=record_id,
        parameters={"criteria": ["test"]},
    )

    # Create and run the mock orchestrator
    orchestrator = MockOrchestrator(
        name=flow_name,
        random_generation=False,  # Follow fixed flow for testing
        message_interval=2.0,
        orchestrator="mock",
    )

    await orchestrator.run(request)


if __name__ == "__main__":
    print("Starting Buttermilk Mock Orchestrator...")
    asyncio.run(start_mock_orchestrator())
