"""Unit tests for CLI bootstrap order validation.

These tests specifically validate the CLI bootstrap sequence fix to ensure:
1. ExecutionContext is created BEFORE sessions in cli.py
2. Sessions use infrastructure from ExecutionContext (not create their own)
3. BM singleton is set AFTER ExecutionContext initialization
4. The bootstrap order prevents infrastructure duplication

This addresses the specific issue where cli.py was creating sessions before
ExecutionContext, causing infrastructure to be split between components.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from omegaconf import OmegaConf

from buttermilk._core.execution_context import (
    _global_execution_context,
    _execution_context_initialized
)


class TestCLIBootstrapOrder:
    """Test CLI bootstrap order matches the architecture fix."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
        
        # Clear BM singleton
        from buttermilk import _global_bm
        _global_bm = None
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    @patch('buttermilk.runner.cli.set_bm')
    @patch('buttermilk.runner.cli.FlowRunner')
    def test_cli_main_bootstrap_order(self, mock_flow_runner_class, mock_set_bm, mock_create_bootstrapper):
        """Test that CLI main() follows correct bootstrap order."""
        # Mock configuration
        mock_conf = OmegaConf.create({
            'run': {
                'name': 'cli_session',
                'job': 'cli_operation',
                'mode': 'console'
            },
            'flow': 'test_flow',
            'ui': 'console'
        })
        
        # Mock bootstrapper and its methods
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        
        # Setup async methods
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            return_value=(mock_execution_context, mock_infrastructure)
        )
        mock_bootstrapper.bootstrap_session_context = AsyncMock(
            return_value=mock_session_bm
        )
        
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # Mock FlowRunner
        mock_flow_runner = Mock()
        mock_flow_runner.mode = 'console'
        mock_flow_runner_class.model_validate.return_value = mock_flow_runner
        
        # Import and run CLI main
        from buttermilk.runner.cli import main
        
        # Patch asyncio.run calls to capture call order
        with patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            # Mock successful asyncio.run calls
            async def mock_run_side_effect(coro):
                if hasattr(coro, '__name__') or hasattr(coro, 'cr_code'):
                    # Handle coroutine objects
                    return await coro
                return coro
            
            mock_asyncio_run.side_effect = mock_run_side_effect
            
            # Mock successful CLI execution to avoid actual flow running
            with patch.object(mock_flow_runner, 'run_flow', new_callable=AsyncMock):
                main(mock_conf)
        
        # Verify bootstrap order:
        # 1. ConfigurationBootstrapper created with existing Hydra config
        mock_create_bootstrapper.assert_called_once_with(
            config_path="../conf",
            config=mock_conf
        )
        
        # 2. bootstrap_full_context called FIRST (ExecutionContext + Infrastructure)
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        
        # 3. bootstrap_session_context called SECOND with existing infrastructure
        mock_bootstrapper.bootstrap_session_context.assert_called_once_with(
            name='cli_session',
            job='cli_operation',
            platform='local',
            infrastructure=mock_infrastructure  # Uses existing infrastructure
        )
        
        # 4. BM singleton set AFTER ExecutionContext initialization
        mock_set_bm.assert_called_once_with(mock_session_bm)
        
        # 5. FlowRunner created with session BM
        mock_flow_runner.set_session_bm.assert_called_once_with(mock_session_bm)
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_infrastructure_sharing_pattern(self, mock_create_bootstrapper):
        """Test that CLI passes infrastructure from ExecutionContext to session."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'test', 'job': 'test', 'mode': 'console'},
            'flow': 'test_flow',
            'ui': 'console'
        })
        
        # Create real bootstrapper to test method interactions
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        
        # Set up bootstrap methods
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            return_value=(mock_execution_context, mock_infrastructure)
        )
        mock_bootstrapper.bootstrap_session_context = AsyncMock(
            return_value=mock_session_bm
        )
        
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # Mock other CLI components
        with patch('buttermilk.runner.cli.FlowRunner') as mock_flow_runner_class, \
             patch('buttermilk.runner.cli.set_bm'), \
             patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            
            mock_flow_runner = Mock()
            mock_flow_runner.mode = 'console'
            mock_flow_runner_class.model_validate.return_value = mock_flow_runner
            
            # Mock asyncio.run to execute coroutines
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            # Mock CLI execution
            with patch.object(mock_flow_runner, 'run_flow', new_callable=AsyncMock):
                from buttermilk.runner.cli import main
                main(mock_conf)
        
        # Verify the infrastructure sharing pattern:
        # 1. Full context bootstrap returns ExecutionContext AND infrastructure
        call_args = mock_bootstrapper.bootstrap_full_context.call_args
        assert call_args == call()  # Called with no arguments
        
        # 2. Session context bootstrap receives the same infrastructure
        session_call_args = mock_bootstrapper.bootstrap_session_context.call_args
        assert session_call_args[1]['infrastructure'] is mock_infrastructure
        
        # This validates that sessions use ExecutionContext's infrastructure
        # rather than creating their own
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_bm_singleton_timing(self, mock_create_bootstrapper):
        """Test that BM singleton is set AFTER ExecutionContext initialization."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'test', 'job': 'test', 'mode': 'console'},
            'flow': 'test_flow',
            'ui': 'console'
        })
        
        # Track call order
        call_order = []
        
        # Mock bootstrapper
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        
        async def track_full_context():
            call_order.append('bootstrap_full_context')
            return mock_execution_context, mock_infrastructure
        
        async def track_session_context(**kwargs):
            call_order.append('bootstrap_session_context')
            return mock_session_bm
        
        mock_bootstrapper.bootstrap_full_context = track_full_context
        mock_bootstrapper.bootstrap_session_context = track_session_context
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # Mock set_bm to track when singleton is set
        def track_set_bm(bm):
            call_order.append('set_bm')
        
        # Mock other components
        with patch('buttermilk.runner.cli.FlowRunner') as mock_flow_runner_class, \
             patch('buttermilk.runner.cli.set_bm', side_effect=track_set_bm), \
             patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            
            mock_flow_runner = Mock()
            mock_flow_runner.mode = 'console'
            mock_flow_runner_class.model_validate.return_value = mock_flow_runner
            
            # Mock asyncio.run to execute coroutines
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            # Mock CLI execution
            with patch.object(mock_flow_runner, 'run_flow', new_callable=AsyncMock):
                from buttermilk.runner.cli import main
                main(mock_conf)
        
        # Verify call order
        expected_order = [
            'bootstrap_full_context',      # ExecutionContext + Infrastructure first
            'bootstrap_session_context',   # Session BM second
            'set_bm'                      # Singleton set last
        ]
        assert call_order == expected_order
    
    def test_cli_configuration_bootstrapper_usage(self):
        """Test that CLI uses ConfigurationBootstrapper correctly."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'test', 'job': 'test', 'mode': 'console'},
            'ui': 'console'
        })
        
        with patch('buttermilk.runner.cli.create_configuration_bootstrapper') as mock_create:
            mock_bootstrapper = Mock()
            mock_bootstrapper.bootstrap_full_context = AsyncMock(return_value=(Mock(), Mock()))
            mock_bootstrapper.bootstrap_session_context = AsyncMock(return_value=Mock())
            mock_create.return_value = mock_bootstrapper
            
            # Mock other components to prevent actual execution
            with patch('buttermilk.runner.cli.FlowRunner'), \
                 patch('buttermilk.runner.cli.set_bm'), \
                 patch('buttermilk.runner.cli.asyncio.run') as mock_run:
                
                # Mock asyncio.run to avoid actual execution
                async def run_coro(coro):
                    return await coro
                mock_run.side_effect = run_coro
                
                from buttermilk.runner.cli import main
                
                # Mock FlowRunner to prevent actual flow execution
                with patch.object(mock_bootstrapper, 'bootstrap_session_context', 
                                new_callable=AsyncMock) as mock_session_bootstrap:
                    mock_session_bm = Mock()
                    mock_session_bootstrap.return_value = mock_session_bm
                    
                    with patch('buttermilk.runner.cli.FlowRunner.model_validate') as mock_validate:
                        mock_flow_runner = Mock()
                        mock_flow_runner.mode = 'console'
                        mock_validate.return_value = mock_flow_runner
                        
                        with patch.object(mock_flow_runner, 'run_flow', new_callable=AsyncMock):
                            main(mock_conf)
            
            # Verify ConfigurationBootstrapper was created with existing Hydra config
            mock_create.assert_called_once_with(
                config_path="../conf",
                config=mock_conf  # Existing config passed to avoid double Hydra init
            )


class TestCLIBootstrapErrorHandling:
    """Test CLI bootstrap error handling scenarios."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_execution_context_failure_handling(self, mock_create_bootstrapper):
        """Test CLI behavior when ExecutionContext bootstrap fails."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'test', 'job': 'test', 'mode': 'console'}
        })
        
        # Mock bootstrapper that fails on full context bootstrap
        mock_bootstrapper = Mock()
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            side_effect=RuntimeError("ExecutionContext initialization failed")
        )
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # CLI should propagate the ExecutionContext failure
        with patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            from buttermilk.runner.cli import main
            
            with pytest.raises(RuntimeError, match="ExecutionContext initialization failed"):
                main(mock_conf)
        
        # Session bootstrap should not be called if ExecutionContext fails
        assert not hasattr(mock_bootstrapper, 'bootstrap_session_context') or \
               not mock_bootstrapper.bootstrap_session_context.called
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_session_failure_with_valid_execution_context(self, mock_create_bootstrapper):
        """Test CLI behavior when session bootstrap fails but ExecutionContext is valid."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'test', 'job': 'test', 'mode': 'console'}
        })
        
        # Mock bootstrapper with successful ExecutionContext but failing session
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            return_value=(mock_execution_context, mock_infrastructure)
        )
        mock_bootstrapper.bootstrap_session_context = AsyncMock(
            side_effect=Exception("Session creation failed")
        )
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # CLI should propagate the session failure
        with patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            from buttermilk.runner.cli import main
            
            with pytest.raises(Exception, match="Session creation failed"):
                main(mock_conf)
        
        # ExecutionContext bootstrap should have been called successfully
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        
        # Session bootstrap should have been attempted
        mock_bootstrapper.bootstrap_session_context.assert_called_once()


class TestCLIBootstrapModeHandling:
    """Test CLI bootstrap with different modes."""
    
    def setup_method(self):
        """Reset global state before each test."""
        global _global_execution_context, _execution_context_initialized
        _global_execution_context = None
        _execution_context_initialized = False
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_api_mode_bootstrap(self, mock_create_bootstrapper):
        """Test CLI bootstrap sequence for API mode."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'api_test', 'job': 'api_job', 'mode': 'api'},
            'host': '0.0.0.0',
            'port': 8000
        })
        
        # Mock bootstrapper
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            return_value=(mock_execution_context, mock_infrastructure)
        )
        mock_bootstrapper.bootstrap_session_context = AsyncMock(
            return_value=mock_session_bm
        )
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # Mock FastAPI components
        with patch('buttermilk.runner.cli.FlowRunner') as mock_flow_runner_class, \
             patch('buttermilk.runner.cli.create_fastapi_app') as mock_create_app, \
             patch('buttermilk.runner.cli.uvicorn.Server') as mock_server_class, \
             patch('buttermilk.runner.cli.set_bm'), \
             patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            
            mock_flow_runner = Mock()
            mock_flow_runner.mode = 'api'
            mock_flow_runner_class.model_validate.return_value = mock_flow_runner
            
            mock_app = Mock()
            mock_app.state.flow_runner = mock_flow_runner
            mock_app.state.infrastructure = mock_infrastructure
            mock_create_app.return_value = mock_app
            
            mock_server = Mock()
            mock_server_class.return_value = mock_server
            
            # Mock asyncio.run
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            from buttermilk.runner.cli import main
            main(mock_conf)
        
        # Verify bootstrap order is same for API mode
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once_with(
            name='api_test',
            job='api_job', 
            platform='local',
            infrastructure=mock_infrastructure
        )
        
        # Verify FastAPI app was created with bootstrapped infrastructure
        mock_create_app.assert_called_once_with(
            infrastructure=mock_infrastructure,
            flows=mock_flow_runner
        )
    
    @patch('buttermilk.runner.cli.create_configuration_bootstrapper')
    def test_cli_batch_mode_bootstrap(self, mock_create_bootstrapper):
        """Test CLI bootstrap sequence for batch mode."""
        mock_conf = OmegaConf.create({
            'run': {'name': 'batch_test', 'job': 'batch_job', 'mode': 'batch'},
            'flow': 'test_flow',
            'dataset_key': 'test_dataset'
        })
        
        # Mock bootstrapper
        mock_bootstrapper = Mock()
        mock_execution_context = Mock()
        mock_infrastructure = Mock()
        mock_session_bm = Mock()
        
        mock_bootstrapper.bootstrap_full_context = AsyncMock(
            return_value=(mock_execution_context, mock_infrastructure)
        )
        mock_bootstrapper.bootstrap_session_context = AsyncMock(
            return_value=mock_session_bm
        )
        mock_create_bootstrapper.return_value = mock_bootstrapper
        
        # Mock batch components
        with patch('buttermilk.runner.cli.FlowRunner') as mock_flow_runner_class, \
             patch('buttermilk.runner.cli.set_bm'), \
             patch('buttermilk.runner.cli.asyncio.run') as mock_asyncio_run:
            
            mock_flow_runner = Mock()
            mock_flow_runner.mode = 'batch'
            mock_flow_runner.create_batch = AsyncMock()
            mock_flow_runner_class.model_validate.return_value = mock_flow_runner
            
            # Mock asyncio.run
            async def run_coro(coro):
                return await coro
            mock_asyncio_run.side_effect = run_coro
            
            from buttermilk.runner.cli import main
            main(mock_conf)
        
        # Verify same bootstrap order for batch mode
        mock_bootstrapper.bootstrap_full_context.assert_called_once()
        mock_bootstrapper.bootstrap_session_context.assert_called_once_with(
            name='batch_test',
            job='batch_job',
            platform='local', 
            infrastructure=mock_infrastructure
        )
        
        # Verify batch operation was called with FlowRunner that has session BM
        mock_flow_runner.set_session_bm.assert_called_once_with(mock_session_bm)
        mock_flow_runner.create_batch.assert_called_once()