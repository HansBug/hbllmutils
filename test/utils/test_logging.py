"""
Unit tests for the logging module.

This module contains comprehensive tests for the global logger configuration
and management utilities in hbllmutils.utils.logging.
"""

import logging

import pytest

from hbllmutils.utils.logging import get_global_logger


@pytest.fixture
def reset_logger():
    """
    Reset the global logger to its default state after each test.
    
    This fixture ensures test isolation by clearing handlers and resetting
    the logger configuration after each test execution.
    """
    logger = logging.getLogger('hbllmutils')
    original_level = logger.level
    original_handlers = logger.handlers.copy()
    original_propagate = logger.propagate

    yield logger

    # Cleanup: restore original state
    logger.setLevel(original_level)
    logger.handlers.clear()
    for handler in original_handlers:
        logger.addHandler(handler)
    logger.propagate = original_propagate


@pytest.mark.unittest
class TestGetGlobalLogger:
    """Tests for the get_global_logger function."""

    def test_returns_logger_instance(self):
        """Test that get_global_logger returns a logging.Logger instance."""
        logger = get_global_logger()
        assert isinstance(logger, logging.Logger)

    def test_returns_correct_logger_name(self):
        """Test that the returned logger has the correct name 'hbllmutils'."""
        logger = get_global_logger()
        assert logger.name == 'hbllmutils'

    def test_returns_same_instance(self):
        """Test that multiple calls return the same logger instance."""
        logger1 = get_global_logger()
        logger2 = get_global_logger()
        assert logger1 is logger2

    def test_logger_can_log_messages(self, reset_logger, caplog):
        """Test that the logger can successfully log messages at various levels."""
        logger = get_global_logger()

        with caplog.at_level(logging.DEBUG):
            logger.debug('Debug message')
            logger.info('Info message')
            logger.warning('Warning message')
            logger.error('Error message')
            logger.critical('Critical message')

        assert 'Debug message' in caplog.text
        assert 'Info message' in caplog.text
        assert 'Warning message' in caplog.text
        assert 'Error message' in caplog.text
        assert 'Critical message' in caplog.text

    def test_logger_level_configuration(self, reset_logger, caplog):
        """Test that logger level can be configured and affects message filtering."""
        logger = get_global_logger()
        logger.setLevel(logging.WARNING)

        with caplog.at_level(logging.DEBUG):
            logger.debug('Debug message')
            logger.info('Info message')
            logger.warning('Warning message')
            logger.error('Error message')

        # Debug and Info should not appear due to WARNING level
        assert 'Debug message' not in caplog.text
        assert 'Info message' not in caplog.text
        assert 'Warning message' in caplog.text
        assert 'Error message' in caplog.text

    def test_logger_handler_addition(self, reset_logger, capsys):
        """Test that handlers can be added to the logger."""
        logger = get_global_logger()
        logger.handlers.clear()  # Clear any existing handlers

        # Add a stream handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        logger.info('Test message')

        captured = capsys.readouterr()
        assert 'hbllmutils - INFO - Test message' in captured.err

    @pytest.mark.parametrize("level,level_name", [
        (logging.DEBUG, 'DEBUG'),
        (logging.INFO, 'INFO'),
        (logging.WARNING, 'WARNING'),
        (logging.ERROR, 'ERROR'),
        (logging.CRITICAL, 'CRITICAL'),
    ])
    def test_logger_various_levels(self, reset_logger, caplog, level, level_name):
        """Test logging at various levels using parameterization."""
        logger = get_global_logger()
        logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            logger.log(level, f'Message at {level_name}')

        assert f'Message at {level_name}' in caplog.text
        assert level_name in caplog.text

    def test_logger_propagation(self, reset_logger):
        """Test that logger propagation can be controlled."""
        logger = get_global_logger()

        # Default propagation should be True
        assert logger.propagate is True

        # Test disabling propagation
        logger.propagate = False
        assert logger.propagate is False

    def test_logger_parent_hierarchy(self):
        """Test that the logger follows Python's logging hierarchy."""
        logger = get_global_logger()

        # The parent should be the root logger
        assert logger.parent is logging.getLogger()

    def test_logger_effective_level(self, reset_logger):
        """Test that effective level is correctly determined."""
        logger = get_global_logger()

        # Set a specific level
        logger.setLevel(logging.WARNING)
        assert logger.level == logging.WARNING
        assert logger.getEffectiveLevel() == logging.WARNING

    def test_logger_with_exception_info(self, reset_logger, caplog):
        """Test that logger can log exception information."""
        logger = get_global_logger()

        try:
            raise ValueError('Test exception')
        except ValueError:
            with caplog.at_level(logging.ERROR):
                logger.exception('An error occurred')

        assert 'An error occurred' in caplog.text
        assert 'ValueError: Test exception' in caplog.text
        assert 'Traceback' in caplog.text

    def test_logger_multiple_handlers(self, reset_logger):
        """Test that multiple handlers can be added to the logger."""
        logger = get_global_logger()
        logger.handlers.clear()

        handler1 = logging.StreamHandler()
        handler2 = logging.StreamHandler()

        logger.addHandler(handler1)
        logger.addHandler(handler2)

        assert len(logger.handlers) == 2
        assert handler1 in logger.handlers
        assert handler2 in logger.handlers

    def test_logger_handler_removal(self, reset_logger):
        """Test that handlers can be removed from the logger."""
        logger = get_global_logger()
        logger.handlers.clear()

        handler = logging.StreamHandler()
        logger.addHandler(handler)
        assert handler in logger.handlers

        logger.removeHandler(handler)
        assert handler not in logger.handlers

    def test_logger_filter_addition(self, reset_logger, caplog):
        """Test that filters can be added to the logger."""
        logger = get_global_logger()

        # Create a simple filter that blocks messages containing 'blocked'
        class SimpleFilter(logging.Filter):
            def filter(self, record):
                return 'blocked' not in record.getMessage()

        logger.addFilter(SimpleFilter())

        with caplog.at_level(logging.INFO):
            logger.info('This message should appear')
            logger.info('This message is blocked')

        assert 'This message should appear' in caplog.text
        assert 'This message is blocked' not in caplog.text

    def test_logger_is_enabled_for(self, reset_logger):
        """Test the isEnabledFor method of the logger."""
        logger = get_global_logger()
        logger.setLevel(logging.WARNING)

        assert not logger.isEnabledFor(logging.DEBUG)
        assert not logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.WARNING)
        assert logger.isEnabledFor(logging.ERROR)
        assert logger.isEnabledFor(logging.CRITICAL)
