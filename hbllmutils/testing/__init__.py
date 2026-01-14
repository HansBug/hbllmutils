"""
This module provides a testing framework for binary tests on language models.

It exports the core testing infrastructure including base classes for binary tests,
result containers for single and multiple test runs, and basic test implementations.
The framework enables structured testing of language models with pass/fail outcomes
and statistical analysis of multiple test runs.

Exported Classes:
    BinaryTest: Base class for implementing binary tests on language models
    BinaryTestResult: Stores the result of a single binary test
    MultiBinaryTestResult: Stores and analyzes results from multiple binary tests

Exported Functions:
    hello: Performs a basic greeting test on an LLM model
    ping: Performs a ping-pong response test on an LLM model
"""

from .alive import hello, ping
from .base import BinaryTest, MultiBinaryTestResult, BinaryTestResult
