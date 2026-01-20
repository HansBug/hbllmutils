"""
This module provides functionality for matching and pairing Python code samples with their corresponding prompts.

It defines matchers for Python files and prompt files, and provides a mechanism to pair them together
for use in prompt engineering and testing scenarios. The module uses pattern matching to identify
related files and organize them into structured pairs.

Classes:
    PyMatcher: Matcher for Python sample files with .py.txt extension
    PromptMatcher: Matcher for prompt files with .prompt.txt extension
    PromptSamplePair: Pairs Python samples with their corresponding prompts

Functions:
    get_prompt_samples: Retrieves all matched prompt-sample pairs from the current directory
"""

import os.path

from hbllmutils.template import BaseMatcher, BaseMatcherPair


class PyMatcher(BaseMatcher):
    """
    Matcher for Python sample files.
    
    This matcher identifies files with the pattern '<name>.py.txt' and extracts
    the name component for pairing with corresponding prompt files.
    
    :ivar __pattern__: The file pattern to match Python sample files
    :vartype __pattern__: str
    :ivar name: The extracted name from the matched file
    :vartype name: str
    """
    __pattern__ = '<name>.py.txt'
    name: str


class PromptMatcher(BaseMatcher):
    """
    Matcher for prompt files.
    
    This matcher identifies files with the pattern '<name>.prompt.txt' and extracts
    the name component for pairing with corresponding Python sample files.
    
    :ivar __pattern__: The file pattern to match prompt files
    :vartype __pattern__: str
    :ivar name: The extracted name from the matched file
    :vartype name: str
    """
    __pattern__ = '<name>.prompt.txt'
    name: str


class PromptSamplePair(BaseMatcherPair):
    """
    Pairs Python sample files with their corresponding prompt files.
    
    This class creates pairs of Python samples and prompts that share the same name,
    enabling structured access to related prompt engineering materials.
    
    :ivar py: The matched Python sample file
    :vartype py: PyMatcher
    :ivar prompt: The matched prompt file
    :vartype prompt: PromptMatcher
    """
    py: PyMatcher
    prompt: PromptMatcher


def get_prompt_samples():
    """
    Retrieve all prompt-sample pairs from the current module's directory.
    
    This function scans the directory containing this module and matches all
    Python sample files (.py.txt) with their corresponding prompt files (.prompt.txt)
    that share the same base name.
    
    :return: A collection of matched prompt-sample pairs found in the directory
    :rtype: list[PromptSamplePair]
    
    Example::
        >>> pairs = get_prompt_samples()
        >>> for pair in pairs:
        ...     print(f"Sample: {pair.py.name}, Prompt: {pair.prompt.name}")
        Sample: example, Prompt: example
    """
    return PromptSamplePair.match_all(os.path.dirname(__file__))
