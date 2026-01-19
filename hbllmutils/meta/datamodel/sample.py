import os.path

from hbllmutils.template import BaseMatcher, BaseMatcherPair


class PyMatcher(BaseMatcher):
    __pattern__ = '<name>.py.txt'
    name: str


class PromptMatcher(BaseMatcher):
    __pattern__ = '<name>.prompt.txt'
    name: str


class PromptSamplePair(BaseMatcherPair):
    py: PyMatcher
    prompt: PromptMatcher


def get_prompt_samples():
    return PromptSamplePair.match_all(os.path.dirname(__file__))
