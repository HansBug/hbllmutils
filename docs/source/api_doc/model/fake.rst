hbllmutils.model.fake
========================================================

.. currentmodule:: hbllmutils.model.fake

.. automodule:: hbllmutils.model.fake


FakeResponseTyping
-----------------------------------------------------

.. autodata:: FakeResponseTyping


FakeResponseSequence
-----------------------------------------------------

.. autoclass:: FakeResponseSequence
    :members: __init__,current_index,total_responses,has_more_responses,rule_check,response,advance,reset,__eq__,__hash__,__repr__


FakeResponseStream
-----------------------------------------------------

.. autoclass:: FakeResponseStream


FakeLLMModel
-----------------------------------------------------

.. autoclass:: FakeLLMModel
    :members: __init__,__setattr__,__delattr__,stream_wps,rules_count,with_stream_wps,response_always,response_when,response_when_keyword_in_last_message,response_sequence,clear_rules,ask,ask_stream,__repr__


