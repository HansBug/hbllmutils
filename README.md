# hbllmutils

[![PyPI](https://img.shields.io/pypi/v/hbllmutils)](https://pypi.org/project/hbllmutils/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/hbllmutils)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/hbllmutils)
![PyPI - Downloads](https://img.shields.io/pypi/dm/hbllmutils)

![Loc](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/f6212c5576d61750212301a636d6c794/raw/loc.json)
![Comments](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/HansBug/f6212c5576d61750212301a636d6c794/raw/comments.json)
[![Maintainability](https://api.codeclimate.com/v1/badges/5b6e14a915b63faeae90/maintainability)](https://codeclimate.com/github/HansBug/hbllmutils/maintainability)
[![codecov](https://codecov.io/gh/hansbug/hbllmutils/branch/main/graph/badge.svg?token=XJVDP4EFAT)](https://codecov.io/gh/hansbug/hbllmutils)

[![Code Test](https://github.com/hansbug/hbllmutils/workflows/Code%20Test/badge.svg)](https://github.com/hansbug/hbllmutils/actions?query=workflow%3A%22Code+Test%22)
[![Badge Creation](https://github.com/hansbug/hbllmutils/workflows/Badge%20Creation/badge.svg)](https://github.com/hansbug/hbllmutils/actions?query=workflow%3A%22Badge+Creation%22)
[![Package Release](https://github.com/hansbug/hbllmutils/workflows/Package%20Release/badge.svg)](https://github.com/hansbug/hbllmutils/actions?query=workflow%3A%22Package+Release%22)

[![GitHub stars](https://img.shields.io/github/stars/hansbug/hbllmutils)](https://github.com/hansbug/hbllmutils/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/hansbug/hbllmutils)](https://github.com/hansbug/hbllmutils/network)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/hansbug/hbllmutils)
[![GitHub issues](https://img.shields.io/github/issues/hansbug/hbllmutils)](https://github.com/hansbug/hbllmutils/issues)
[![GitHub pulls](https://img.shields.io/github/issues-pr/hansbug/hbllmutils)](https://github.com/hansbug/hbllmutils/pulls)
[![Contributors](https://img.shields.io/github/contributors/hansbug/hbllmutils)](https://github.com/hansbug/hbllmutils/graphs/contributors)
[![GitHub license](https://img.shields.io/github/license/hansbug/hbllmutils)](https://github.com/HansBug/hbllmutils/blob/master/LICENSE)

`hbllmutils` is a Python utility library designed to streamline interactions with Large Language Models (LLMs) by
providing robust configuration management, a unified API for OpenAI-compatible endpoints, and intuitive conversation
history handling.

## Features

- **Flexible LLM Configuration**: Easily manage multiple LLM API endpoints and models through a simple YAML
  configuration file (`.llmconfig.yaml`), supporting default and fallback settings.
- **OpenAI-Compatible API**: Interact with various LLM providers that adhere to the OpenAI API specification, offering
  both synchronous and asynchronous request methods.
- **Streaming Responses**: Efficiently handle streaming responses from LLMs, including optional extraction of reasoning
  content.
- **Conversation History Management**: Build and maintain complex conversation histories with support for different
  roles (system, user, assistant) and multimodal content (text, images).
- **Extensible Design**: Built with extensibility in mind, allowing for easy integration of new models or custom
  behaviors.

## Installation

You can simply install it with the `pip` command line from the official PyPI site.

```bash
pip install hbllmutils
```

For more information about installation, you can refer to
the [Installation Guide](https://hbllmutils.readthedocs.io/en/latest/tutorials/installation/index.html).

## Configuration: `.llmconfig.yaml`

The library uses a `.llmconfig.yaml` file to manage your LLM API credentials and model configurations. This file can be
placed in your project's root directory or specified explicitly. Below is an example configuration demonstrating how to
set up multiple API providers and define models, including default and fallback options.

```yaml
deepseek: &deepseek
  base_url: https://api.deepseek.com/v1
  api_token: sk-457***af74

aihubmix: &aihubmix
  base_url: https://aihubmix.com/v1
  api_token: sk-6B9***F0Ad

aigcbest: &aigcbest
  base_url: https://api2.aigcbest.top/v1
  api_token: sk-tbK***49kA

openroute: &openroute
  base_url: https://openrouter.ai/api/v1
  api_token: sk-or-v1-9bf***a3d4

models:
  __default__:
    <<: *deepseek
    model_name: deepseek-chat

  deepseek-R1:
    <<: *deepseek
    model_name: deepseek-reasoner

  deepseek-V3:
    <<: *deepseek
    model_name: deepseek-chat

  __fallback__:
    <<: *aihubmix
```

**Explanation of the configuration:**

- **Anchors (`&` and `*`)**: YAML anchors are used to define reusable blocks. For example, `&deepseek` defines a block
  named `deepseek` which can be referenced later using `*deepseek`.
- **`models` section**: This is the core of your model definitions.
    - `__default__`: Specifies the default model to use if no `model_name` is explicitly provided to `load_llm_model`.
    - `deepseek-R1`, `deepseek-V3`: Specific model configurations that inherit properties from the defined anchors and
      can override them (e.g., `model_name`).
    - `__fallback__`: Defines a fallback API endpoint. If a requested `model_name` is not found in the `models` section,
      the `__fallback__` configuration will be used, with the requested `model_name` automatically assigned.

## Quick Start Example

This example demonstrates how to load a model using the configuration file and interact with it using streaming
responses and conversation history.

First, ensure you have a `.llmconfig.yaml` file set up as described above in your project directory.

```python
import sys
from pprint import pprint

from hbllmutils.history import LLMHistory
from hbllmutils.model import load_llm_model

# Load the LLM model named 'deepseek-V3' from your .llmconfig.yaml
# The library will automatically look for .llmconfig.yaml in the current directory
model = load_llm_model(model_name='deepseek-V3')
print(f"Loaded Model: {model}")
pprint(model)

# Initialize conversation history with a system prompt
history = LLMHistory().set_system_prompt(
    'tell me the appearance of this guy, use json format, like {\'description\': \'xxxxx\', \'name\': \'original name\'}.'
).append_user(
    'donald trump'
)

# Ask the model a question and get a streaming response
# with_reasoning=True will include any internal reasoning from the model in the stream
f = model.ask_stream(
    messages=history.to_json(),
    with_reasoning=True,
)
print(f"\nStreaming Response (with reasoning):\n")

# Iterate through the stream and print chunks as they arrive
for chunk in f:
    print(chunk, end='')
    sys.stdout.flush()

print(f"\n\nAccumulated Reasoning: {f.reasoning_content}")
print(f"Accumulated Content: {f.content}")

# Alternatively, for non-streaming responses, use the ask method:
# response_content = model.ask(messages=history.to_json())
# print(f"\nNon-streaming Response: {response_content}")

# If you need reasoning content for non-streaming:
# reasoning, content = model.ask(messages=history.to_json(), with_reasoning=True)
# print(f"\nNon-streaming Reasoning: {reasoning}")
# print(f"Non-streaming Content: {content}")
```

## Advanced Features

### FakeLLMModel for Testing and Development

`hbllmutils` provides a `FakeLLMModel` that allows developers to simulate LLM behavior for testing, debugging, and rapid
prototyping without incurring API costs or waiting for real API responses. This model can be configured with predefined
rules to return specific responses based on input messages, supporting both synchronous and streaming interactions.

#### Key Features of `FakeLLMModel`:

- **Configurable Responses**: Define rules to return specific text or (reasoning, content) tuples.
- **Conditional Logic**: Set up responses based on conditions like keywords in the last message or custom functions.
- **Streaming Simulation**: Simulate streaming responses with a customizable words-per-second rate.
- **Method Chaining**: Rules can be added using a fluent API.

#### Usage Examples:

```python
from hbllmutils.model import FakeLLMModel
from hbllmutils.history import LLMHistory
import sys

# Initialize FakeLLMModel with a streaming speed of 10 words per second
model = FakeLLMModel(stream_wps=10)

# 1. Always return a specific response
model.response_always("Hello, I am a fake LLM model ready for your commands!")
history_always = LLMHistory().append_user("Hi there!")
response_always = model.ask(history_always.to_json())
print(f"Always Response: {response_always}")
# Expected Output: Always Response: Hello, I am a fake LLM model ready for your commands!

# 2. Respond based on a keyword in the last message
model = FakeLLMModel(stream_wps=10)  # Re-initialize to clear previous rules
model.response_when_keyword_in_last_message("weather", "The weather is sunny with a chance of fake clouds.")
model.response_when_keyword_in_last_message(["time", "hour"], "It\'s always coffee o\'clock in the fake world.")

history_weather = LLMHistory().append_user("What\'s the weather like?")
response_weather = model.ask(history_weather.to_json())
print(f"Weather Response: {response_weather}")
# Expected Output: Weather Response: The weather is sunny with a chance of fake clouds.

history_time = LLMHistory().append_user("What time is it?")
response_time = model.ask(history_time.to_json())
print(f"Time Response: {response_time}")
# Expected Output: Time Response: It\'s always coffee o\'clock in the fake world.

# 3. Respond based on a custom condition
model = FakeLLMModel(stream_wps=10)  # Re-initialize to clear previous rules


def long_conversation_check(messages, **params):
    return len(messages) > 2


model.response_when(long_conversation_check, "This is a long conversation, isn\'t it?")
model.response_always("Short conversation.")  # Fallback for shorter conversations

history_short = LLMHistory().append_user("Hello.")
response_short = model.ask(history_short.to_json())
print(f"Short Conversation: {response_short}")
# Expected Output: Short Conversation: Short conversation.

history_long = LLMHistory().append_user("Hello.").append_assistant("Hi!").append_user("How are you?")
response_long = model.ask(history_long.to_json())
print(f"Long Conversation: {response_long}")
# Expected Output: Long Conversation: This is a long conversation, isn\'t it?

# 4. Streaming responses with reasoning
model = FakeLLMModel(stream_wps=5)  # Slower streaming for demonstration
model.response_always(("Thinking step by step...", "The final answer is 42."))

history_stream = LLMHistory().append_user("What is the meaning of life?")
stream = model.ask_stream(history_stream.to_json(), with_reasoning=True)

print("\nStreaming Response (with reasoning):\n")
for chunk in stream:
    print(chunk, end='')
    sys.stdout.flush()

    print(f"\n\nAccumulated Reasoning: {stream.reasoning_content}")
    print(f"Accumulated Content: {stream.content}")
# Expected Output (with simulated delay):
# Streaming Response (with reasoning):
# Thinking step by step...The final answer is 42.
#
# Accumulated Reasoning: Thinking step by step...
# Accumulated Content: The final answer is 42.
```

### LLM Liveness and Readiness Probes: `hello` and `ping`

`hbllmutils.testing.alive` module provides simple, yet effective, binary tests to check the liveness and readiness of
your LLM models. These functions are particularly useful for ensuring that your integrated LLM services are operational
and responding as expected.

#### `hello` Function

The `hello` function sends a basic greeting to the LLM and checks if it receives any response. It\'s a fundamental
liveness probe to confirm that the model is accessible and capable of generating output.

**Usage Example:**

```python
from hbllmutils.model import FakeLLMModel
from hbllmutils.testing.alive import hello

# Create a fake model for demonstration
model = FakeLLMModel()
model.response_always("Hello! How can I assist you today?")

# Perform a single hello test
hello_result = hello(model)
print(f"Hello Test Passed: {hello_result.passed}")
print(f"Hello Test Content: {hello_result.content}")
# Expected Output:
# Hello Test Passed: True
# Hello Test Content: Hello! How can I assist you today?

# Perform multiple hello tests
multi_hello_results = hello(model, n=5)
print(f"Multi Hello Tests Passed Count: {multi_hello_results.passed_count}")
print(f"Multi Hello Tests Passed Ratio: {multi_hello_results.passed_ratio}")
# Expected Output:
# Multi Hello Tests Passed Count: 5
# Multi Hello Tests Passed Ratio: 1.0
```

#### `ping` Function

The `ping` function sends a "ping!" message to the LLM and expects a response containing "pong" (case-insensitive). This
serves as a readiness probe, verifying that the model can process specific input and generate a predictable response,
indicating its readiness for more complex tasks.

**Usage Example:**

```python
from hbllmutils.model import FakeLLMModel
from hbllmutils.testing.alive import ping

# Create a fake model for demonstration
model = FakeLLMModel()
model.response_when_keyword_in_last_message("ping!", "Pong! I am ready.")

# Perform a single ping test
ping_result = ping(model)
print(f"Ping Test Passed: {ping_result.passed}")
print(f"Ping Test Content: {ping_result.content}")
# Expected Output:
# Ping Test Passed: True
# Ping Test Content: Pong! I am ready.

# Perform multiple ping tests
multi_ping_results = ping(model, n=3)
print(f"Multi Ping Tests Passed Count: {multi_ping_results.passed_count}")
print(f"Multi Ping Tests Passed Ratio: {multi_ping_results.passed_ratio}")
# Expected Output:
# Multi Ping Tests Passed Count: 3
# Multi Ping Tests Passed Ratio: 1.0
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests on
the [GitHub repository](https://github.com/HansBug/hbllmutils).