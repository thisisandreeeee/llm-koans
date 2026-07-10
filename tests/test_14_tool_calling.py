import json

import pytest

from llm_koans import koans as K


def test_make_tool_schema_describes_a_python_function_to_the_model():
    schema = K.make_tool_schema(
        name="get_weather",
        description="Look up the current weather for a city.",
        properties={"city": {"type": "string", "description": "City name"}},
        required=["city"],
    )

    assert schema == {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    }


def test_parse_tool_arguments_turns_model_json_into_python_kwargs():
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": '{"left": 2, "right": 3}',
        },
    }

    assert K.parse_tool_arguments(tool_call) == {"left": 2, "right": 3}


def test_execute_tool_call_dispatches_to_registered_python_function():
    def add(left: int, right: int) -> dict[str, int]:
        return {"sum": left + right}

    tool_call = {
        "id": "call_2",
        "type": "function",
        "function": {
            "name": "add",
            "arguments": '{"left": 8, "right": 13}',
        },
    }

    tool_message = K.execute_tool_call(tool_call, {"add": add})

    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_2"
    assert tool_message["name"] == "add"
    assert json.loads(tool_message["content"]) == {"sum": 21}


def test_run_tool_calling_chat_drills_the_assistant_tool_tool_final_pattern():
    class FakeFunctionCallingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, messages, tools):
            self.calls.append((list(messages), tools))
            if len(self.calls) == 1:
                assert messages[-1] == {"role": "user", "content": "Weather in Singapore?"}
                return {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Singapore"}',
                            },
                        }
                    ],
                }

            assert messages[-1]["role"] == "tool"
            assert messages[-1]["tool_call_id"] == "call_weather"
            assert json.loads(messages[-1]["content"]) == {"city": "Singapore", "forecast": "sunny"}
            return {"role": "assistant", "content": "Singapore is sunny."}

    def get_weather(city: str) -> dict[str, str]:
        return {"city": city, "forecast": "sunny"}

    tool_schema = K.make_tool_schema(
        "get_weather",
        "Look up the current weather for a city.",
        {"city": {"type": "string"}},
        ["city"],
    )
    model = FakeFunctionCallingModel()

    conversation = K.run_tool_calling_chat(
        model=model,
        messages=[{"role": "user", "content": "Weather in Singapore?"}],
        tool_registry={"get_weather": get_weather},
        tool_schemas=[tool_schema],
    )

    assert [message["role"] for message in conversation] == ["user", "assistant", "tool", "assistant"]
    assert conversation[-1] == {"role": "assistant", "content": "Singapore is sunny."}
    assert len(model.calls) == 2
    assert model.calls[0][1] == [tool_schema]


def test_run_tool_calling_chat_stops_instead_of_looping_forever():
    def always_calls_tool(messages, tools):
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_again",
                    "type": "function",
                    "function": {"name": "ping", "arguments": "{}"},
                }
            ],
        }

    try:
        K.run_tool_calling_chat(
            model=always_calls_tool,
            messages=[{"role": "user", "content": "Keep going"}],
            tool_registry={"ping": lambda: "pong"},
            tool_schemas=[],
            max_tool_rounds=2,
        )
    except K.KoanIncomplete:
        raise
    except RuntimeError as exc:
        assert "max_tool_rounds" in str(exc)
    else:
        pytest.fail("Expected the chatbot loop to stop at max_tool_rounds")
