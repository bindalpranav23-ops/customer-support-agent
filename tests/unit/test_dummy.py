import pytest
from google.genai import types
from google.adk.events.event import Event
from google.adk.agents.context import Context

from app.agent import preprocess_query, route_query, decline_query


class MockContext:
    def __init__(self, state=None):
        self.state = state or {}


def test_preprocess_query_string() -> None:
    ctx = MockContext()
    event = preprocess_query(ctx, "hello query")
    assert isinstance(event, Event)
    assert event.output == "hello query"
    assert event.actions.state_delta == {"original_query": "hello query"}


def test_preprocess_query_content() -> None:
    ctx = MockContext()
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text="hello content")]
    )
    event = preprocess_query(ctx, content)
    assert isinstance(event, Event)
    assert event.output == "hello content"
    assert event.actions.state_delta == {"original_query": "hello content"}


def test_route_query_shipping() -> None:
    ctx = MockContext(state={"original_query": "where is my package"})
    event = route_query(ctx, {"category": "shipping", "reasoning": "some reasoning"})
    assert isinstance(event, Event)
    assert event.output == "where is my package"
    assert event.actions.route == "shipping"


def test_route_query_unrelated() -> None:
    ctx = MockContext(state={"original_query": "what is 2+2"})
    event = route_query(ctx, {"category": "unrelated", "reasoning": "math query"})
    assert isinstance(event, Event)
    assert event.output == "what is 2+2"
    assert event.actions.route == "unrelated"


def test_decline_query() -> None:
    generator = decline_query("some query")
    events = list(generator)
    
    assert len(events) == 2
    
    # First event should have the content for the UI
    content_event = events[0]
    assert content_event.content is not None
    assert "shipping customer support" in content_event.content.parts[0].text
    
    # Second event should have the final output string
    output_event = events[1]
    assert output_event.output is not None
    assert "shipping customer support" in output_event.output
