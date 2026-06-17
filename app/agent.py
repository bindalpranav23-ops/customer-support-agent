# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Literal
from pydantic import BaseModel, Field

import google.auth
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.apps import App
from google.adk.events.event import Event
from google.adk.models import Gemini
from google.adk.workflow import Workflow, START
from google.genai import types

# Load .env file if it exists to support local development with GOOGLE_API_KEY
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

# Configure Vertex AI vs Gemini Developer API based on GOOGLE_API_KEY presence
if (
    os.environ.get("GOOGLE_API_KEY")
    and os.environ.get("GOOGLE_GENAI_USE_VERTEXAI") != "True"
):
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "False"
else:
    # Default to Vertex AI as per standard scaffold setup, but handle authentication gracefully
    try:
        _, project_id = google.auth.default()
        os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"
    except Exception:
        # If credentials aren't available, we don't block the import
        pass


def extract_text(node_input) -> str:
    if isinstance(node_input, str):
        return node_input
    if hasattr(node_input, "parts"):
        parts = []
        for part in node_input.parts:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
        return "".join(parts)
    return str(node_input)


# Classification Output Schema
class ClassificationOutput(BaseModel):
    category: Literal["shipping", "unrelated"] = Field(
        description="Choose 'shipping' if the user's query is about rates, tracking, delivery, or returns. Choose 'unrelated' otherwise."
    )
    reasoning: str = Field(
        description="Brief explanation of why the query was classified as such."
    )


# Node 1: Preprocess user input
def preprocess_query(ctx: Context, node_input) -> Event:
    query_text = extract_text(node_input)
    return Event(output=query_text, state={"original_query": query_text})


# Node 2: Classify query using LlmAgent
classify_query = LlmAgent(
    name="classify_query",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
        "You are a routing classifier. Determine if the user's query is shipping-related "
        "(such as tracking, shipping rates, shipping time, delivery status, returns, "
        "packaging, or shipping addresses) or completely unrelated.\n"
        "Choose either 'shipping' or 'unrelated' and explain your reasoning."
    ),
    output_schema=ClassificationOutput,
    output_key="classification",
)


# Node 3: Route query based on classification
def route_query(ctx: Context, node_input: dict) -> Event:
    category = node_input.get("category", "unrelated")
    original_query = ctx.state.get("original_query", "")
    return Event(output=original_query, route=category)


# Node 4 (Shipping Branch): Answer shipping questions
shipping_faq_agent = LlmAgent(
    name="shipping_faq_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    instruction=(
       "You are a cheerful and enthusiastic shipping company customer support representative.\n"
    "Answer questions about shipping rates, tracking packages, delivery issues, and returns politely and concisely.\n"
    "Use friendly emojis such as 📦, 🚚, and ✨ for shipping-rate responses.\n"
    "For this demo, orders of $50 or more qualify for FREE standard shipping. Clearly highlight this threshold.\n"
    "If the customer asks something you don't know, politely tell them to contact direct support.\n"
    ),
)


# Node 5 (Unrelated Branch): Decline query politely
def decline_query(node_input: str):
    refusal = (
        "I am a shipping customer support representative. I can only assist you with shipping-related "
        "queries such as rates, tracking, delivery, and returns. How can I help you with your shipping needs today?"
    )
    yield Event(
        content=types.Content(
            role="model",
            parts=[types.Part.from_text(text=refusal)],
        )
    )
    yield Event(output=refusal)


# Define Graph Workflow edges
root_agent = Workflow(
    name="customer_support",
    edges=[
        (START, preprocess_query),
        (preprocess_query, classify_query),
        (classify_query, route_query),
        (
            route_query,
            {
                "shipping": shipping_faq_agent,
                "unrelated": decline_query,
            },
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="app",
)
