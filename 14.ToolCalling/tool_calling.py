# 1. Tool creation
# 2. Tool binding
# 3. Tool calling
# 4. Tool execution

from pathlib import Path
from typing import Annotated

import requests
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, InjectedToolArg
from langchain_google_genai import ChatGoogleGenerativeAI


# -----------------------------
# Environment
# -----------------------------

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# -----------------------------
# LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# =========================================================
# 1. TOOL CREATION
# =========================================================

@tool
def get_conversion_factor(
    base_currency: str,
    target_currency: str
) -> float:
    """Get the exchange rate between two currencies."""

    url = (
        f"https://api.frankfurter.dev/v2/"
        f"rate/{base_currency}/{target_currency}"
    )

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()

    return data["rate"]

@tool
def convert(
    base_currency_value: float,
    conversion_rate: Annotated[float, InjectedToolArg]
) -> float:
    """Convert a currency value using the given conversion rate."""

    return base_currency_value * conversion_rate

# =========================================================
# 2. TOOL BINDING
# =========================================================

llm_with_tools = llm.bind_tools([
    get_conversion_factor,
    convert
])

# =========================================================
# 3. TOOL CALLING - ROUND 1
# =========================================================

messages = []

query = HumanMessage(
    content=(
        "What is the conversion factor between INR and USD and "
        "based on that convert 10 INR to USD."
    )
)

messages.append(query)

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)

print("LLM requested tools (round 1):")
print(ai_message.tool_calls)


# =========================================================
# 4. TOOL EXECUTION - ROUND 1 (get_conversion_factor)
# =========================================================

conversion_rate_value = None

for tool_call in ai_message.tool_calls:

    if tool_call["name"] == "get_conversion_factor":

        tool_message = get_conversion_factor.invoke(tool_call)
        messages.append(tool_message)

        conversion_rate_value = float(tool_message.content)

        print("\nConversion factor:")
        print(conversion_rate_value)


# =========================================================
# 5. LLM CALL - ROUND 2 (now model has the rate in context
#    and will request the `convert` tool)
# =========================================================

ai_message_2 = llm_with_tools.invoke(messages)
messages.append(ai_message_2)

print("\nLLM requested tools (round 2):")
print(ai_message_2.tool_calls)


# =========================================================
# 6. TOOL EXECUTION - ROUND 2 (convert)
#    conversion_rate is InjectedToolArg -> LLM won't fill it,
#    we inject it manually before invoking.
# =========================================================

for tool_call in ai_message_2.tool_calls:

    if tool_call["name"] == "convert":

        if conversion_rate_value is not None:
            tool_call["args"]["conversion_rate"] = conversion_rate_value

        tool_message = convert.invoke(tool_call)
        messages.append(tool_message)

        print("\nConverted value:")
        print(tool_message.content)


# =========================================================
# FINAL LLM CALL
# =========================================================

final_response = llm_with_tools.invoke(messages)

print("\nFinal Answer:")
print(final_response.content)