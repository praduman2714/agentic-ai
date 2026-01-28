from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
import os

import requests


load_dotenv()

OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
model = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_HOST,
        temperature=0.2,        # critical for factual stability
        # num_predict=800,        # max tokens to generate
        timeout=60              # avoid hanging workers
    )


from langgraph.graph.message import add_messages

class InputType(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetches the current stock price for the given symbol.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()

@tool
def purchase_stock(symbol: str, quantity: int) -> str:
    """
    Simulates purchasing a given quantity of stock for the specified symbol.
    """
    decision = interrupt(f"Do you want to purchase {quantity} shares of {symbol}? (yes/no)")

    if isinstance(decision, str) and decision.lower() == "yes":
        return f"Purchased {quantity} shares of {symbol}."
    else:
        return "Purchase cancelled."

tools = [get_stock_price, purchase_stock]
tool_node = ToolNode(tools)

llm_with_tools = model.bind_tools(tools)

def chat_node(state: InputType) -> InputType:
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

checkpointer = MemorySaver()
graph = StateGraph(InputType)


graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    config = {"configurable":{"thread_id" : "1234"}}

    while True:
        user_input = input("User: ")
        intitial_input = {
            "messages" : [
                ("user" , user_input)
            ]
        }
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        result = chatbot.invoke(intitial_input, config=config)
        message = result.get("__interrupt__" , [])

        if message:
            prompt_to_human = message[0]
            print(f"Human Intervention Required: {prompt_to_human}")
            decison = input("Enter your decision: ").strip().lower()

            result = chatbot.invoke(
                Command(resume={"approved": decison}),
                config=config,
            )
            
        messages = result["messages"]
        last_message = messages[-1]
        print(f"Chatbot: {last_message.content}")
        
