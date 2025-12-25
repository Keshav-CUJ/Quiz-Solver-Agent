# app_prep_gemini.py
import json
from langgraph.graph import StateGraph, END
from state import AppState
from nodes.prep_agent import prep_agent_node
from nodes.execution_agent import execution_agent_node
from nodes.submit_agent import submit_agent_node

def build_app():
    graph = StateGraph(AppState)

    # Adding nodes
    graph.add_node("prep_agent", prep_agent_node)
    graph.add_node("execution_agent", execution_agent_node)
    graph.add_node("submit_agent", submit_agent_node)

    # Entry point: start with prep
    graph.set_entry_point("prep_agent")

    # Flow: prep_agent -> execution_agent -> END
    graph.add_edge("prep_agent", "execution_agent")
    graph.add_edge("execution_agent", "submit_agent")
    graph.add_edge("submit_agent", END)

    return graph.compile()

