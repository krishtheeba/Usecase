import os
from typing import TypedDict

import streamlit as st
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END


# -----------------------------
# Connect LLM
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"  # replace your api_key
)


# -----------------------------
# Shared state across all nodes
# -----------------------------
class UIState(TypedDict):
    user_request: str
    plan: str
    reasoning: str
    task_type: str
    response: str


# -----------------------------
# Node 1: Planning step
# Understand what user wants
# -----------------------------
def planner(state: UIState):
    request = state["user_request"]

    prompt = f"Create a short plan for solving this frontend UI request: {request}"
    plan = llm.invoke(prompt).content

    return {"plan": plan}


# -----------------------------
# Node 2: Reasoning step
# Simulates LLM thinking / classification
# -----------------------------
def reasoning_node(state: UIState):
    request = state["user_request"]

    prompt = (
        "Classify this frontend request into one of these: "
        "component_generation, css_fix, general_ui_help. "
        f"Also explain why. Request: {request}"
    )

    llm_response = llm.invoke(prompt).content

    if "component_generation" in llm_response.lower():
        task = "component_generation"
    elif "css_fix" in llm_response.lower():
        task = "css_fix"
    else:
        task = "general_ui_help"

    reasoning = llm_response

    return {
        "task_type": task,
        "reasoning": reasoning
    }


# -----------------------------
# Node 3: Final solution generation
# LLM-style response generation
# -----------------------------
def generate_solution(state: UIState):
    request = state["user_request"]
    task = state["task_type"]

    prompt = (
        f"Generate a simple frontend solution for this request: {request}. "
        f"Task type: {task}"
    )

    result = llm.invoke(prompt).content

    return {"response": result}


# -----------------------------
# Build LangGraph workflow
# -----------------------------
def build_graph():
    builder = StateGraph(UIState)

    builder.add_node("planner", planner)
    builder.add_node("reasoning_node", reasoning_node)
    builder.add_node("generate_solution", generate_solution)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "reasoning_node")
    builder.add_edge("reasoning_node", "generate_solution")
    builder.add_edge("generate_solution", END)

    return builder.compile()


graph = build_graph()


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(page_title="Frontend UI Agent", layout="wide")

    st.title("Frontend UI Agent using LangGraph + Groq")
    st.write("Enter your frontend UI request and generate a plan, reasoning, and final solution.")

    user_request = st.text_area(
        "Enter your frontend request:",
        placeholder="Example: Create a responsive login form using React",
        height=150
    )

    if st.button("Generate Solution"):
        if not user_request.strip():
            st.warning("Please enter a frontend request.")
            return

        with st.spinner("Running LangGraph workflow..."):
            input_state = {
                "user_request": user_request,
                "plan": "",
                "reasoning": "",
                "task_type": "",
                "response": ""
            }

            output = graph.invoke(input_state)

        st.success("Generation complete!")

        st.subheader("Plan")
        st.write(output.get("plan", ""))

        st.subheader("Reasoning")
        st.write(output.get("reasoning", ""))

        st.subheader("Task Type")
        st.write(output.get("task_type", ""))

        st.subheader("Final Output")
        st.write(output.get("response", ""))


if __name__ == "__main__":
    main()
