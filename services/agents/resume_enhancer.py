import asyncio
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from pinecone import Pinecone
from services.agents.tools.get_user_resume import get_user_resume

load_dotenv()


#############################################
# Define the Agent State Schema
#############################################
class EnhancementState(MessagesState):
    """
    Agent state. In this case, it holds the resume text and conversation messages.
    """
    resume_text: str


#############################################
# Structured Output Schema for Response
#############################################
class ResumeEnhancementResponse(BaseModel):
    """
    Structured output for resume enhancement.

    Attributes:
        resume_text: The original resume text.
        formatting_issues: A list of identified formatting issues.
    """
    resume_text: str
    formatting_issues: List[Dict[str, Any]] = Field(
        default_factory=list, description="Identified formatting issues"
    )


#############################################
# Define a Single Tool for Formatting Analysis
#############################################
@tool
def analyze_formatting_tool(tool_call_id: Optional[str] = None, resume_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Analyze the resume for formatting issues.

    Args:
        tool_call_id: A unique identifier for this tool call.
        resume_text: The complete resume text provided as input.

    Returns:
        A dictionary update containing the resume_text, list of formatting issues, and a log message.
    """
    if tool_call_id is None:
        tool_call_id = "default_tool_call_id"
    if not resume_text:
        tm = ToolMessage(
            content="No resume text provided.",
            name="analyze_formatting_tool",
            tool_call_id=tool_call_id
        )
        return {"resume_text": "", "formatting_issues": [], "messages": [tm]}

    print("Analyzing formatting...", tool_call_id, "Resume length:", len(resume_text))
    issues = [
        {"issue": "Inconsistent bullet points", "severity": "low"},
        {"issue": "Irregular spacing", "severity": "medium"}
    ]
    tm = ToolMessage(
        content=f"Formatting analysis found {len(issues)} issues.",
        name="analyze_formatting_tool",
        tool_call_id=tool_call_id
    )
    return {"resume_text": resume_text, "formatting_issues": issues, "messages": [tm]}


tools = [analyze_formatting_tool]

#############################################
# Initialize ChatGoogleGenerativeAI and Bind Tools
#############################################
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
llm_with_tools = llm.bind_tools(tools)


#############################################
# Create the Assistant Function
#############################################
def assistant(state: EnhancementState) -> Dict[str, Any]:
    """
    The assistant node that performs reasoning. It creates a system message that includes
    the resume text, then calls the LLM (with tools bound) using the system message appended
    to the conversation history.
    """
    print("Assistant invoked", state)
    resume_text = state.get("resume_text", "")
    sys_msg = SystemMessage(
        content=f"You are a helpful assistant. Your task is to analyze resumes for formatting issues. "
                f"Here is the resume text:\n{resume_text}"
    )
    return {"messages": [llm_with_tools.invoke([sys_msg] + state.get("messages", []))]}


#############################################
# Build the ReAct Agent using a StateGraph
#############################################
builder = StateGraph(EnhancementState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()


#############################################
# Main Execution
#############################################
async def main():
    # Retrieve resume data from get_user_resume.
    user_id = "oPmOJhSE0VQid56yYyg19hdH5DV2"
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index("resumes")
    resume_data = await get_user_resume(index, user_id)
    if not resume_data.get("text"):
        print("Resume text not found")
        return
    react_graph.get_graph().print_ascii()
    initial_state = {
        "resume_text": resume_data["text"],
        "messages": [
            {"role": "user", "content": "Please analyze my resume for formatting issues."}
        ]
    }

    # Invoke the ReAct agent graph with the initial state.
    result = react_graph.invoke(initial_state)
    print("Final Result:")
    print(result)

    # Optionally, print all processing messages to see the complete chain-of-thought.
    for m in result["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
