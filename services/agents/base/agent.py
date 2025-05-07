from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union, TypedDict

from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from services.llm.base.llm import LLM


class AgentResponse(BaseModel):
    """Base response model for all agents"""
    answer: Union[str, Dict[str, Any]]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


AgentState = Union[Dict[str, Any], TypedDict, BaseModel]


class Agent(ABC):
    """Base class for all agents"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.workflow = self._create_workflow()

    @abstractmethod
    async def invoke(self, **kwargs) -> AgentResponse:
        """Main entry point for agent execution"""
        pass

    @abstractmethod
    def _create_workflow(self) -> CompiledStateGraph:
        """Create and return the agent's workflow"""
        pass

    def _format_response(self, result: AgentState) -> AgentResponse:
        """Format raw result into standardized response"""
        try:
            return AgentResponse(
                answer=self._extract_answer(result),
                metadata=self._extract_metadata(result)
            )
        except Exception as e:
            return AgentResponse(
                answer="",
                error=str(e)
            )

    def _extract_answer(self, state: AgentState) -> str:
        """Extract final answer from result. Override if needed."""
        return str(state.get("answer", ""))

    def _extract_metadata(self, state: AgentState) -> Optional[AgentState]:
        """Extract metadata from result. Override if needed."""
        return None


class ReActAgent(Agent):
    """Base class for ReAct pattern agents"""

    def __init__(self, llm: LLM):
        self.llm = llm
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.chat.bind_tools(self.tools)
        super().__init__(llm)

    @abstractmethod
    def _create_tools(self) -> List:
        """Create and return list of tools for the agent"""
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass

    def think(self, state: MessagesState) -> AgentState:
        """Default thinking implementation for ReAct pattern"""
        sys_msg = SystemMessage(content=self._get_system_prompt())
        messages: List[BaseMessage] = [sys_msg] + state["messages"]
        invocation = self.llm_with_tools.invoke(messages)
        return {"messages": state["messages"] + [invocation]}

    def _extract_answer(self, state: AgentState) -> str:
        """Extract answer from ReAct agent result"""
        messages = state.get("messages", [])
        return messages[-1].content if messages else ""
