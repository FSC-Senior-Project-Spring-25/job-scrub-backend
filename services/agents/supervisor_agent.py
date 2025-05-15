import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pinecone import Pinecone
from pydantic import BaseModel

from models.chat import Message
from services.agents.base.agent import AgentResponse
from services.agents.job_search_agent import JobSearchAgent
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.tools.chat_handler import handle_chat
from services.agents.user_profile_agent import UserProfileAgent
from services.agents.user_search_agent import UserSearchAgent
from services.llm.base.llm import LLM
from services.llm.gemini import GeminiLLM

load_dotenv()


class AgentSelection(BaseModel):
    agents: List[str]


VALID_AGENTS = {
    "RESUME_MATCHER",
    "RESUME_ENHANCER",
    "USER_PROFILE",
    "JOB_SEARCH",
    "USER_SEARCH",
    "CHAT",
}


@dataclass
class SupervisorState:
    user_id: str
    current_message: str
    resume_text: str
    conversation_history: List[Message]
    active_agents: List[str]
    files: Optional[List[Dict[str, Any]]] = None
    agent_results: Dict[str, AgentResponse] = None
    final_response: str = ""
    error: Optional[str] = None


class SupervisorAgent:
    """Coordinates specialized agents and streams a unified answer."""

    def __init__(
            self,
            pc: Pinecone,
            resume_matcher: ResumeMatchingAgent,
            resume_enhancer: ResumeEnhancementAgent,
            user_profile_agent: UserProfileAgent,
            job_search_agent: JobSearchAgent,
            user_search_agent: UserSearchAgent,
            llm: LLM = GeminiLLM(),
            resume_text: str = "",
            conversation_history: Optional[List[Message]] = None,
            files: Optional[List[Dict[str, Any]]] = None,
    ):
        self.llm = llm
        self.pc = pc
        self.resume_matcher = resume_matcher
        self.resume_enhancer = resume_enhancer
        self.user_profile_agent = user_profile_agent
        self.job_search_agent = job_search_agent
        self.user_search_agent = user_search_agent
        self.resume_text = resume_text
        self.processed_history = conversation_history or []
        self.processed_files = files or []
        self.workflow = self._create_workflow()

    async def process_message(
            self, user_id: str, message: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream results while letting LangGraph drive control‑flow."""

        init_state = SupervisorState(
            user_id=user_id,
            current_message=message,
            resume_text=self.resume_text,
            conversation_history=self.processed_history,
            files=self.processed_files,
            active_agents=[],
            agent_results={},
        )

        # Track the current state between events
        current_state = init_state

        # Set stream_mode to "updates" to get node updates with state
        async for output in self.workflow.astream(init_state, stream_mode="updates"):
            # When using "updates" mode, output will be a dict with node name as key
            # and the node's updates as value
            if output:  # Check if output is not empty
                node_name = next(iter(output.keys()), None)
                if node_name == "analyze":
                    # After analyze node runs, active_agents is populated
                    current_state.active_agents = output["analyze"].get("active_agents", [])
                    yield {
                        "type": "agents_selected",
                        "agents": [a.lower() for a in current_state.active_agents],
                    }

                elif node_name == "execute":
                    # Immediate chat short‑circuit
                    if current_state.active_agents == ["CHAT"] and "chat" in output["execute"].get("agent_results", {}):
                        current_state.agent_results = output["execute"]["agent_results"]
                        chat_resp = current_state.agent_results["chat"]
                        text = chat_resp.answer if isinstance(chat_resp, AgentResponse) else str(chat_resp)
                        yield {"type": "content_chunk", "content": text}
                        current_state.final_response = text  # Make sure this is a string

                        # Add the messages to conversation history directly
                        ts = int(datetime.utcnow().timestamp())
                        current_state.conversation_history.append(Message(
                            role="user",
                            content=current_state.current_message,
                            timestamp=ts
                        ))
                        current_state.conversation_history.append(Message(
                            role="assistant",
                            content=text,
                            timestamp=ts,
                            activeAgents=current_state.active_agents
                        ))
                    else:
                        # Update state with execute results
                        for k, v in output["execute"].items():
                            setattr(current_state, k, v)

                        # Stream synthesized response
                        async for chunk in self._stream_synthesize_response(current_state):
                            yield {"type": "content_chunk", "content": chunk}

                        # Ensure final_response is set after streaming completes
                        if not current_state.final_response:
                            print("[SUPERVISOR] Warning: final_response is empty after streaming")
                            current_state.final_response = "I apologize, but I couldn't generate a proper response."

                elif node_name == "update":
                    # Update final state
                    for k, v in output["update"].items():
                        setattr(current_state, k, v)
                    yield {
                        "type": "complete",
                        "response": current_state.final_response,
                        "conversation": [
                            {
                                "role": m.role,
                                "content": m.content,
                                "timestamp": m.timestamp,
                                "metadata": m.metadata if hasattr(m, "metadata") else {},
                            }
                            for m in current_state.conversation_history
                        ],
                        "agents": current_state.active_agents,
                    }

    async def _analyze_message(self, state: SupervisorState) -> SupervisorState:
        prompt = f"""
        Analyze the following user message and determine which specialized agent(s) should handle it.
        We are using a multi‑agent system, so multiple agents can be activated simultaneously.

        User message: "{state.current_message}"

        HISTORY: {json.dumps([m.content for m in state.conversation_history[-2:]])}

        Choose the most relevant agents based on the user message and conversation history.
        Available agents:
        - RESUME_MATCHER: For questions about job fit, matching resumes to jobs, or compatibility with positions
        - RESUME_ENHANCER: For messages related to analyzing, improving resumes, ats optimization, or resume writing
        - USER_PROFILE: For questions about the user's skills, background, profile, or identity
        - JOB_SEARCH: For requests to find or search for job postings
        - USER_SEARCH: For requests to find or search users with resumes matching specific criteria
        - CHAT: For general conversation or questions not covered by specialized agents

        Multiple agents can be activated simultaneously. Select all applicable agents.
        Return a JSON object with the following format: {{"agents": ["AGENT_NAME1", "AGENT_NAME2", ...]}}
        """
        response = await self.llm.agenerate(
            system_prompt="You are a routing assistant that analyzes user messages and determines which specialized agents should process them.",
            user_message=prompt,
            response_format=AgentSelection,
        )
        print(f"[SUPERVISOR] Routing Response: {response.content}")
        chosen = [a for a in (response.content.agents if response.success else []) if a in VALID_AGENTS] or ["CHAT"]
        state.active_agents = chosen
        print(f"[SUPERVISOR] Active agents: {state.active_agents}")
        return state

    async def _execute_agents(self, state: SupervisorState) -> SupervisorState:
        state.agent_results = {}
        print(f"[SUPERVISOR] Executing agents: {state.active_agents}")
        calls = {
            "USER_PROFILE": lambda: self.user_profile_agent.invoke(prompt=state.current_message),
            "RESUME_MATCHER": lambda: self.resume_matcher.invoke(job_description=state.current_message),
            "RESUME_ENHANCER": lambda: self.resume_enhancer.invoke(prompt=state.current_message),
            "JOB_SEARCH": lambda: self.job_search_agent.invoke(prompt=state.current_message),
            "USER_SEARCH": lambda: self.user_search_agent.invoke(prompt=state.current_message),
            "CHAT": lambda: handle_chat(
                message=state.current_message,
                conversation_history=state.conversation_history,
                files=state.files,
            ),
        }

        # Create pairs of (agent_name, task_coroutine) for tracking
        agent_tasks = [(agent, calls[agent]()) for agent in state.active_agents if agent in calls]

        # Extract just the coroutines for asyncio.gather
        tasks = [task for _, task in agent_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Pair agent names with results
        for (agent, _), result in zip(agent_tasks, results):
            state.agent_results[agent.lower()] = result if not isinstance(result, Exception) else AgentResponse(answer="Error occurred", error=str(result))

        return state

    async def _update_conversation(self, state: SupervisorState) -> Dict[str, Any]:
        ts = int(datetime.utcnow().timestamp())

        # Ensure we have a valid final_response
        if state.final_response is None:
            state.final_response = "No response generated"

        state.conversation_history.append(Message(role="user", content=state.current_message, timestamp=ts))
        state.conversation_history.append(
            Message(
                role="assistant",
                content=state.final_response,  # This should now always be a string
                timestamp=ts,
                activeAgents=state.active_agents,
            )
        )
        return state.__dict__

    async def _stream_synthesize_response(self, state: SupervisorState) -> AsyncGenerator[str, None]:
        serialised = {k: (v.model_dump_json() if hasattr(v, "model_dump_json") else json.dumps(v)) for k, v in state.agent_results.items()}
        prompt = f"""
        You are the **Supervisor Agent**. Your task is to merge the specialist-agent outputs into a single, user-facing reply in concise Markdown.

        Formatting rules
        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        1. Write clear, scannable Markdown—use headings, bullet lists, and short paragraphs.
        2. Preserve every hyperlink exactly as provided by the agents; do **not** invent URLs.
        3. If multiple agents respond, integrate their information logically, remove duplicates, and resolve conflicts.
        4. If an agent returned an error, add a one-line **Note:** summarising it.
        5. Do **not** reveal internal agent names or system details.
        6. End with a "**Next&nbsp;steps**" bullet list (2–3 items) suggesting relevant follow-up actions.
        7. If the combined data is empty, apologise briefly and ask the user for clarification.
        8. Always remove duplicate or irrelevant information.

        Scope of response
        ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        • Answer only from the structured JSON below—labelled AGENT_OUTPUTS.  
        • Be succinct: aim for <= 150 words unless the user explicitly requested depth.

        USER_MESSAGE
        ------------
        {state.current_message}

        AGENT_OUTPUTS
        -------------
        {json.dumps(serialised, indent=2)}
        """
        buffer = ""  # keep track of what we've already sent
        async for chunk in self.llm.generate_stream("Assistant", prompt):
            # ----------  NEWLINES GUARD‑RAIL  ----------
            if chunk.startswith("- ") and not buffer.endswith("\n"):
                chunk = "\n" + chunk  # ensure list starts on its own line
            # bold delimiters sometimes appear with a spurious space before the
            # closing ** → strip that out as well
            if chunk.endswith(" **"):
                chunk = chunk[:-1]  # remove that extra space
            # -------------------------------------------

            buffer += chunk
            state.final_response = buffer  # Update final_response with each chunk
            yield chunk  # keep the stream flowing

    def _create_workflow(self) -> CompiledStateGraph:
        builder = StateGraph(SupervisorState)
        builder.add_node("analyze", self._analyze_message)
        builder.add_node("execute", self._execute_agents)
        builder.add_node("update", self._update_conversation)
        builder.add_edge(START, "analyze")
        builder.add_edge("analyze", "execute")
        builder.add_edge("execute", "update")
        builder.add_edge("update", END)
        return builder.compile()
