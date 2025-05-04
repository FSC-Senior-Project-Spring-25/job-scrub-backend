import asyncio
import json
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Flag
from typing import Dict, List, Optional, Any, AsyncGenerator

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pinecone import Pinecone

from models.chat import Message
from services.agents.base.agent import AgentResponse
from services.agents.job_search_agent import JobSearchAgent
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.tools.chat_handler import handle_chat
from services.agents.tools.get_user_resume import get_user_resume
from services.agents.user_profile_agent import UserProfileAgent
from services.agents.user_search_agent import UserSearchAgent
from services.llm.base.llm import LLM, ResponseFormat
from services.llm.gemini import GeminiLLM
from services.resume_parser import ResumeParser
from services.text_embedder import TextEmbedder

load_dotenv()


class AgentFlags(Flag):
    """Flags for each agent type to allow multiple selections"""
    NONE = 0
    CHAT = 1
    RESUME_MATCHER = 2
    RESUME_ENHANCER = 4
    USER_PROFILE = 8
    JOB_SEARCH = 16
    USER_SEARCH = 32


@dataclass
class SupervisorState:
    """State for the supervisor agent workflow"""
    user_id: str
    current_message: str
    resume_text: str
    conversation_history: List[Message]
    files: Optional[List[Dict[str, Any]]] = None
    active_agents: AgentFlags = AgentFlags.NONE
    agent_results: Dict[str, AgentResponse] = None
    final_response: Optional[str] = None
    error: Optional[str] = None


class SupervisorAgent:
    """
    Supervisor agent that manages conversation flow and delegates to specialized agents
    """

    def __init__(
            self,
            llm: LLM,
            pc: Pinecone,
            resume_matcher: ResumeMatchingAgent,
            resume_enhancer: ResumeEnhancementAgent,
            user_profile_agent: UserProfileAgent,
            job_search_agent: JobSearchAgent,
            user_search_agent: UserSearchAgent,
            resume_data: Optional[Dict[str, Any]] = None,
            processed_conversation_history: Optional[List[Message]] = None,
            processed_files: Optional[List[Dict[str, Any]]] = None
    ):
        self.llm = llm
        self.pc = pc
        self.resume_matcher = resume_matcher
        self.resume_enhancer = resume_enhancer
        self.user_profile_agent = user_profile_agent
        self.job_search_agent = job_search_agent
        self.user_search_agent = user_search_agent
        self.resume_text = resume_data.get("text", "") if resume_data else ""
        self.processed_history = processed_conversation_history or []
        self.processed_files = processed_files or []
        self.workflow = self._create_workflow()

    async def process_message(
            self,
            user_id: str,
            message: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a message with streaming response

        Args:
            message: The user's message
            files: Any uploaded files

        Yields:
            Chunks of the response as they're generated
        """

        # Initialize state
        state = SupervisorState(
            user_id=user_id,
            current_message=message,
            resume_text=self.resume_text,
            conversation_history=self.processed_history,
            files=self.processed_files,
            active_agents=AgentFlags.NONE,
            agent_results={},
            final_response=None,
            error=None
        )

        print(f"Processing message: {message} for user: {user_id}")

        try:
            # 1. Analyze message to determine agents
            routing_result = await self._analyze_message(state)
            active_agents = routing_result["active_agents"]
            state.active_agents = active_agents

            # Yield the agent selection information
            yield {
                "type": "agents_selected",
                "agents": [f.name.lower() for f in AgentFlags if f in active_agents and f != AgentFlags.NONE]
            }

            # 2. Execute agents to get results
            execution_result = await self._execute_agents(state)
            state.agent_results = execution_result["agent_results"]

            # 3. Generate a response - stream when possible
            accumulated_response = []

            # Special handling for chat-only responses
            if active_agents == AgentFlags.CHAT and "chat" in state.agent_results:
                chat_result = state.agent_results["chat"]
                # If there was an error, yield the error
                if isinstance(chat_result, dict) and "error" in chat_result:
                    error_msg = f"I'm sorry, I encountered an error: {chat_result['error']}"
                    yield {
                        "type": "content_chunk",
                        "content": error_msg
                    }
                    accumulated_response.append(error_msg)
                else:
                    # Chat result is now a string in chat_result.answer - no streaming needed
                    response_text = chat_result.answer
                    yield {
                        "type": "content_chunk",
                        "content": response_text
                    }
                    accumulated_response.append(response_text)
            else:
                # For multiple agents or non-chat agents, stream the synthesis
                async for chunk in self._stream_synthesize_response(state):
                    if chunk:
                        yield {
                            "type": "content_chunk",
                            "content": chunk
                        }
                        accumulated_response.append(chunk)

            # Combine accumulated response chunks
            final_response = "".join(accumulated_response)
            state.final_response = final_response

            # 4. Update the conversation history
            result = await self._update_conversation(state)
            updated_history = result["conversation_history"]

            # 5. Yield the complete response with conversation history
            yield {
                "type": "complete",
                "response": final_response,
                "conversation": [
                    {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp, "metadata": msg.metadata}
                    for msg in updated_history
                ],
                "active_agents": active_agents.name if active_agents else None
            }

        except Exception as e:
            print(f"Error in process_message: {str(e)}")
            import traceback
            traceback.print_exc()
            yield {
                "type": "error",
                "error": str(e)
            }

    async def _analyze_message(self, state: SupervisorState) -> Dict[str, Any]:
        """Determine which agents to run based on the user message using LLM analysis"""

        prompt = f"""
        Analyze the following user message and determine which specialized agent(s) should handle it.

        User message: "{state.current_message}"

        Available agents:
        - RESUME_MATCHER: For questions about job fit, matching resumes to jobs, or compatibility with positions
        - RESUME_ENHANCER: For requests to improve, enhance, optimize, or get feedback on resumes
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
            response_format=ResponseFormat.JSON
        )

        active = AgentFlags.NONE

        if response.success and isinstance(response.content, dict) and "agents" in response.content:
            agent_names = response.content["agents"]

            # Map string agent names to AgentFlags
            mapping = {
                "RESUME_MATCHER": AgentFlags.RESUME_MATCHER,
                "RESUME_ENHANCER": AgentFlags.RESUME_ENHANCER,
                "USER_PROFILE": AgentFlags.USER_PROFILE,
                "JOB_SEARCH": AgentFlags.JOB_SEARCH,
                "USER_SEARCH": AgentFlags.USER_SEARCH,
                "CHAT": AgentFlags.CHAT
            }

            # Combine all applicable flags
            for agent_name in agent_names:
                if agent_name in mapping:
                    active |= mapping[agent_name]

        # Fallback to CHAT if no agents were selected or if the LLM response failed
        if active == AgentFlags.NONE:
            active = AgentFlags.CHAT

        print("Active agents determined:", active)
        return {"active_agents": active}

    async def _execute_agents(self, state: SupervisorState) -> Dict[str, Any]:
        active_agents = state.active_agents
        agent_results: Dict[str, AgentResponse] = {}
        resume_text = state.resume_text
        print("Executing agents with active agents:", active_agents)
        print("Current message:", state.current_message)
        async def run_user_profile():
            return await self.user_profile_agent.invoke(
                prompt=state.current_message
            )

        async def run_matcher():
            return await self.resume_matcher.invoke(
                job_description=state.current_message
            )

        async def run_enhancer():
            return await self.resume_enhancer.invoke(
                prompt=state.current_message
            )

        async def run_job_search():
            return await self.job_search_agent.invoke(
                prompt=state.current_message
            )

        async def run_user_search():
            return await self.user_search_agent.invoke(
                prompt=state.current_message
            )

        async def run_chat():
            resp = await handle_chat(
                llm=self.llm,
                message=state.current_message,
                conversation_history=state.conversation_history,
                files=state.files
            )
            return AgentResponse(answer=resp)

        # Map agent flags to their corresponding functions (without calling them yet)
        agent_functions = {
            AgentFlags.USER_PROFILE: ("user_profile", run_user_profile),
            AgentFlags.RESUME_MATCHER: ("resume_matcher", run_matcher),
            AgentFlags.RESUME_ENHANCER: ("resume_enhancer", run_enhancer),
            AgentFlags.JOB_SEARCH: ("job_search", run_job_search),
            AgentFlags.USER_SEARCH: ("user_search", run_user_search),
        }

        # Build task list
        tasks = [(key, func()) for key, func in
                 [agent_functions[flag] for flag in agent_functions if flag in active_agents]]

        # Add chat task if no other tasks or if explicitly requested
        if AgentFlags.CHAT in active_agents or not tasks:
            tasks.append(("chat", run_chat()))

        # Execute in parallel
        results = await asyncio.gather(*(task for _, task in tasks), return_exceptions=True)

        for (key, _), result in zip(tasks, results):
            agent_results[key] = result if not isinstance(result, Exception) else {"error": str(result)}

        return {"agent_results": agent_results}

    async def _stream_synthesize_response(self, state: SupervisorState) -> AsyncGenerator[str, None]:
        """
        Stream the synthesis of multiple agent results

        Args:
            state: Current supervisor state with agent results

        Yields:
            Chunks of the synthesized response
        """
        active_agents = state.active_agents
        agent_results = state.agent_results
        user_query = state.current_message
        print("Streaming synthesis for user query:", user_query)
        print("Active agents for synthesis:", active_agents)
        print("Agent results for synthesis:", agent_results)
        # If there was an error and no agent results, return the error
        if state.error and not agent_results:
            yield f"I'm sorry, I encountered an error: {state.error}"
            return

        # Convert AgentResponse objects to dictionaries for JSON serialization
        try:
            serialized_results = {}
            for key, response in agent_results.items():
                if hasattr(response, "model_dump_json"):
                    serialized_results[key] = response.model_dump_json()
                elif hasattr(response, "dict"):
                    serialized_results[key] = json.dumps(response.dict())
                else:
                    serialized_results[key] = json.dumps(response)
        except Exception as e:
            serialized_results = {"error": f"Error serializing results: {str(e)}"}

        synthesis_prompt = f"""
        Synthesize the following agent results into a well-formatted markdown response that directly addresses the user's query:

        USER QUERY:
        {user_query}

        RESULTS:
        {json.dumps(serialized_results, indent=2)}

        Focus on providing a response that actually answers what the user is asking. Only include information that is relevant to their specific question.
        Guidelines for response formatting:
        1. Use proper markdown formatting with headers, lists, and emphasis
        2. Start with a direct answer to the user's specific question
        3. Include ONLY information relevant to what the user asked about
        4. Use bullet points for lists of issues or suggestions
        5. Keep the response concise and focused
        6. Make sure to directly address what the user was asking
        7. Don't include information that wasn't asked for

        Important: The response should be well-structured, focused only on answering what the user asked,
        and appear as if it came from a single intelligent assistant.
        """

        # Stream the synthesized response
        async for chunk in self.llm.generate_stream(
                system_prompt="You are an expert resume consultant that provides focused, relevant answers to user questions.",
                user_message=synthesis_prompt
        ):
            if chunk:
                yield chunk

    async def _update_conversation(self, state: SupervisorState) -> Dict[str, Any]:
        ts = datetime.now().isoformat()
        # Append user and assistant messages
        state.conversation_history.append(Message(role="user", content=state.current_message, timestamp=ts))
        state.conversation_history.append(Message(role="assistant", content=state.final_response, timestamp=ts, metadata={
            "agents": [f.name.lower() for f in AgentFlags if f in state.active_agents and f != AgentFlags.NONE]}))
        return {"conversation_history": state.conversation_history}

    def _create_workflow(self) -> CompiledStateGraph:
        builder = StateGraph(SupervisorState)
        builder.add_node("analyze_message", self._analyze_message)
        builder.add_node("execute_agents", self._execute_agents)
        builder.add_node("update_conversation", self._update_conversation)
        builder.add_edge(START, "analyze_message")
        builder.add_edge("analyze_message", "execute_agents")
        builder.add_edge("execute_agents", "update_conversation")
        builder.add_edge("update_conversation", END)
        return builder.compile()


if __name__ == "__main__":
    user_id = "oPmOJhSE0VQid56yYyg19hdH5DV2"
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    # Instantiate GeminiLLM
    gemini_llm = GeminiLLM()
    resume_data = asyncio.run(get_user_resume(index=pc.Index("resumes"), user_id=user_id))
    # Create the ResumeEnhancer instance
    supervisor = SupervisorAgent(
        llm=gemini_llm,
        pc=pc,
        resume_matcher=ResumeMatchingAgent(resume_parser=ResumeParser(), embedder=TextEmbedder(), llm=gemini_llm),
        resume_enhancer=ResumeEnhancementAgent(gemini_llm),
        user_profile_agent=UserProfileAgent(gemini_llm),
        job_search_agent=JobSearchAgent(gemini_llm, pc.Index("job-postings"), resume_data["vector"]),
        user_search_agent=UserSearchAgent(gemini_llm, pc.Index("resumes"), resume_data["vector"])
    )

    # Example usage
    conversation_history = [
        {
            "role": "user",
            "content": "What are my skills?",
            "timestamp": datetime.now().isoformat(),
            "metadata": {}
        }
    ]


    # For testing the streaming functionality
    async def test_streaming():
        message = "Search for users with Python skills"
        async for chunk in supervisor.process_message(user_id, message, conversation_history, None):
            print(f"Received chunk: {chunk}")


    # Run the test
    asyncio.run(test_streaming())