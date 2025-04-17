import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Flag
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pinecone import Pinecone

from models.chat import Message
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.tools.chat_handler import handle_chat
from services.agents.tools.get_user_resume import get_user_resume
from services.agents.user_profile_agent import UserProfileAgent
from services.gemini import GeminiLLM
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


@dataclass
class SupervisorState:
    """State for the supervisor agent workflow"""
    user_id: str
    current_message: str
    resume_text: str
    conversation_history: List[Message]
    files: Optional[List[Dict[str, Any]]] = None
    active_agents: AgentFlags = AgentFlags.NONE
    agent_results: Dict[str, Any] = None
    final_response: Optional[str] = None
    error: Optional[str] = None


class SupervisorAgent:
    """
    Supervisor agent that manages conversation flow and delegates to specialized agents
    """

    def __init__(
        self,
        llm: GeminiLLM,
        pc: Pinecone,
        resume_matcher: ResumeMatchingAgent,
        resume_enhancer: ResumeEnhancementAgent,
        user_profile_agent: UserProfileAgent,
    ):
        self.llm = llm
        self.pc = pc
        self.resume_matcher = resume_matcher
        self.resume_enhancer = resume_enhancer
        self.user_profile_agent = user_profile_agent
        self.workflow = self._create_workflow()

    async def process_message(
        self,
        user_id: str,
        message: str,
        conversation_history: List[Dict[str, Any]],
        files: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        # Convert conversation history to Message objects
        history = [
            Message(
                role=item["role"],
                content=item["content"],
                timestamp=item["timestamp"],
                metadata=item.get("metadata", {})
            )
            for item in conversation_history
        ]

        # Pre-fetch the user's resume text
        resume_data = await get_user_resume(index=self.pc.Index("resumes"), user_id=user_id)
        resume_text = resume_data.get("text", "")

        # Initialize workflow state
        initial_state = SupervisorState(
            user_id=user_id,
            current_message=message,
            resume_text=resume_text,
            conversation_history=history,
            files=files,
            active_agents=AgentFlags.NONE,
            agent_results={},
            final_response=None,
            error=None
        )

        try:
            result = await self.workflow.ainvoke(initial_state)
            new_history = result["conversation_history"]
            updated_history = [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp, "metadata": msg.metadata}
                for msg in new_history
            ]

            print("Final Response:", result["final_response"])
            return {
                "response": result["final_response"],
                "conversation": updated_history,
                "active_agents": result["active_agents"].name if result.get("active_agents") else None,
                "error": result.get("error")
            }

        except Exception as e:
            return {"response": None, "conversation": conversation_history, "active_agents": None, "error": str(e)}

    async def _analyze_message(self, state: SupervisorState) -> Dict[str, Any]:
        # Determine which agents to run based on the user message and resume context
        active = AgentFlags.NONE
        msg = state.current_message.lower()
        if "match" in msg or "fit" in msg:
            active |= AgentFlags.RESUME_MATCHER
        if "improve" in msg or "enhance" in msg or "optimize" in msg:
            active |= AgentFlags.RESUME_ENHANCER
        if "who am i" in msg or "my skills" in msg or "profile" in msg:
            active |= AgentFlags.USER_PROFILE
        if active == AgentFlags.NONE:
            active = AgentFlags.CHAT
        return {"active_agents": active}

    async def _execute_agents(self, state: SupervisorState) -> Dict[str, Any]:
        active_agents = state.active_agents
        agent_results: Dict[str, Any] = {}
        resume_text = state.resume_text

        async def run_user_profile():
            return await self.user_profile_agent.process_user_query(
                resume_text=resume_text,
                prompt=state.current_message
            )

        async def run_matcher():
            return await self.resume_matcher.analyze_resume_text(
                resume_text=resume_text,
                job_description=state.current_message
            )

        async def run_enhancer():
            return await self.resume_enhancer.enhance_resume(
                resume_text=resume_text,
                prompt=state.current_message
            )

        async def run_chat():
            resp = await handle_chat(
                llm=self.llm,
                message=state.current_message,
                conversation_history=state.conversation_history,
                files=state.files
            )
            return {"response": resp}

        # Build task list based on flags
        tasks = []
        if AgentFlags.USER_PROFILE in active_agents:
            tasks.append(("user_profile", run_user_profile()))
        if AgentFlags.RESUME_MATCHER in active_agents:
            tasks.append(("resume_matcher", run_matcher()))
        if AgentFlags.RESUME_ENHANCER in active_agents:
            tasks.append(("resume_enhancer", run_enhancer()))
        if AgentFlags.CHAT in active_agents or not tasks:
            tasks.append(("chat", run_chat()))

        # Execute in parallel
        results = await asyncio.gather(*(t for _, t in tasks), return_exceptions=True)
        for (key, _), result in zip(tasks, results):
            agent_results[key] = ("error" if isinstance(result, Exception) else result)

        return {"agent_results": agent_results}

    async def _synthesize_response(self, state: SupervisorState) -> Dict[str, Any]:
        """
        Synthesize the results from multiple agents into a coherent response focused on the user's query

        Args:
            state: Current supervisor state with agent results

        Returns:
            Updated state with final focused response
        """
        active_agents = state.active_agents
        agent_results = state.agent_results
        user_query = state.current_message

        # If there was an error and no agent results, return the error
        if state.error and not agent_results:
            return {"final_response": f"I'm sorry, I encountered an error: {state.error}"}

        # If only the chat agent was used, return its response directly
        if active_agents == AgentFlags.CHAT and "chat" in agent_results:
            chat_result = agent_results["chat"]
            if "error" in chat_result:
                return {"final_response": f"I'm sorry, I encountered an error: {chat_result['error']}"}
            return {"final_response": chat_result["response"]}

        # For complex responses requiring synthesis from multiple agents
        synthesis_prompt = f"""
        Synthesize the following agent results into a well-formatted markdown response that directly addresses the user's query:

        USER QUERY:
        {user_query}

        Focus on providing a response that actually answers what the user is asking. Only include information that is relevant to their specific question.
        """

        # Add relevant results from each agent to the prompt, but keeping focus on the user's query
        if "user_profile" in agent_results:
            profile_result = agent_results["user_profile"]
            if "error" not in profile_result:
                synthesis_prompt += f"\nUSER PROFILE INFORMATION:\n{profile_result.get('answer', '')}\n"

        if "resume_matcher" in agent_results:
            matcher_result = agent_results["resume_matcher"]
            if "error" not in matcher_result:
                synthesis_prompt += "\nRESUME MATCHER RESULTS:\n"
                # Include only the most relevant parts for job matching
                synthesis_prompt += f"- Match Score: {matcher_result.get('match_score')}\n"
                if "missing_keywords" in matcher_result and matcher_result["missing_keywords"]:
                    synthesis_prompt += f"- Missing Keywords: {', '.join(matcher_result.get('missing_keywords', []))}\n"

        if "resume_enhancer" in agent_results:
            enhancer_result = agent_results["resume_enhancer"]
            if "error" not in enhancer_result:
                # Check what areas the user was asking about
                query_context = enhancer_result.get("query_context", "")
                focus_areas = enhancer_result.get("focus_areas", [])

                synthesis_prompt += "\nRESUME ENHANCER RESULTS:\n"

                # Only include information about areas the user asked about
                if "ats" in query_context.lower() and "ats_issues" in enhancer_result:
                    synthesis_prompt += "ATS ISSUES:\n"
                    for issue in enhancer_result["ats_issues"]:
                        synthesis_prompt += f"- {issue.get('impact', 'Medium').upper()}: {issue.get('issue')}\n"
                        synthesis_prompt += f"  Recommendation: {issue.get('recommendation')}\n"

                if any(term in query_context.lower() for term in
                       ["content", "wording", "language"]) and "content_issues" in enhancer_result:
                    synthesis_prompt += "CONTENT ISSUES:\n"
                    for issue in enhancer_result["content_issues"]:
                        synthesis_prompt += f"- {issue.get('impact', 'Medium').upper()}: {issue.get('issue')}\n"
                        synthesis_prompt += f"  Recommendation: {issue.get('recommendation')}\n"

                if any(term in query_context.lower() for term in
                       ["format", "layout", "structure"]) and "formatting_issues" in enhancer_result:
                    synthesis_prompt += "FORMATTING ISSUES:\n"
                    for issue in enhancer_result["formatting_issues"]:
                        synthesis_prompt += f"- {issue.get('impact', 'Medium').upper()}: {issue.get('issue')}\n"
                        synthesis_prompt += f"  Recommendation: {issue.get('recommendation')}\n"

                # Always include enhancement suggestions but filter them by relevance
                if "enhancement_suggestions" in enhancer_result:
                    synthesis_prompt += "ENHANCEMENT SUGGESTIONS:\n"
                    for suggestion in enhancer_result["enhancement_suggestions"]:
                        # Only include suggestions relevant to the focus areas
                        if not focus_areas or suggestion.get("section", "").lower() in [area.lower() for area in
                                                                                        focus_areas]:
                            synthesis_prompt += f"- {suggestion.get('priority', 'Medium').upper()}: {suggestion.get('description')} ({suggestion.get('section')})\n"
                            if "examples" in suggestion:
                                synthesis_prompt += f"  Examples: {', '.join(suggestion.get('examples', []))}\n"

        # Add instructions for synthesis focused on relevance to the user's query
        synthesis_prompt += """
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

        # Generate synthesized response
        response = await self.llm.agenerate(
            system_prompt="You are an expert resume consultant that provides focused, relevant answers to user questions.",
            user_message=synthesis_prompt
        )

        if not response.success:
            # Fallback to a simple response
            final_response = "I'm sorry, I couldn't properly synthesize the results for your specific question."
        else:
            final_response = response.content

        return {"final_response": final_response}

    async def _update_conversation(self, state: SupervisorState) -> Dict[str, Any]:
        ts = datetime.now().isoformat()
        # Append user and assistant messages
        state.conversation_history.append(Message(role="user", content=state.current_message, timestamp=ts))
        state.conversation_history.append(Message(role="assistant", content=state.final_response, timestamp=ts, metadata={"agents": [f.name.lower() for f in AgentFlags if f in state.active_agents and f != AgentFlags.NONE]}))
        return {"conversation_history": state.conversation_history}

    def _create_workflow(self) -> CompiledStateGraph:
        builder = StateGraph(SupervisorState)
        builder.add_node("analyze_message", self._analyze_message)
        builder.add_node("execute_agents", self._execute_agents)
        builder.add_node("synthesize_response", self._synthesize_response)
        builder.add_node("update_conversation", self._update_conversation)
        builder.add_edge(START, "analyze_message")
        builder.add_edge("analyze_message", "execute_agents")
        builder.add_edge("execute_agents", "synthesize_response")
        builder.add_edge("synthesize_response", "update_conversation")
        builder.add_edge("update_conversation", END)
        return builder.compile()


if __name__ == "__main__":
    user_id = "oPmOJhSE0VQid56yYyg19hdH5DV2"

    # Instantiate GeminiLLM
    gemini_llm = GeminiLLM()

    # Create the ResumeEnhancer instance
    supervisor = SupervisorAgent(
        llm=gemini_llm,
        pc=Pinecone(api_key=os.environ["PINECONE_API_KEY"]),
        resume_matcher=ResumeMatchingAgent(resume_parser=ResumeParser(), text_embedder=TextEmbedder(), llm=gemini_llm),
        resume_enhancer=ResumeEnhancementAgent(gemini_llm),
        user_profile_agent=UserProfileAgent(gemini_llm)
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
    files = None  # Replace with actual file data if needed
    message = "What are my skills?"
    response = asyncio.run(supervisor.process_message(user_id, message, conversation_history, files))