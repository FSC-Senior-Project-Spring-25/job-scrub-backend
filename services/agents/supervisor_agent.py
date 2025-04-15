import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, Flag
from typing import Dict, List, Optional, Any, Set

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pinecone import Pinecone

from models.chat import Message
from services.agents.resume_enhancer import ResumeEnhancementAgent
from services.agents.resume_matcher import ResumeMatchingAgent
from services.agents.tools.chat_handler import handle_chat
from services.agents.tools.get_user_resume import get_user_resume
from services.agents.user_profile_agent import UserProfileAgent
from services.gemini import GeminiLLM, ResponseFormat


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
    conversation_history: List[Message]
    files: Optional[List[Dict[str, Any]]] = None
    active_agents: Optional[AgentFlags] = AgentFlags.NONE
    agent_results: Optional[Dict[str, Any]] = None
    final_response: Optional[str] = None
    agent_params: Optional[Dict[str, Any]] = None
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
        """
        Process a new user message and determine appropriate handling

        Args:
            user_id: User identifier
            message: Current user message
            conversation_history: Previous conversation messages
            files: Optional files attached to the message

        Returns:
            Dictionary with agent response and updated conversation
        """
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

        # Initialize workflow state
        initial_state = SupervisorState(
            user_id=user_id,
            current_message=message,
            conversation_history=history,
            files=files,
            active_agents=AgentFlags.NONE,
            agent_results={},
            final_response=None,
            agent_params={},
            error=None
        )

        # Run the workflow
        try:
            result = await self.workflow.ainvoke(initial_state)

            # Update conversation history
            new_history = result["conversation_history"]

            # Convert Message objects back to dictionaries
            updated_history = [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in new_history
            ]

            return {
                "response": result["final_response"],
                "conversation": updated_history,
                "active_agents": result["active_agents"].name if result["active_agents"] else None,
                "error": result["error"]
            }
        except Exception as e:
            return {
                "response": None,
                "conversation": conversation_history,
                "active_agents": None,
                "error": str(e)
            }

    async def _analyze_message(self, state: SupervisorState) -> Dict[str, Any]:
        """
        Analyze the message and determine which agents should handle it

        Args:
            state: Current supervisor state

        Returns:
            Updated state with selected agents and parameters
        """
        # Create a prompt for the LLM to analyze the message
        conversation_context = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in state.conversation_history[-5:]  # Use last 5 messages for context
        ])

        routing_prompt = f"""
        Analyze the conversation and the current message to determine which specialized agents should handle it.
        You can select MULTIPLE agents if the request requires their combined expertise.

        CONVERSATION HISTORY:
        {conversation_context}

        CURRENT MESSAGE:
        {state.current_message}

        FILE ATTACHMENTS: {'Yes' if state.files else 'No'}

        Available agents:
        1. RESUME_MATCHER - For matching resumes against job descriptions, analyzing fit, etc.
        2. RESUME_ENHANCER - For improving existing resumes, suggesting optimizations, etc.
        3. USER_PROFILE - For questions about the user's own resume, skills, projects, or personal information
        4. CHAT - For general conversation, questions, and anything not specifically for the other agents

        Critical: Select ALL agents needed to fully address the request. Many requests may require multiple agents working together.
        For example:
        - "Match my resume to this job and suggest improvements" would need BOTH resume_matcher AND resume_enhancer
        - "How can I improve my resume's skills section based on my project experience?" would need BOTH user_profile AND resume_enhancer

        Format your response as a JSON object with:
        {{
            "selected_agents": ["CHAT", "RESUME_MATCHER", "RESUME_ENHANCER", "USER_PROFILE"],  // Include ALL needed agents
            "reasoning": "Detailed explanation justifying each selected agent's involvement",
            "execution_order": ["AGENT1", "AGENT2", ...],  // Optimal order for execution
            "parameters": {{
                // Parameters for each agent
                "RESUME_MATCHER": {{
                    "job_description": "extracted job description if present"
                }},
                "RESUME_ENHANCER": {{
                    "focus_areas": ["skills", "experience"] // Specific areas to enhance
                }},
                "USER_PROFILE": {{
                    "question": "What are my skills?" // Specific question about user profile
                }}
            }}
        }}
        """

        response = await self.llm.generate(
            system_prompt="You are an intelligent routing system for a resume improvement platform. You analyze requests to determine which specialized agents should handle different parts of the request.",
            user_message=routing_prompt,
            response_format=ResponseFormat.JSON
        )

        if not response.success:
            return {
                "error": f"Failed to route message: {response.error}",
                "active_agents": AgentFlags.CHAT  # Default to chat for fallback
            }

        try:
            # Parse agent decision
            decision = response.content
            selected_agents = decision.get("selected_agents", ["CHAT"])
            agent_params = decision.get("parameters", {})
            execution_order = decision.get("execution_order", selected_agents)

            # Convert string agent names to flags
            active_agents = AgentFlags.NONE
            if "CHAT" in selected_agents:
                active_agents |= AgentFlags.CHAT
            if "RESUME_MATCHER" in selected_agents:
                active_agents |= AgentFlags.RESUME_MATCHER
            if "RESUME_ENHANCER" in selected_agents:
                active_agents |= AgentFlags.RESUME_ENHANCER
            if "USER_PROFILE" in selected_agents:
                active_agents |= AgentFlags.USER_PROFILE

            return {
                "active_agents": active_agents,
                "agent_params": agent_params,
                "execution_order": execution_order
            }
        except Exception as e:
            return {
                "error": f"Error parsing agent decision: {str(e)}",
                "active_agents": AgentFlags.CHAT  # Default to chat for fallback
            }

    async def _execute_agents(self, state: SupervisorState) -> Dict[str, Any]:
        """
        Execute all selected agents in parallel and collect their results

        Args:
            state: Current supervisor state with selected agents

        Returns:
            Updated state with agent results
        """
        active_agents = state.active_agents
        params = state.agent_params or {}
        agent_results = {}

        print(f"[SUPERVISOR] Executing agents: {active_agents}")
        print(f"[SUPERVISOR] User message: {state.current_message[:50]}...")
        print(f"[SUPERVISOR] Files: {len(state.files) if state.files else 0}")

        try:
            # Get user resume data - used by multiple agents
            user_resume = None
            resume_file = next((f for f in (state.files or []) if f.get("type") == "resume"), None)

            if not resume_file:
                # Try to fetch the user's resume from Pinecone if needed by any agent
                if (AgentFlags.RESUME_MATCHER in active_agents or
                        AgentFlags.RESUME_ENHANCER in active_agents):
                    print(f"[SUPERVISOR] Fetching resume from Pinecone for user: {state.user_id}")
                    user_resume = await get_user_resume(index=self.pc.Index("resumes"), user_id=state.user_id)

            # Define coroutines for each agent
            async def run_user_profile_agent():
                try:
                    # Extract the specific question from params or use the whole message
                    profile_params = params.get("USER_PROFILE", {})
                    question = profile_params.get("question", state.current_message)

                    # Call the user profile agent
                    return await self.user_profile_agent.process_user_query(
                        user_id=state.user_id,
                        question=question
                    )
                except Exception as e:
                    return {"error": str(e)}

            async def run_resume_matcher_agent():
                try:
                    matcher_params = params.get("RESUME_MATCHER", {})
                    job_description = matcher_params.get("job_description", "")

                    # If no job description was extracted, try to get it from the message
                    if not job_description:
                        job_description = state.current_message

                    if resume_file:
                        # Use uploaded resume file
                        return await self.resume_matcher.analyze_resume(
                            resume_bytes=resume_file["bytes"],
                            job_description=job_description
                        )
                    elif user_resume and user_resume.get("text"):
                        # Use resume text from Pinecone
                        return await self.resume_matcher.analyze_resume_text(
                            resume_text=user_resume["text"],
                            job_description=job_description
                        )
                    else:
                        return {"error": "No resume found for matching"}
                except Exception as e:
                    return {"error": str(e)}

            async def run_resume_enhancer_agent():
                try:
                    enhancer_params = params.get("RESUME_ENHANCER", {})
                    focus_areas = enhancer_params.get("focus_areas", [])

                    # Call resume enhancer agent
                    return await self.resume_enhancer.enhance_resume(
                        user_id=state.user_id
                    )
                except Exception as e:
                    return {"error": str(e)}

            async def run_chat_agent():
                try:
                    # Handle general chat
                    chat_response = await handle_chat(
                        llm=self.llm,
                        message=state.current_message,
                        conversation_history=state.conversation_history,
                        files=state.files
                    )
                    return {"response": chat_response}
                except Exception as e:
                    return {"error": str(e)}

            # Prepare tasks to run in parallel based on active agents
            tasks = []
            task_names = []

            if AgentFlags.USER_PROFILE in active_agents:
                tasks.append(run_user_profile_agent())
                task_names.append("user_profile")

            if AgentFlags.RESUME_MATCHER in active_agents:
                tasks.append(run_resume_matcher_agent())
                task_names.append("resume_matcher")

            if AgentFlags.RESUME_ENHANCER in active_agents:
                tasks.append(run_resume_enhancer_agent())
                task_names.append("resume_enhancer")

            if AgentFlags.CHAT in active_agents or not tasks:  # Run chat agent if selected or as fallback
                tasks.append(run_chat_agent())
                task_names.append("chat")

            # Execute all tasks in parallel
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        agent_results[task_names[i]] = {"error": str(result)}
                    else:
                        agent_results[task_names[i]] = result

            return {"agent_results": agent_results}

        except Exception as e:
            print(f"[SUPERVISOR] Error executing agents: {str(e)}")
            return {
                "error": f"Error executing agents: {str(e)}",
                "agent_results": {"error": str(e)}
            }

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
        response = await self.llm.generate(
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
        """
        Update conversation history with the new message and response

        Args:
            state: Current supervisor state

        Returns:
            Updated state with new conversation history
        """
        # Create timestamp for new messages
        timestamp = datetime.now().isoformat()

        # Add the user message to history if not already there
        if not state.conversation_history or state.conversation_history[-1].role != "user" or \
                state.conversation_history[-1].content != state.current_message:
            state.conversation_history.append(
                Message(
                    role="user",
                    content=state.current_message,
                    timestamp=timestamp
                )
            )

        # Add the assistant response to history
        if state.final_response:
            # Create metadata about which agents were used
            agent_metadata = {}
            if state.active_agents:
                active_agent_names = []
                if AgentFlags.CHAT in state.active_agents:
                    active_agent_names.append("chat")
                if AgentFlags.RESUME_MATCHER in state.active_agents:
                    active_agent_names.append("resume_matcher")
                if AgentFlags.RESUME_ENHANCER in state.active_agents:
                    active_agent_names.append("resume_enhancer")
                if AgentFlags.USER_PROFILE in state.active_agents:
                    active_agent_names.append("user_profile")

                agent_metadata["agents"] = active_agent_names

            state.conversation_history.append(
                Message(
                    role="assistant",
                    content=state.final_response,
                    timestamp=timestamp,
                    metadata=agent_metadata
                )
            )

        return {
            "conversation_history": state.conversation_history
        }

    def _create_workflow(self) -> CompiledStateGraph:
        """
        Create and compile the supervisor workflow graph

        Returns:
            Compiled workflow graph
        """
        builder = StateGraph(SupervisorState)

        # Add workflow nodes
        builder.add_node("analyze_message", self._analyze_message)
        builder.add_node("execute_agents", self._execute_agents)
        builder.add_node("synthesize_response", self._synthesize_response)
        builder.add_node("update_conversation", self._update_conversation)

        # Define workflow edges
        builder.add_edge(START, "analyze_message")
        builder.add_edge("analyze_message", "execute_agents")
        builder.add_edge("execute_agents", "synthesize_response")
        builder.add_edge("synthesize_response", "update_conversation")
        builder.add_edge("update_conversation", END)

        # Compile the workflow
        workflow = builder.compile()
        print("Supervisor Agent Workflow:")
        print(workflow.get_graph().draw_ascii())

        return workflow