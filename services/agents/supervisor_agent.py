from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

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


class AgentType(Enum):
    """Types of agents that the supervisor can delegate to"""
    CHAT = "chat"
    RESUME_MATCHER = "resume_matcher"
    RESUME_ENHANCER = "resume_enhancer"
    USER_PROFILE = "user_profile"


@dataclass
class SupervisorState:
    """State for the supervisor agent workflow"""
    user_id: str
    current_message: str
    conversation_history: List[Message]
    files: Optional[List[Dict[str, Any]]] = None
    selected_agent: Optional[AgentType] = None
    agent_response: Optional[str] = None
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
            selected_agent=None,
            agent_response=None,
            agent_params=None,
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
                "response": result["agent_response"],
                "conversation": updated_history,
                "selected_agent": result["selected_agent"].value if result["selected_agent"] else None,
                "error": result["error"]
            }
        except Exception as e:
            return {
                "response": None,
                "conversation": conversation_history,
                "selected_agent": None,
                "error": str(e)
            }

    async def _route_message(self, state: SupervisorState) -> Dict[str, Any]:
        """
        Analyze the message and determine which agent should handle it

        Args:
            state: Current supervisor state

        Returns:
            Updated state with selected agent and parameters
        """
        # Create a prompt for the LLM to analyze the message
        conversation_context = "\n".join([
            f"{msg.role.upper()}: {msg.content}"
            for msg in state.conversation_history[-5:]  # Use last 5 messages for context
        ])

        routing_prompt = f"""
        Analyze the conversation and the current message to determine which specialized agent should handle it.

        CONVERSATION HISTORY:
        {conversation_context}

        CURRENT MESSAGE:
        {state.current_message}

        FILE ATTACHMENTS: {'Yes' if state.files else 'No'}

        Select the most appropriate agent:
        1. RESUME_MATCHER - For matching resumes against job descriptions, analyzing fit, etc.
        2. RESUME_ENHANCER - For improving existing resumes, suggesting optimizations, etc.
        3. USER_PROFILE - For questions about the user's own resume, skills, projects, or personal information
        4. CHAT - For general conversation, questions, and anything not specifically for the other agents

        Important rules:
        - If the user is asking about their own resume content, skills, projects, or personal information, use USER_PROFILE
        - If the user is asking to match their resume to a job, use RESUME_MATCHER
        - If the user is asking for resume improvements, use RESUME_ENHANCER
        - Use CHAT only for general questions not related to the user's specific resume or profile

        Format your response as a JSON object with:
        {{
            "agent": "RESUME_MATCHER|RESUME_ENHANCER|USER_PROFILE|CHAT",
            "reasoning": "Brief explanation of your selection",
            "parameters": {{
                // Any parameters needed for the agent
                // For profile agent, extract the specific question being asked
                "question": "What are my skills?" // If using USER_PROFILE
            }}
        }}
        """

        response = await self.llm.generate(
            system_prompt="You are an intelligent routing system for a resume improvement platform.",
            user_message=routing_prompt,
            response_format=ResponseFormat.JSON
        )

        if not response.success:
            return {
                "error": f"Failed to route message: {response.error}",
                "selected_agent": AgentType.CHAT  # Default to chat for fallback
            }

        try:
            # Parse agent decision
            decision = response.content
            agent_name = decision.get("agent", "CHAT")
            agent_params = decision.get("parameters", {})

            # Map to enum
            agent_map = {
                "RESUME_MATCHER": AgentType.RESUME_MATCHER,
                "RESUME_ENHANCER": AgentType.RESUME_ENHANCER,
                "USER_PROFILE": AgentType.USER_PROFILE,
                "CHAT": AgentType.CHAT
            }

            selected_agent = agent_map.get(agent_name, AgentType.CHAT)

            return {
                "selected_agent": selected_agent,
                "agent_params": agent_params
            }
        except Exception as e:
            return {
                "error": f"Error parsing agent decision: {str(e)}",
                "selected_agent": AgentType.CHAT  # Default to chat for fallback
            }

    async def _execute_agent(self, state: SupervisorState) -> Dict[str, Any]:
        """
        Execute the selected agent with appropriate parameters

        Args:
            state: Current supervisor state with selected agent

        Returns:
            Updated state with agent response
        """
        agent_type = state.selected_agent
        params = state.agent_params or {}

        print(f"[SUPERVISOR] Executing agent: {agent_type}")
        print(f"[SUPERVISOR] User message: {state.current_message[:50]}...")
        print(f"[SUPERVISOR] Files: {len(state.files) if state.files else 0}")

        try:
            if agent_type == AgentType.USER_PROFILE:
                print(f"[SUPERVISOR] Using user profile agent for user: {state.user_id}")

                # Extract the specific question from params or use the whole message
                question = params.get("question", state.current_message)

                # Call the user profile agent to analyze the question
                result = await self.user_profile_agent.process_user_query(
                    user_id=state.user_id,
                    question=question
                )

                if "error" in result and result.get("answer") is None:
                    return {
                        "agent_response": f"I couldn't analyze your profile: {result['error']}"
                    }

                return {"agent_response": result["answer"]}
            elif agent_type == AgentType.RESUME_MATCHER:
                resume_file = next((f for f in (state.files or []) if f.get("type") == "resume"), None)
                job_description = params.get("job_description", "")

                print(f"[SUPERVISOR] Resume matcher - Job description: {job_description[:50]}...")
                print(f"[SUPERVISOR] Resume matcher - Resume file found: {resume_file is not None}")

                # If no job description was extracted by the router, try to get it from the message
                if not job_description:
                    job_description = state.current_message

                try:
                    if not resume_file:
                        # Try to fetch the user's resume from Pinecone
                        print(f"[SUPERVISOR] No resume file provided, fetching from Pinecone for user: {state.user_id}")
                        user_resume = await get_user_resume(index=self.pc.Index("resumes"), user_id=state.user_id)

                        if not user_resume or not user_resume.get("text"):
                            return {
                                "agent_response": "I need a resume file to perform matching. Please upload a PDF resume."}

                        # Create a simulated resume file with the text from Pinecone
                        print(f"[SUPERVISOR] Using stored resume text for matching")
                        result = await self.resume_matcher.analyze_resume_text(
                            resume_text=user_resume["text"],
                            job_description=job_description
                        )
                    else:
                        # Call resume matcher agent with the uploaded file
                        result = await self.resume_matcher.analyze_resume(
                            resume_bytes=resume_file["bytes"],
                            job_description=job_description
                        )

                    print(f"[SUPERVISOR] Resume matcher results: {result}")

                    # Format the response
                    response = (
                        f"Resume Match Results:\n"
                        f"- Match Score: {result['match_score']:.2f}\n"
                        f"- Keyword Coverage: {result['keyword_coverage']:.2f}\n"
                        f"- Missing Keywords: {', '.join(result['missing_keywords'])}\n\n"
                        f"Would you like me to suggest improvements based on this analysis?"
                    )

                    return {"agent_response": response}

                except Exception as e:
                    print(f"[SUPERVISOR] Error in resume matching: {str(e)}")
                    return {"agent_response": f"I encountered an error while analyzing your resume: {str(e)}"}

            elif agent_type == AgentType.RESUME_ENHANCER:
                print(f"[SUPERVISOR] Resume enhancer")

                # Get user ID from state
                user_id = state.user_id

                # Call resume enhancer agent
                result = await self.resume_enhancer.enhance_resume(
                    user_id=user_id,
                )

                print(f"[SUPERVISOR] Resume enhancer results: {result['content_quality_score']}")

                # Format all issues and suggestions
                ats_issues = "\n".join([
                    f"- {issue['impact'].upper()}: {issue['issue']} - {issue['recommendation']}"
                    for issue in result["ats_issues"]
                ])

                content_issues = "\n".join([
                    f"- {issue['impact'].upper()}: {issue['issue']}\nRecommendation: {issue['recommendation']}\nExample: {issue['example']}"
                    for issue in result["content_issues"]
                ])

                formatting_issues = "\n".join([
                    f"- {issue['impact'].upper()}: {issue['issue']} - {issue['recommendation']}"
                    for issue in result["formatting_issues"]
                ])

                suggestions = "\n".join([
                    f"- {s['priority'].upper()}: {s['description']} ({s['section']})\n  Examples: {', '.join(s['examples'])}"
                    for s in result["enhancement_suggestions"]
                ])

                response = (
                    f"Resume Analysis Results\n"
                    f"======================\n\n"
                    f"Overall Content Score: {result['content_quality_score']:.2f}\n"
                    f"ATS Compatibility Issues:\n{ats_issues}\n\n"
                    f"Content Quality Issues:\n{content_issues}\n\n"
                    f"Formatting Issues:\n{formatting_issues}\n\n"
                    f"Key Enhancement Suggestions:\n{suggestions}\n\n"
                    f"Would you like me to explain any of these suggestions in more detail?"
                )

                return {"agent_response": response}

            else:  # Default to chat
                print(f"[SUPERVISOR] Using chat handler with message: {state.current_message[:50]}...")
                # Handle general chat - pass the llm instance
                response = await handle_chat(
                    llm=self.llm,
                    message=state.current_message,
                    conversation_history=state.conversation_history,
                    files=state.files
                )
                print(f"[SUPERVISOR] Chat handler response: {response}")
                return {"agent_response": response}

        except Exception as e:
            print(f"[SUPERVISOR] Error executing agent: {str(e)}")
            return {
                "error": f"Error executing agent: {str(e)}",
                "agent_response": f"I encountered an error while processing your request: {str(e)}"
            }

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
        if state.agent_response:
            state.conversation_history.append(
                Message(
                    role="assistant",
                    content=state.agent_response,
                    timestamp=timestamp,
                    metadata={
                        "agent": state.selected_agent.value if state.selected_agent else "unknown"
                    }
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
        builder.add_node("route_message", self._route_message)
        builder.add_node("execute_agent", self._execute_agent)
        builder.add_node("update_conversation", self._update_conversation)

        # Define workflow edges
        builder.add_edge(START, "route_message")
        builder.add_edge("route_message", "execute_agent")
        builder.add_edge("execute_agent", "update_conversation")
        builder.add_edge("update_conversation", END)

        # Compile the workflow
        workflow = builder.compile()
        print("Supervisor Agent Workflow:")
        print(workflow.get_graph().draw_ascii())

        return workflow
