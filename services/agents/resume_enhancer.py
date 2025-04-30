import json
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from services.agents.base.agent import ReActAgent, AgentResponse
from services.gemini import ResponseFormat, GeminiLLM


class EnhancementState(MessagesState):
    """
    The agent state for resume enhancement.
    It holds the resume text and conversation messages.
    """
    resume_text: str
    prompt: str
    job_description: Optional[str]
    formatting_issues: Optional[List[Dict[str, Any]]]
    ats_analysis: Optional[Dict[str, Any]]
    content_quality: Optional[Dict[str, Any]]
    improvement_suggestions: Optional[List[Dict[str, Any]]]


class ResumeEnhancementAgent(ReActAgent):
    """
    Agent that analyzes and provides enhancement suggestions for resumes.
    """

    # List of metadata fields to track
    METADATA_FIELDS = [
        "formatting_issues",
        "ats_analysis",
        "content_quality",
        "improvement_suggestions"
    ]

    def __init__(self, llm: GeminiLLM = GeminiLLM()):
        """
        Initializes the ResumeEnhancementAgent.

        Args:
            llm: An instance of GeminiLLM.
        """
        super().__init__(llm)

    def _get_system_prompt(self) -> str:
        """
        Return the system prompt for this agent.
        Implements abstract method from ReActAgent.
        """
        return (
            "You are a resume enhancement assistant focused on targeted analysis. "
            "Choose tools based on the user prompt and the resume content provided. "
            "You do NOT need to execute tools in sequence - only use tools that are directly relevant to the user's request. "
            "You can call the suggestions_tool directly without requiring prior analysis if appropriate."
        )

    async def invoke(self, resume_text: str, prompt: str, job_description: Optional[str] = None) -> AgentResponse:
        """
        Main entry point for agent execution, implementing the abstract method.

        Args:
            resume_text: The resume text to analyze
            prompt: The prompt to guide the analysis
            job_description: Optional job description for ATS analysis

        Returns:
            Standardized AgentResponse
        """
        try:
            # Initialize state
            initial_state = {
                "resume_text": resume_text,
                "prompt": prompt,
                "job_description": job_description,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }
            # Initialize metadata fields
            for field in self.METADATA_FIELDS:
                initial_state[field] = None

            result = await self.workflow.ainvoke(initial_state)
            return self._format_response(result)
        except Exception as e:
            return AgentResponse(
                answer=f"An error occurred during resume enhancement: {str(e)}",
                error=str(e),
                metadata={"agent_type": "resume_enhancer", "error_type": type(e).__name__}
            )

    def _extract_answer(self, state: Dict[str, Any]) -> str:
        """
        Extract final answer from result.
        Overrides method from Agent base class.
        """
        # Use the parent implementation to extract the answer
        answer = super()._extract_answer(state)
        print("Enhancer Final Result:")
        print(answer)
        return answer

    def _extract_metadata(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract metadata from the state.
        Overrides method from Agent base class.

        Returns:
            Dictionary with metadata fields
        """
        metadata = {"agent_type": "resume_enhancer"}

        # Add all tracked fields to metadata
        for field in self.METADATA_FIELDS:
            if field in state:
                metadata[field] = state.get(field)

        return metadata

    def _create_workflow(self) -> CompiledStateGraph:
        """
        Create and return the agent's workflow.
        Implements abstract method from Agent base class.
        """
        # Build the ReAct agent state graph
        builder = StateGraph(EnhancementState)
        builder.add_node("think", self.think)
        builder.add_node("tools", ToolNode(self.tools))
        builder.add_edge(START, "think")
        builder.add_conditional_edges("think", tools_condition)
        builder.add_edge("tools", "think")
        return builder.compile()

    def _create_tools(self) -> List:
        """
        Creates and returns all the tools for the ResumeEnhancer.
        Implements abstract method from ReActAgent.

        Returns:
            List of tools for the ReAct agent.
        """
        tools = [
            self._create_analyze_formatting_tool(),
            self._create_ats_analysis_tool(),
            self._create_content_quality_tool(),
            self._create_suggestions_tool()
        ]
        return tools

    def _create_analyze_formatting_tool(self):
        @tool(parse_docstring=True)
        def analyze_formatting_tool(resume_text: Optional[str] = None) -> Dict[str, Any]:
            """
            Analyze the resume for formatting issues.

            Args:
                resume_text: The complete resume text provided as input

            Returns:
                A dictionary containing formatting issues and messages.
            """
            system_prompt = (
                "You are a resume formatting expert. Analyze the resume text for formatting issues "
                "like inconsistent bullets, irregular spacing, alignment problems. "
                "Format your response as a JSON object with the following structure:\n"
                "{\n"
                '  "formatting_issues": [\n'
                "    {\n"
                '      "type": "string", // Type of issue (bullets, spacing, alignment, etc)\n'
                '      "severity": "string", // high, medium, or low\n'
                '      "location": "string", // Section or area where issue occurs\n'
                '      "description": "string", // Detailed description of the issue\n'
                '      "fix": "string" // Suggested fix for the issue\n'
                "    }\n"
                "  ]\n"
                "}"
            )
            user_message = f"Resume text to analyze:\n{resume_text}"

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                response_format=ResponseFormat.JSON
            )
            print("Response from Gemini (formatting):", response)
            if not response.success:
                raise ValueError(response.error)

            issues = response.content.get("formatting_issues", [])
            return {"formatting_issues": issues}

        return analyze_formatting_tool

    def _create_ats_analysis_tool(self):
        @tool(parse_docstring=True)
        def ats_analysis_tool(resume_text: Optional[str] = None, job_description: Optional[str] = None) -> Dict[
            str, Any]:
            """
            Analyze the resume for ATS compatibility and keyword optimization.

            Args:
                resume_text: The complete resume text provided as input
                job_description: Optional job description to analyze keywords against

            Returns:
                A dictionary containing ATS analysis results.
            """
            system_prompt = (
                "You are an ATS (Applicant Tracking System) expert. Analyze the resume for ATS compatibility "
                "including keyword optimization, scannable format, and parsability. "
                "If a job description is provided, evaluate keyword matching against that description. "
                "If no job description is provided, focus on general ATS best practices. "
                "Format your response as a JSON object with the following structure:\n"
                "{\n"
                '  "ats_analysis": {\n'
                '    "format_compatibility": {\n'
                '      "is_parseable": boolean,\n'
                '      "issues": [\n'
                '        {\n'
                '          "type": "string", // headers, tables, images, fonts, etc\n'
                '          "description": "string",\n'
                '          "impact": "string" // high, medium, low\n'
                '        }\n'
                '      ]\n'
                '    },\n'
                '    "recommendations": ["string"]\n'
                "  }\n"
                "}"
            )

            user_message = f"Resume text to analyze:\n{resume_text}"
            if job_description:
                user_message += f"\n\nJob description to match against:\n{job_description}"
            else:
                user_message += "\n\nNo specific job description provided. Analyze for general ATS best practices."

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                response_format=ResponseFormat.JSON
            )
            print("Response from Gemini (ATS):", response)
            if not response.success:
                raise ValueError(response.error)

            ats_results = response.content.get("ats_analysis", {})
            return {"ats_analysis": ats_results}

        return ats_analysis_tool

    def _create_content_quality_tool(self):
        @tool(parse_docstring=True)
        def content_quality_tool(resume_text: Optional[str] = None) -> Dict[str, Any]:
            """
            Analyze the resume content quality including impact statements, clarity, and professionalism.

            Args:
                resume_text: The complete resume text provided as input

            Returns:
                A dictionary containing content quality assessment.
            """
            system_prompt = (
                "You are a professional resume content evaluator. Analyze the quality of the resume content "
                "focusing on the following aspects:\n"
                "Format your response as a JSON object with the following structure:\n"
                "{\n"
                '  "content_quality": {\n'
                '    "impact_statements": {\n'
                '      "strengths": ["string"],\n'
                '      "weaknesses": ["string"],\n'
                '      "examples_found": ["string"]\n'
                '    },\n'
                '    "clarity": {\n'
                '      "issues": ["string"],\n'
                '      "improvements": ["string"]\n'
                '    },\n'
                '    "language": {\n'
                '      "professional_terms": ["string"],\n'
                '      "weak_phrases": ["string"]\n'
                '    },\n'
                '    "relevance": {\n'
                '      "irrelevant_content": ["string"]\n'
                '    },\n'
                '    "recommendations": ["string"]\n'
                "  }\n"
                "}"
            )
            user_message = f"Resume text to analyze:\n{resume_text}"

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                response_format=ResponseFormat.JSON
            )
            print("Response from Gemini (content quality):", response)
            if not response.success:
                raise ValueError(response.error)

            quality_assessment = response.content.get("content_quality", {})
            return {"content_quality": quality_assessment}

        return content_quality_tool

    def _create_suggestions_tool(self):
        @tool(parse_docstring=True)
        def suggestions_tool(
                resume_text: Optional[str] = None,
                formatting_issues: Optional[List[Dict[str, Any]]] = None,
                ats_analysis: Optional[Dict[str, Any]] = None,
                content_quality: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Generate specific suggestions to improve the resume based on analysis results.
            This tool can be called directly without requiring previous analysis.

            Args:
                resume_text: The complete resume text provided as input
                formatting_issues: Previously identified formatting issues (optional)
                ats_analysis: Results from ATS analysis (optional)
                content_quality: Content quality assessment (optional)

            Returns:
                A dictionary containing specific improvement suggestions.
            """
            system_prompt = (
                "You are a resume improvement expert. Based on the resume text and any available analysis results, "
                "provide specific, actionable suggestions to improve the resume. "
                "If previous analysis is not available, conduct your own analysis as part of generating suggestions. "
                "Format your response as a JSON object with the following structure:\n"
                "{\n"
                '  "improvement_suggestions": [\n'
                "    {\n"
                '      "category": "string", // formatting, content, ats, or language\n'
                '      "priority": "string", // high, medium, or low\n'
                '      "current_state": "string", // Description of the current issue\n'
                '      "suggested_change": "string", // Specific change to make\n'
                '      "expected_impact": "string", // Expected improvement after change\n'
                '      "section": "string" // Resume section this applies to\n'
                "    }\n"
                "  ]\n"
                "}"
            )

            # Create a structured analysis summary for the LLM
            analysis_summary = {}
            if formatting_issues:
                analysis_summary["formatting_issues"] = formatting_issues
            if ats_analysis:
                analysis_summary["ats_analysis"] = ats_analysis
            if content_quality:
                analysis_summary["content_quality"] = content_quality

            user_message = f"Resume text:\n{resume_text}\n\n"

            if analysis_summary:
                user_message += f"Analysis results:\n{json.dumps(analysis_summary, indent=2)}\n\n"
            else:
                user_message += "No previous analysis available. Please analyze the resume and provide suggestions directly.\n\n"

            user_message += "Please provide specific suggestions for improvement."

            response = self.llm.generate(
                system_prompt=system_prompt,
                user_message=user_message,
                response_format=ResponseFormat.JSON
            )
            print("Response from Gemini (suggestions):", response)
            if not response.success:
                raise ValueError(response.error)

            suggestions = response.content.get("improvement_suggestions", [])
            return {"improvement_suggestions": suggestions}

        return suggestions_tool

    def think(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Assistant node that evaluates the current state and executes the next required tool
        based on the user's request, not necessarily in sequence.
        Extends the think method from ReActAgent.
        """
        resume_text = state.get("resume_text", "")
        prompt = state.get("prompt", "")
        job_description = state.get("job_description", "")

        # Create a new state dictionary, preserving all existing fields
        new_state = dict(state)

        # Transfer all tracked fields from current state to new state
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'tool_call_id'):
                try:
                    # Try to parse content as JSON if it's a tool response
                    content = json.loads(last_message.content)
                    # Update any matching metadata fields from the content
                    for field in self.METADATA_FIELDS:
                        if field in content:
                            new_state[field] = content[field]
                except (json.JSONDecodeError, AttributeError):
                    pass

        sys_msg = SystemMessage(
            content=(
                "You are a resume enhancement assistant focused on targeted analysis. "
                "Choose tools based on what the user is specifically asking for. "
                "Important: You do NOT need to execute tools in a specific sequence. "
                "You can call any tool directly based on what's needed, including the suggestions_tool "
                "without requiring prior analysis tools."
                "\n\n"
                f"User Prompt: {prompt}"
            )
        )

        messages: list[BaseMessage] = [sys_msg] + state.get("messages", [])

        if not messages[-1].content.startswith("Here is a candidate's resume"):
            messages.append(
                HumanMessage(
                    content=(
                        f"Here is a candidate's resume:\n{resume_text}\n\n"
                        f"Job description (if provided): {job_description}\n\n"
                        "Based on the user's request, choose the appropriate tool(s) to execute. "
                        "You don't need to run all tools - only what's directly relevant to the request."
                    )
                )
            )

        # Use the llm_with_tools from ReActAgent parent class
        invocation = self.llm_with_tools.invoke(messages)
        new_state["messages"] = messages + [invocation]
        return new_state
