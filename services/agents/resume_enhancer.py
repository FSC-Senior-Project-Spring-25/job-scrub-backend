import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pinecone import Pinecone

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM, ResponseFormat

load_dotenv()


class EnhancementState(MessagesState):
    """
    The agent state for resume enhancement.
    It holds the resume text and conversation messages.
    """
    resume_text: str
    prompt: str

class ResumeEnhancer:
    def __init__(self, llm: GeminiLLM):
        """
        Initializes the ResumeEnhancer.

        Args:
            llm: An instance of GeminiLLM.
        """
        self.llm = llm

        # Create all the tools using the _create_tools method
        self.tools = self._create_tools()

        # Bind the tools to the LLM
        self.llm_with_tools = self.llm.chat.bind_tools(self.tools)

        # Build the ReAct agent state graph.
        self.builder = StateGraph(EnhancementState)
        self.builder.add_node("think", self.think)
        self.builder.add_node("tools", ToolNode(self.tools))
        self.builder.add_edge(START, "think")
        self.builder.add_conditional_edges("think", tools_condition)
        self.builder.add_edge("tools", "think")
        self.agent = self.builder.compile()
        print("==" * 20)
        print("Resume Enhancer Graph:")
        print("==" * 20)
        self.agent.get_graph().print_ascii()

    def _create_tools(self):
        """
        Creates and returns all the tools for the ResumeEnhancer.

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
                "Format your response as a JSON object with the following structure:\n"
                "{\n"
                '  "ats_analysis": {\n'
                '    "score": number, // Overall ATS compatibility score (0-100)\n'
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
                '    "keyword_analysis": {\n'
                '      "found_keywords": ["string"],\n'
                '      "missing_keywords": ["string"],\n'
                '      "keyword_match_rate": number\n'
                '    },\n'
                '    "recommendations": ["string"]\n'
                "  }\n"
                "}"
            )

            user_message = f"Resume text to analyze:\n{resume_text}"
            if job_description:
                user_message += f"\n\nJob description to match against:\n{job_description}"

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
                '    "overall_score": number, // Score from 0-100\n'
                '    "impact_statements": {\n'
                '      "score": number,\n'
                '      "strengths": ["string"],\n'
                '      "weaknesses": ["string"],\n'
                '      "examples_found": ["string"]\n'
                '    },\n'
                '    "clarity": {\n'
                '      "score": number,\n'
                '      "issues": ["string"],\n'
                '      "improvements": ["string"]\n'
                '    },\n'
                '    "language": {\n'
                '      "score": number,\n'
                '      "professional_terms": ["string"],\n'
                '      "weak_phrases": ["string"]\n'
                '    },\n'
                '    "relevance": {\n'
                '      "score": number,\n'
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

            Args:
                resume_text: The complete resume text provided as input
                formatting_issues: Previously identified formatting issues
                ats_analysis: Results from ATS analysis
                content_quality: Content quality assessment

            Returns:
                A dictionary containing specific improvement suggestions.
            """
            system_prompt = (
                "You are a resume improvement expert. Based on the resume text and analysis results, "
                "provide specific, actionable suggestions to improve the resume. "
                "Focus on transforming identified issues into concrete improvements. "
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
            analysis_summary = {
                "formatting_issues": formatting_issues or [],
                "ats_analysis": ats_analysis or {},
                "content_quality": content_quality or {}
            }

            user_message = (
                f"Resume text:\n{resume_text}\n\n"
                f"Analysis results:\n{json.dumps(analysis_summary, indent=2)}\n\n"
                f"Please provide specific suggestions for improvement."
            )

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
        without repeating previously used tools.
        """
        resume_text = state.get("resume_text", "")
        prompt = state.get("prompt", "")
        sys_msg = SystemMessage(
            content=(
                "You are a resume enhancement assistant focused on systematic analysis. "
                "Important: Do not repeat tools that have already been called. "
                "Choose tools only based on the user prompt\n"
                f"User Prompt: {prompt}"
            )
        )

        messages: list[BaseMessage] = [sys_msg] + state["messages"]

        if not messages[-1].content.startswith("Here is a candidate's resume"):
            messages.append(
                HumanMessage(
                    content=(
                        f"Here is a candidate's resume:\n{resume_text}\n\n"
                        "Continue the analysis using the next appropriate tool in sequence."
                    )
                )
            )

        invocation = self.llm_with_tools.invoke(messages)
        return {"messages": [invocation]}

    def enhance_resume(self, resume_text: str, prompt: str) -> Dict[str, Any]:
        """
        Runs the ReAct agent graph using the provided resume text and prints the result.

        Args:
            resume_text: The resume text to analyze
            prompt: The prompt to guide the analysis
        """
        initial_state = {
            "resume_text": resume_text,
            "prompt": prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
        }
        result = self.agent.invoke(initial_state)
        print("Enhancer Final Result:")
        print(result["messages"][-1].content)
        return {
            "answer": result["messages"][-1].content,
            "error": None
        }


if __name__ == "__main__":
    user_id = "oPmOJhSE0VQid56yYyg19hdH5DV2"
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index("resumes")
    resume_data = asyncio.run(get_user_resume(index, user_id))
    if not resume_data.get("text"):
        print("Resume text not found")
        exit(1)

    # Instantiate GeminiLLM
    gemini_llm = GeminiLLM()

    # Create the ResumeEnhancer instance
    enhancer = ResumeEnhancer(gemini_llm)

    # Run the agent graph to analyze the resume with optional job description
    enhancer.enhance_resume(resume_data["text"], "Analyze this resume for ATS compatibility and content quality.")
