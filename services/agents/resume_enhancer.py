import json
import operator
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, List, Dict, Any, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import CompiledStateGraph
from pinecone import Pinecone

from services.agents.tools.get_user_resume import get_user_resume
from services.gemini import GeminiLLM, ResponseFormat
from services.s3 import S3Service
from services.text_embedder import TextEmbedder


class EnhancementPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SuggestionItem:
    """Container for a single resume enhancement suggestion"""
    description: str
    priority: EnhancementPriority
    section: str
    examples: List[str]
    reasoning: str


class EnhancementState(TypedDict):
    """State for the resume enhancement workflow"""
    user_id: str
    resume_text: str
    resume_keywords: List[str]
    # Using Annotated with operator.add to enable parallel processing and result accumulation
    ats_issues: Annotated[List[Dict[str, Any]], operator.add]
    content_issues: Annotated[List[Dict[str, Any]], operator.add]
    formatting_issues: Annotated[List[Dict[str, Any]], operator.add]
    enhancement_suggestions: List[SuggestionItem]
    content_quality_score: float
    overall_enhancement_score: float


class ResumeEnhancementAgent:
    """Agent for analyzing resumes and suggesting general improvements"""

    def __init__(
            self,
            llm: GeminiLLM,
            text_embedder: TextEmbedder,
            s3_service: S3Service,
            resumes_index: Pinecone.Index
    ):
        self.llm = llm
        self.text_embedder = text_embedder
        self.s3_service = s3_service
        self.resumes_index = resumes_index
        self.workflow = self._create_workflow()

    async def enhance_resume(
            self,
            user_id: str
    ) -> Dict[str, Any]:
        """
        Generate general resume enhancement suggestions

        Args:
            user_id: User identifier

        Returns:
            Dictionary containing enhancement suggestions and metrics
        """
        # Get user's resume data from Pinecone
        resume_data = await get_user_resume(self.resumes_index, user_id)
        print(f"Resume data for user {user_id}: {resume_data}")
        # Initialize workflow state
        initial_state = EnhancementState(
            user_id=user_id,
            resume_text=resume_data["text"],
            resume_keywords=resume_data["keywords"],
            ats_issues=[],
            content_issues=[],
            formatting_issues=[],
            enhancement_suggestions=[],
            content_quality_score=0.0,
            overall_enhancement_score=0.0
        )

        # Run the enhancement workflow
        result = await self.workflow.ainvoke(initial_state)
        # Format the response
        return {
            "file_id": resume_data["file_id"],
            "filename": resume_data["filename"],
            "content_quality_score": result["content_quality_score"],
            "overall_enhancement_score": result["overall_enhancement_score"],
            "ats_issues": result["ats_issues"],
            "content_issues": result["content_issues"],
            "formatting_issues": result["formatting_issues"],
            "enhancement_suggestions": [
                {
                    "description": suggestion.description,
                    "priority": suggestion.priority.value,
                    "section": suggestion.section,
                    "examples": suggestion.examples,
                    "reasoning": suggestion.reasoning
                }
                for suggestion in result["enhancement_suggestions"]
            ]
        }

    async def _analyze_resume_keywords(self, state: EnhancementState) -> EnhancementState:
        """
        Analyze keywords in the resume for completeness and quality

        Args:
            state: Current enhancement state

        Returns:
            Updated state with keyword analysis
        """
        # If keywords weren't already extracted, get them now
        if not state["resume_keywords"]:
            kw_response = await self.llm.generate(
                system_prompt=(
                    "You are an expert in technical skill extraction. "
                    "Extract all technical skills, tools, frameworks, programming languages, "
                    "and industry-specific terms from the provided resume text. "
                    "Return only a JSON array of keywords, with no explanations."
                ),
                user_message=state["resume_text"],
                response_format=ResponseFormat.JSON
            )
            print("Keyword extraction response:", kw_response)
            if kw_response.success and isinstance(kw_response.content, dict):
                # Extract the array from the response
                if "keywords" in kw_response.content:
                    state["resume_keywords"] = kw_response.content["keywords"]
                else:
                    # If the response is already an array or the first key contains the array
                    first_key = next(iter(kw_response.content), None)
                    if first_key:
                        state["resume_keywords"] = kw_response.content[first_key]

        return state

    async def _analyze_ats_compatibility(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Analyze resume for ATS (Applicant Tracking System) compatibility issues

        Args:
            state: Current enhancement state

        Returns:
            Dictionary with ATS issues and scores
        """
        ats_prompt = f"""
        You are an expert in resume optimization for ATS (Applicant Tracking System) scanning.
        Analyze the resume below for ATS compatibility issues.

        Focus on:
        1. Format problems that may confuse ATS systems
        2. Keyword usage and placement
        3. Section organization and labeling
        4. Use of tables, graphics, or unusual formatting that ATS might miss
        5. File format compatibility issues

        Resume Text:
        {state['resume_text']}

        Format your response as a JSON object with the following structure:
        {{
            "ats_issues": [
                {{
                    "issue": "Brief description of the issue",
                    "impact": "high/medium/low",
                    "recommendation": "How to fix it"
                }}
            ]
        }}

        Only include actual issues - don't make up problems if the resume is well-formatted.
        """

        response = await self.llm.generate(
            system_prompt="You are an expert ATS optimization assistant.",
            user_message=ats_prompt,
            response_format=ResponseFormat.JSON
        )
        print("ATS analysis response:", response)
        ats_issues = []
        if response.success and isinstance(response.content, dict):
            ats_issues = response.content.get("ats_issues", [])

        return {"ats_issues": ats_issues}

    async def _analyze_content_quality(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Analyze resume content quality for general effectiveness

        Args:
            state: Current enhancement state

        Returns:
            Dictionary with content issues and content quality score
        """
        content_prompt = f"""
        You are an expert resume writer with deep experience in crafting effective resumes.

        RESUME:
        {state['resume_text']}

        Analyze this resume for content quality and effectiveness.
        Focus on:

        1. Achievement descriptions (quantified results vs vague statements)
        2. Action verbs and impact language
        3. Technical terminology and clarity
        4. Overall positioning and narrative focus
        5. Relevance of included information

        Format your response as a JSON object with:
        {{
            "content_issues": [
                {{
                    "issue": "Brief description",
                    "impact": "high/medium/low",
                    "recommendation": "How to improve",
                    "example": "Example of improved wording"
                }}
            ],
            "content_quality_score": 0.75 // Score from 0.0 to 1.0
        }}

        Be honest but constructive. Focus on the most important improvements.
        """

        response = await self.llm.generate(
            system_prompt="You are an expert resume content optimization assistant.",
            user_message=content_prompt,
            response_format=ResponseFormat.JSON
        )
        print("Content analysis response:", response)

        content_issues = []
        content_quality_score = 0.75

        if response.success:
            content_issues = response.content.get("content_issues", [])
            content_quality_score = float(response.content.get("content_quality_score", 0.75))

        return {
            "content_issues": content_issues,
            "content_quality_score": content_quality_score
        }

    async def _analyze_formatting(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Analyze resume formatting and structure

        Args:
            state: Current enhancement state

        Returns:
            Dictionary with formatting issues
        """
        format_prompt = f"""
        You are an expert resume designer and formatter.

        Analyze the following resume for formatting and structure issues:

        {state['resume_text']}

        Focus on:
        1. Overall structure and readability
        2. Section organization and hierarchy
        3. Consistency in formatting
        4. Use of white space and visual balance
        5. Length and conciseness

        Format your response as a JSON object with:
        {{
            "formatting_issues": [
                {{
                    "issue": "Brief description",
                    "impact": "high/medium/low",
                    "recommendation": "How to improve"
                }}
            ]
        }}

        Be specific and actionable in your recommendations.
        """

        response = await self.llm.generate(
            system_prompt="You are an expert resume formatting assistant.",
            user_message=format_prompt,
            response_format=ResponseFormat.JSON
        )
        print("Formatting analysis response:", response)
        formatting_issues = []
        if response.success and isinstance(response.content, dict):
            formatting_issues = response.content.get("formatting_issues", [])

        return {"formatting_issues": formatting_issues}

    async def _process_content_quality_score(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Process and extract content quality score from content analysis results

        Args:
            state: Current enhancement state

        Returns:
            Dictionary with content quality score
        """
        # Extract the content quality score from the state
        # This node runs after content analysis to extract just the score
        return {"content_quality_score": state.get("content_quality_score", 0.75)}

    async def _generate_suggestions(self, state: EnhancementState) -> Dict[str, Any]:
        """
        Generate prioritized enhancement suggestions based on all analyses

        Args:
            state: Current enhancement state

        Returns:
            Dictionary with suggestions and overall enhancement score
        """
        # Prepare context for the LLM with all analyses
        context = {
            "resume_keywords": state["resume_keywords"],
            "content_quality_score": state["content_quality_score"],
            "ats_issues": state["ats_issues"],
            "content_issues": state["content_issues"],
            "formatting_issues": state["formatting_issues"]
        }

        suggestion_prompt = f"""
        You are an expert resume coach helping a candidate improve their general resume quality.

        Based on the following analysis, generate 3-5 high-impact, actionable enhancement suggestions:

        Analysis context:
        {json.dumps(context, indent=2)}

        Format your response as a JSON object with:
        {{
            "enhancement_suggestions": [
                {{
                    "description": "Clear, actionable suggestion title",
                    "priority": "high/medium/low",
                    "section": "skills/experience/education/summary/etc.",
                    "examples": ["Example 1", "Example 2"],
                    "reasoning": "Why this change improves the overall resume quality"
                }}
            ],
            "overall_enhancement_score": 0.68 // Score from 0.0 to 1.0
        }}

        Prioritize suggestions that will have the biggest impact on the candidate's resume quality.
        Be specific, constructive, and focus on general best practices for effective resumes.
        """

        response = await self.llm.generate(
            system_prompt="You are an expert resume enhancement coach.",
            user_message=suggestion_prompt,
            response_format=ResponseFormat.JSON
        )
        print("Enhancement suggestions response:", response)
        enhancement_suggestions = []
        overall_enhancement_score = 0.0

        if response.success and isinstance(response.content, dict):
            suggestion_list = response.content.get("enhancement_suggestions", [])
            overall_enhancement_score = float(response.content.get("overall_enhancement_score", 0.0))

            # Convert dictionary suggestions to SuggestionItem objects
            for item in suggestion_list:
                enhancement_suggestions.append(SuggestionItem(
                    description=item["description"],
                    priority=EnhancementPriority(item["priority"]),
                    section=item["section"],
                    examples=item["examples"],
                    reasoning=item["reasoning"]
                ))

        return {
            "enhancement_suggestions": enhancement_suggestions,
            "overall_enhancement_score": overall_enhancement_score
        }

    def _create_workflow(self) -> CompiledStateGraph:
        """
        Create and compile the enhancement workflow graph with parallel execution

        Returns:
            Compiled workflow graph
        """
        builder = StateGraph(EnhancementState)

        # Add workflow nodes
        builder.add_node("analyze_resume_keywords", self._analyze_resume_keywords)
        builder.add_node("analyze_ats_compatibility", self._analyze_ats_compatibility)
        builder.add_node("analyze_content_quality", self._analyze_content_quality)
        builder.add_node("analyze_formatting", self._analyze_formatting)
        builder.add_node("process_content_quality_score", self._process_content_quality_score)
        builder.add_node("generate_suggestions", self._generate_suggestions)

        # Define workflow edges for parallel execution
        builder.add_edge(START, "analyze_resume_keywords")

        # Fan out from keywords analysis to three parallel processes
        builder.add_edge("analyze_resume_keywords", "analyze_ats_compatibility")
        builder.add_edge("analyze_resume_keywords", "analyze_content_quality")
        builder.add_edge("analyze_resume_keywords", "analyze_formatting")

        # Extract content quality score for use in generate_suggestions
        builder.add_edge("analyze_content_quality", "process_content_quality_score")

        # Fan in from all analysis nodes to generate suggestions
        # We use a list to indicate that all these nodes must complete before continuing
        builder.add_edge(
            ["analyze_ats_compatibility", "analyze_content_quality", "analyze_formatting",
             "process_content_quality_score"],
            "generate_suggestions"
        )

        builder.add_edge("generate_suggestions", END)

        # Compile the workflow
        workflow = builder.compile()
        print("General Resume Enhancement Workflow:")
        print(workflow.get_graph().draw_ascii())

        return workflow