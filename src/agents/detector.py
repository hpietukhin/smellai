"""
Code smell detector agent using LLM with RAG.

This module implements the code smell detection agent that uses:
- LiteLLM for Cerebras model access
- RAG (Retrieval-Augmented Generation) for smell knowledge
- **Pydantic models for structured output** (SmellDetection)

**Implementation based on**: pipeline_reference/pipeline.py
**Structured Output**: Uses Pydantic models with PydanticOutputParser for type-safe LLM responses
"""

import os
from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from litellm import completion
from pydantic import BaseModel, Field

from src.models.entities import SmellDetection


# Pydantic model for structured LLM output
class CodeAnalysisResult(BaseModel):
    """
    Structured output model for code smell detection.

    This Pydantic model defines the schema for LLM responses, ensuring
    type-safe and validated structured output.
    """

    analysis_summary: str = Field(
        description="Overall summary of code quality and detected issues"
    )
    smells_detected: List[SmellDetection] = Field(
        description="List of all detected code smells with details"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "analysis_summary": "Code shows 2 major code smells requiring refactoring",
                "smells_detected": [
                    {
                        "smell_type": "Complex Method",
                        "location": "UserService.processData() (lines 45-120)",
                        "description": "Method has cyclomatic complexity of 15",
                        "severity": "HIGH",
                        "refactoring_suggestion": "Extract validation into separate methods",
                        "confidence": 0.92,
                    }
                ],
            }
        }


def get_llm_completion(
    messages: list,
    model: str = "cerebras/llama3.1-8b",
    temperature: float = 0.0,
    max_tokens: int = 4096,
    **kwargs,
) -> str:
    """
    Get completion from Cerebras LLM via LiteLLM.

    Uses LiteLLM's unified API to call Cerebras models with OpenAI-compatible
    interface. Configured for deterministic code analysis (temperature=0.0).

    **Task**: P3-LLM-001 - Initialize Cerebras LLM via LiteLLM
    **Based on**: LiteLLM documentation for Cerebras provider

    Args:
        messages: List of message dicts with 'role' and 'content'
        model: Model identifier (default: "cerebras/llama3.1-8b")
        temperature: Sampling temperature (0.0 = deterministic)
        max_tokens: Maximum tokens in response
        **kwargs: Additional parameters passed to litellm.completion()

    Returns:
        String content of the LLM response

    Raises:
        ValueError: If CEREBRAS_API_KEY environment variable not set
        Exception: If LLM API call fails

    Example:
        >>> messages = [{"role": "user", "content": "Analyze this code..."}]
        >>> response = get_llm_completion(messages)
        >>> print(response)

    Notes:
        - Requires CEREBRAS_API_KEY in environment
        - Model options: llama3.1-8b, llama3-70b-instruct
        - temperature=0.0 for reproducible code analysis
    """
    # Verify API key is set
    if not os.getenv("CEREBRAS_API_KEY"):
        raise ValueError(
            "CEREBRAS_API_KEY environment variable not set. "
            "Please add it to your .env file. "
            "Get a key at: https://inference.cerebras.ai/"
        )

    try:
        # Call Cerebras via LiteLLM
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract content from response
        return response.choices[0].message.content

    except Exception as e:
        raise Exception(
            f"LLM completion failed: {e}. " "Check CEREBRAS_API_KEY and model name."
        ) from e


def create_detection_prompt() -> tuple[PromptTemplate, PydanticOutputParser]:
    """
    Create prompt template for code smell detection with structured output.

    Builds a LangChain PromptTemplate configured with PydanticOutputParser
    for type-safe, structured LLM responses. The prompt includes:
    - System instructions for code analysis
    - RAG context placeholder
    - Code input placeholder
    - Format instructions for Pydantic model

    **Task**: P3-LLM-002 - Create detection prompt template
    **Based on**: pipeline_reference/pipeline.py lines 364-374
    **Structured Output**: Uses PydanticOutputParser for CodeAnalysisResult model

    Returns:
        Tuple of (PromptTemplate, PydanticOutputParser)

    Example:
        >>> prompt, parser = create_detection_prompt()
        >>> formatted = prompt.format(context="...", code="...")
        >>> # Use formatted prompt with LLM
        >>> response = get_llm_completion([{"role": "user", "content": formatted}])
        >>> result = parser.parse(response)
        >>> print(f"Found {len(result.smells_detected)} smells")

    Notes:
        - Parser provides schema via get_format_instructions()
        - Prompt includes both RAG context and code
        - Output conforms to CodeAnalysisResult Pydantic model
    """
    # Create Pydantic output parser for structured responses
    parser = PydanticOutputParser(pydantic_object=CodeAnalysisResult)

    # Create prompt template with RAG context and format instructions
    template = """You are an expert code smell detector. Analyze the given code for code smells using the provided knowledge base context.

**Code Smell Knowledge Base Context**:
{context}

**Code to Analyze**:
```java
{code}
```

**Instructions**:
1. Identify ALL code smells present in the code
2. For each smell, provide:
   - Type (from knowledge base context)
   - Exact location (class/method names, line ranges)
   - Severity (HIGH, MEDIUM, or LOW)
   - Clear description of why it's a smell
   - Specific refactoring suggestion
   - Confidence score (0.0 to 1.0)
3. Provide an overall analysis summary

{format_instructions}

Respond with valid JSON matching the schema above."""

    # Create prompt template
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "code"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    return prompt, parser


def detect_smells(
    code: str, retriever, model: str = "cerebras/llama3.1-8b", temperature: float = 0.0
) -> List[SmellDetection]:
    """
    Detect code smells in given code using RAG and LLM.

    Implements the complete smell detection pipeline:
    1. Retrieve relevant smell documentation (RAG)
    2. Format prompt with context and code
    3. Get structured LLM response
    4. Parse and return SmellDetection objects

    **Task**: P3-LLM-003 - Create smell detection function
    **Based on**: pipeline_reference/pipeline.py lines 380-427
    **Structured Output**: Returns list of SmellDetection Pydantic models

    Args:
        code: Source code to analyze (Java)
        retriever: LangChain retriever for smell knowledge base
        model: Cerebras model to use (default: llama3.1-8b)
        temperature: LLM temperature (default: 0.0 for deterministic)

    Returns:
        List of SmellDetection objects (empty list if none found or parsing fails)

    Example:
        >>> from src.data.vector_db import load_and_create_vector_db
        >>> _, retriever = load_and_create_vector_db()
        >>> code = '''
        ... public class Example {
        ...     public void longMethod() {
        ...         // 100 lines of code...
        ...     }
        ... }
        ... '''
        >>> smells = detect_smells(code, retriever)
        >>> for smell in smells:
        ...     print(f"{smell.smell_type}: {smell.location}")

    Notes:
        - Retrieves k=20 most relevant smell documents
        - Uses Pydantic parsing for type-safe output
        - Returns empty list on parsing errors (graceful degradation)
        - Confidence scores help evaluate detection quality
    """
    try:
        # Step 1: Retrieve relevant smell documentation
        print("Retrieving relevant smell documentation...")
        relevant_docs = retriever.get_relevant_documents(code)

        # Format context from retrieved documents
        context = "\n\n".join(
            [
                f"**{doc.metadata.get('source', 'Unknown')}**:\n{doc.page_content}"
                for doc in relevant_docs[:10]  # Limit to top 10 to avoid token limits
            ]
        )

        print(f"✓ Retrieved {len(relevant_docs)} documents (using top 10)")

        # Step 2: Create prompt with context and code
        prompt_template, parser = create_detection_prompt()
        formatted_prompt = prompt_template.format(context=context, code=code)

        # Step 3: Get LLM response
        print(f"Analyzing code with {model}...")
        messages = [
            {
                "role": "system",
                "content": "You are an expert code smell detector. Always respond with valid JSON.",
            },
            {"role": "user", "content": formatted_prompt},
        ]

        response_text = get_llm_completion(
            messages, model=model, temperature=temperature
        )

        print(f"✓ Received LLM response ({len(response_text)} characters)")

        # Step 4: Parse structured output
        print("Parsing structured output...")
        result = parser.parse(response_text)

        print(f"✓ Detected {len(result.smells_detected)} code smells")
        print(f"  Summary: {result.analysis_summary[:100]}...")

        return result.smells_detected

    except Exception as e:
        print(f"⚠ Error in smell detection: {e}")
        print("  Returning empty list (graceful degradation)")
        return []
