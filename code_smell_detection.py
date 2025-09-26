import marimo
import json

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        # Automated Code Smell Detection Using Large Language Models

        ## Problem Statement

        Code smells - patterns in source code that indicate potential design problems - are critical indicators of software quality issues. Traditionally, identifying code smells has relied on:

        1. Manual code reviews (time-consuming and subjective)
        2. Static analysis tools (limited to predefined patterns)
        3. Software metrics (often producing false positives)

        These approaches either require significant human effort or lack the contextual understanding needed to identify subtle design issues.

        ## How Generative AI Solves This Problem

        This notebook demonstrates how Large Language Models (LLMs) can transform code smell detection by:

        - **Contextual understanding**: Analyzing code with deep semantic understanding rather than just pattern matching
        - **Knowledge integration**: Leveraging structured knowledge about different code smell types and refactoring solutions
        - **Natural language reasoning**: Providing detailed explanations and actionable recommendations in human-readable format

        ## What This Notebook Demonstrates

        This end-to-end implementation shows how to:

        1. Build a knowledge base of code smells using vector embeddings and DeepLake
        2. Create a retrieval-augmented generation (RAG) system to provide context about code smells
        3. Analyze Java code files to detect multiple types of code smells
        4. Generate structured output with smell locations, severity, and refactoring suggestions
        5. Evaluate detection quality against manually labeled data

        The approach provides a practical showcase of how generative AI can be applied to enhance software development practices through automated code quality assessment.

        **Now, let's begin.**
        """)
    return


@app.cell
def __():
    import marimo as mo
    return mo,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""*Install all needed dependencies:*""")
    return


@app.cell
def __():
    # Install dependencies - in marimo these would be handled by script metadata
    # For now, ensure these are installed in your environment:
    # pip install -q -U "langchain-google-genai" "deeplake" "langchain" "langchain-text-splitters" "langchain-community" "tiktoken" "google-ai-generativelanguage==0.6.15" "deeplake[enterprise]<4.0.0"
    # pip install pillow lz4 python-dotenv
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now initialize deep lake vector store and store structured info about code smells in it""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        We will use 2 repos as data sources:

        * pixel-dungeon - as an example of software that contains numerous code smells.
        * smells - a comprehensive classification of code smells. We will use structured information about code smells from this repository in our DeepLake database.
        """)
    return


@app.cell
def __():
    # Clone repositories if they don't exist
    import os
    import subprocess

    if not os.path.exists("pixel-dungeon"):
        subprocess.run(["git", "clone", "https://github.com/watabou/pixel-dungeon.git"], check=True)

    if not os.path.exists("smells"):
        subprocess.run(["git", "clone", "https://github.com/Luzkan/smells.git"], check=True)

    print("Repositories are ready!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Get all files with structured info about code smells""")
    return


@app.cell
def __():
    from glob import glob
    from langchain.vectorstores import DeepLake
    from langchain_text_splitters import (Language, RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain.chains import RetrievalQA
    from langchain.schema import Document
    import deeplake
    return (
        ChatGoogleGenerativeAI, DeepLake, Document, GoogleGenerativeAIEmbeddings, Language, MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter, RetrievalQA, glob,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        Set up your API key

        To run the following cell, your API key must be stored in environment variables or a .env file as GOOGLE_API_KEY.

        If you don't already have an API key, you can grab one from AI Studio.
        """)
    return


@app.cell
def __():
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables. Please set it in your .env file or environment.")

    os.environ["GOOGLE_API_KEY"] = api_key
    print("API key loaded successfully!")
    return api_key,


@app.cell
def _(glob):
    smells_match = "smells/content/smells/**/*.md"
    all_smells = glob(smells_match, recursive=True)
    # Print the first 5 files
    print(f"First 5 of {len(all_smells)} markdown files:")
    for file in all_smells[:5]:
        print(f"  {file}")
    return all_smells,


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
        Each file with a matching path will be loaded and split by RecursiveCharacterTextSplitter. 
        Only Markdown files with structured content will be processed.
        """)
    return


@app.cell
def _(Language, RecursiveCharacterTextSplitter):
    # Common separators used for Markdown files
    separators = RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
    print("Markdown separators:", separators)
    return separators,


@app.cell
def _(Document, MarkdownHeaderTextSplitter, all_smells):
    # Define headers that match your code smell documentation structure
    headers_to_split_on = [("#", "Title"), ("##", "Section"), ("###", "Subsection")]

    docs = []
    for file_path in all_smells:
        try:
            # Load the file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract the main content (everything after the frontmatter)
            if '---' in content:
                # Find the second occurrence of '---' which ends the frontmatter
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    markdown_content = '---' + parts[2]
                else:
                    markdown_content = content
            else:
                markdown_content = content

            # Split by markdown headers
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(markdown_content)

            # Add source file info to metadata
            for doc in header_splits:
                doc.metadata['source'] = file_path

            # Add to collection
            docs.extend(header_splits)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    print(f"Created {len(docs)} chunks from {len(all_smells)} files")
    return docs, headers_to_split_on


@app.cell
def __():
    # Define path to database
    dataset_path = 'mem://deeplake/smells'
    return dataset_path,


@app.cell
def _(GoogleGenerativeAIEmbeddings):
    # Define the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return embeddings,


@app.cell
def _(DeepLake, dataset_path, docs, embeddings):
    smell_db = DeepLake.from_documents(docs, embeddings, dataset_path=dataset_path)
    print("Knowledge base created successfully!")
    return smell_db,


@app.cell
def _(smell_db):
    retriever = smell_db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 20  # number of documents to return
    return retriever,


@app.cell
def _(ChatGoogleGenerativeAI):
    # Define the chat model
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    return llm,


@app.cell
def _(RetrievalQA, llm, retriever):
    qa = RetrievalQA.from_llm(llm, retriever=retriever)
    return qa,


@app.cell
def _(mo, qa):
    # A helper function for calling retrieval chain
    def call_qa_chain(prompt):
        response = qa.invoke(prompt)
        return mo.md(response["result"])

    return call_qa_chain,


@app.cell
def _(call_qa_chain):
    # Test the knowledge base
    test_response = call_qa_chain(
        "List recommendations on how to get rid of 'God class' smell, rating from more simple to more sophisticated strategies.")
    test_response
    return test_response,


@app.cell
def _(all_smells):
    import enum
    from pathlib import Path

    # Create enum class dynamically from smell files
    def create_code_smell_enum():
        # Process each filename to create enum-friendly names
        enum_entries = {}

        for file_path in all_smells:
            # Extract filename without extension
            filename = Path(file_path).stem

            # Convert kebab-case to UPPER_SNAKE_CASE for enum names
            enum_name = filename.replace('-', '_').upper()

            # Use the original filename (without extension) as the value
            enum_entries[enum_name] = filename

        # Create and return the Enum class
        return enum.Enum('CodeSmell', enum_entries)

    # Create the enum
    CodeSmell = create_code_smell_enum()

    # Display some examples
    print(f"Created enum with {len(CodeSmell)} code smells")
    print("\nExample code smells:")
    for i, smell in enumerate(list(CodeSmell)[:5]):
        print(f"{smell.name} = '{smell.value}'")

    return CodeSmell, enum


@app.cell
def _(enum):
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, Field
    from typing import List, Optional

    # Define structured output schema using Pydantic
    class CodeSmellSeverity(str, enum.Enum):
        HIGH = "HIGH"
        MEDIUM = "MEDIUM"
        LOW = "LOW"

    class CodeSmellDetection(BaseModel):
        smell_type: str = Field(description="The type of code smell detected")
        location: str = Field(description="Where in the code the smell was found")
        severity: CodeSmellSeverity = Field(description="How severe the smell is")
        description: str = Field(description="Brief explanation of the issue")
        refactoring_suggestion: str = Field(description="How to fix the code smell")
        code_example: Optional[str] = Field(None, description="Example code showing the fix")

    class CodeAnalysisResult(BaseModel):
        analysis_summary: str = Field(description="Overall summary of code quality")
        smells_detected: List[CodeSmellDetection] = Field(description="List of detected code smells")

    # Create a parser for the structured output
    parser = PydanticOutputParser(pydantic_object=CodeAnalysisResult)

    return BaseModel, CodeAnalysisResult, CodeSmellDetection, CodeSmellSeverity, Field, List, Optional, PromptTemplate, PydanticOutputParser, parser


@app.cell
def _(CodeSmell, llm, parser, qa):
    def analyze_code_with_structure(code_content):
        # Get list of valid smells for reference
        valid_smells = ", ".join([smell.name for smell in CodeSmell])

        # Create a prompt that sends the code directly to the QA system
        analysis_prompt = f"""
        Analyze the following code for code smells:
        
        ```java
        {code_content}
        ```
        
        What code smells can you identify in this code and why? Only consider these code smell types: {valid_smells}.
        
        For each smell found, provide:
        1. The exact smell type (from the list provided)
        2. Location in the code (file, line numbers, method names)
        3. Severity (HIGH, MEDIUM, LOW)
        4. Description of why this is a code smell
        5. Refactoring suggestion to fix the issue
        6. Optional: Example code showing the fix
        
        Format your response as a structured JSON object matching this schema:
        {parser.get_format_instructions()}
        """

        # Use the QA system which leverages the smell knowledge base
        try:
            qa_result = qa.invoke(analysis_prompt)

            # Try to extract and parse JSON from the result
            output_text = qa_result["result"]
            parsed_output = parser.parse(output_text)
            return parsed_output
        except Exception as e:
            print(f"Error in code smell analysis: {e}")
            print(f"Raw QA output: {qa_result['result'] if 'qa_result' in locals() else 'No output'}")

            # Fallback to direct LLM call if QA system parsing fails
            try:
                direct_output = llm.invoke(analysis_prompt)
                parsed_output = parser.parse(direct_output.content)
                return parsed_output
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                return None

    return analyze_code_with_structure,


@app.cell
def _(analyze_code_with_structure):
    def display_code_analysis(file_path):
        # Read the file content 
        with open(file_path, 'r') as f:
            code_content = f.read()

        # Analyze the code
        analysis = analyze_code_with_structure(code_content)

        if not analysis:
            print("Failed to analyze code.")
            return

        # Display results
        print(f"Analysis Summary: {analysis.analysis_summary}\n")
        print(f"Detected {len(analysis.smells_detected)} code smells:")

        for i, smell in enumerate(analysis.smells_detected, 1):
            print(f"\n{i}. {smell.smell_type} ({smell.severity})")
            print(f"   Location: {smell.location}")
            print(f"   Description: {smell.description}")
            print(f"   Refactoring: {smell.refactoring_suggestion}")
            if smell.code_example:
                print(f"\n   Example fix:\n   ```\n{smell.code_example}\n   ```")

    return display_code_analysis,


@app.cell
def _(display_code_analysis):
    # Analyze a Java file from the pixel-dungeon repository
    java_file = "pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java"

    print(f"Analyzing {java_file}...")
    display_code_analysis(java_file)
    return java_file,


@app.cell
def _(BaseModel, Field, List, Optional, PydanticOutputParser, PromptTemplate, analyze_code_with_structure, enum):

    # Define evaluation schema
    class EvaluationScore(str, enum.Enum):
        EXCELLENT = "EXCELLENT"
        GOOD = "GOOD"
        ACCEPTABLE = "ACCEPTABLE"
        POOR = "POOR"
        INCORRECT = "INCORRECT"

    class SmellEvaluation(BaseModel):
        detected_smell: str = Field(description="The detected code smell type")
        location: str = Field(description="Where the smell was detected")
        ground_truth_match: Optional[str] = Field(None, description="The matching ground truth smell if any")
        score: EvaluationScore = Field(description="Evaluation score for this detection")
        justification: str = Field(description="Explanation for the rating")

    class CodeSmellEvaluationResult(BaseModel):
        overall_score: float = Field(description="Overall evaluation score out of 5")
        precision: float = Field(description="Ratio of correctly identified smells to all detections")
        recall: float = Field(description="Ratio of correctly identified smells to all actual smells")
        evaluations: List[SmellEvaluation] = Field(description="Individual smell evaluations")
        summary: str = Field(description="Summary of evaluation results")

    # Create the evaluation prompt template
    eval_template = """
# Instruction
You are an expert evaluator specializing in code smell detection. Your task is to evaluate the quality of code smell detections by comparing them with ground truth data.

# Evaluation
## Metric Definition
You will be assessing code smell detection quality, which measures how accurately the system identifies:
1. The correct type of code smell
2. The correct location of the smell (file, line numbers, method)

## Criteria
Smell Type Accuracy: The detected smell type matches the actual smell type in the code.
Location Accuracy: The location identified for the smell (line numbers, method, class) matches where the smell actually exists.
Justification Quality: The explanation provided for the smell makes sense and correctly describes the issue.
Refactoring Relevance: The suggested refactoring is appropriate for the identified smell.

## Rating Rubric
EXCELLENT: Perfect match of smell type and exact location (score: 5).
GOOD: Correct smell type with minor location imprecision (score: 4).
ACCEPTABLE: Partial match (either correct smell type or approximate location) (score: 3).
POOR: Wrong smell type but area of concern correctly identified (score: 2).
INCORRECT: Completely incorrect detection (wrong smell type and location) (score: 1).

## Evaluation Steps
STEP 1: For each detected smell, find any matching ground truth smells.
STEP 2: Evaluate the accuracy of the detected smell type.
STEP 3: Evaluate the precision of the location identified.
STEP 4: Assign a score based on the rating rubric.
STEP 5: Calculate overall precision and recall metrics.

# Input Data
## Ground Truth Smells
{ground_truth}

## Detected Smells
{detected_smells}

{format_instructions}
"""

    # Create the evaluation parser and prompt
    eval_parser = PydanticOutputParser(pydantic_object=CodeSmellEvaluationResult)
    eval_prompt = PromptTemplate(template=eval_template, input_variables=["ground_truth", "detected_smells"],
        partial_variables={"format_instructions": eval_parser.get_format_instructions()})

    def evaluate_smell_detection(ground_truth_data, detected_smells, llm):
        """Evaluate the quality of code smell detection by comparing with ground truth."""
        # Convert data to JSON strings
        ground_truth_str = json.dumps(ground_truth_data, indent=2)
        detected_str = json.dumps(detected_smells, indent=2)

        # Format the prompt
        formatted_prompt = eval_prompt.format(ground_truth=ground_truth_str, detected_smells=detected_str)

        # Get response from LLM
        try:
            output = llm.invoke(formatted_prompt)
            parsed_output = eval_parser.parse(output.content)
            return parsed_output
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print(f"Raw output: {output.content if 'output' in locals() else 'No output'}")
            return None

    def create_ground_truth(file_path, manual_annotations):
        """Create ground truth data structure"""
        return {"file_path": file_path, "smells": manual_annotations}

    def evaluate_code_analysis(file_path, manual_annotations, llm):
        """Run a full evaluation on a code file"""
        # Get the code content
        with open(file_path, 'r') as f:
            code_content = f.read()

        # Run code smell detection
        analysis_result = analyze_code_with_structure(code_content)

        if not analysis_result:
            print("Failed to analyze code")
            return None

        # Format detected smells
        detected_smells = {"file_path": file_path, "smells": []}

        for smell in analysis_result.smells_detected:
            detected_smells["smells"].append(
                {"smell_type": smell.smell_type, "location": smell.location, "description": smell.description,
                    "severity": str(smell.severity), "refactoring": smell.refactoring_suggestion})

        # Create ground truth
        ground_truth = create_ground_truth(file_path, manual_annotations)

        # Run evaluation
        evaluation = evaluate_smell_detection(ground_truth, detected_smells, llm)

        # Display results
        if evaluation:
            print(f"Overall Score: {evaluation.overall_score:.2f}/5.0")
            print(f"Precision: {evaluation.precision:.2f}")
            print(f"Recall: {evaluation.recall:.2f}")
            print(f"\nSummary: {evaluation.summary}\n")

            print("Individual Evaluations:")
            for i, eval_item in enumerate(evaluation.evaluations, 1):
                print(f"\n{i}. {eval_item.detected_smell} ({eval_item.location})")
                print(f"   Score: {eval_item.score.value}")
                print(
                    f"   Matched with: {eval_item.ground_truth_match if eval_item.ground_truth_match else 'No match'}")
                print(f"   Justification: {eval_item.justification}")

        return evaluation

    return EvaluationScore, SmellEvaluation, CodeSmellEvaluationResult, create_ground_truth, eval_parser, eval_prompt, eval_template, evaluate_code_analysis, evaluate_smell_detection


@app.cell
def _(evaluate_code_analysis, llm):
    # Example ground truth data for evaluation
    evaluation_files = {"pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java": [
        {"smell_type": "MAGIC_NUMBER", "location": "multiple locations: lines 25-26, 37, 52-53, 85, 89",
            "severity": "MEDIUM"},
        {"smell_type": "LONG_METHOD", "location": "decorate() method, lines 74-101", "severity": "MEDIUM"},
        {"smell_type": "CONDITIONAL_COMPLEXITY", "location": "decorate() method, lines 76-89", "severity": "LOW"},
        {"smell_type": "DUPLICATED_CODE", "location": "tileName() and tileDesc() methods, lines 104-138",
            "severity": "LOW"},
        {"smell_type": "DEAD_CODE", "location": "map[i] == 63 condition in addVisuals method, line 151",
            "severity": "MEDIUM"},
        {"smell_type": "FEATURE_ENVY", "location": "addVisuals method, lines 149-153", "severity": "LOW"},
        {"smell_type": "PRIMITIVE_OBSESSION", "location": "throughout class, using boolean arrays and ints for terrain",
            "severity": "LOW"}]}

    # Run evaluation on the sample file
    print("=== Running Evaluation ===")
    evaluation = evaluate_code_analysis("pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java",
        evaluation_files["pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java"], llm)

    return evaluation, evaluation_files


@app.cell
def _(evaluate_code_analysis, evaluation_files, llm):
    # Run evaluations and collect results
    results = {}
    for file_path, annotations in evaluation_files.items():
        print(f"\n=== Evaluating {file_path} ===")
        eval_result = evaluate_code_analysis(file_path, annotations, llm)
        if eval_result:
            results[file_path] = eval_result

    # Calculate overall metrics
    if results:
        total_score = sum(result.overall_score for result in results.values())
        avg_score = total_score / len(results)
        avg_precision = sum(result.precision for result in results.values()) / len(results)
        avg_recall = sum(result.recall for result in results.values()) / len(results)

        print("\n=== OVERALL EVALUATION ===")
        print(f"Average Score: {avg_score:.2f}/5.0")
        print(f"Average Precision: {avg_precision:.2f}")
        print(f"Average Recall: {avg_recall:.2f}")

    return avg_precision, avg_recall, avg_score, results, total_score


if __name__ == "__main__":
    app.run()
