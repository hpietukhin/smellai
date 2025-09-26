import marimo

__generated_with = "0.14.17"
app = marimo.App()


app._unparsable_cell(
    r"""
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
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        *install all needed dependencies:*
        """
    )
    return


@app.cell
def _():
    # '%pip install -q -U "langchain-google-genai" "deeplake" "langchain" "langchain-text-splitters" "langchain-community" "tiktoken" "google-ai-generativelanguage==0.6.15" "deeplake[enterprise]<4.0.0"' command supported automatically in marimo
    # '%pip install pillow lz4 python-dotenv' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        now initialize deep lake vectore store and store structured info about code smells in it
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        we will use 2 repos as a data sources:

        * pixel-dungeon - as an example of software that contains numerous code smells.
        * smells - a comprehensive classification of code smells. we will use structured information about code smells from this repository in our DeepLake database.
        """
    )
    return


app._unparsable_cell(
    r"""
    !git clone https://github.com/watabou/pixel-dungeon.git
    !git clone https://github.com/Luzkan/smells.git
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        get all files with structured info about code smells
        """
    )
    return


@app.cell
def _():
    from glob import glob
    from IPython.display import Markdown, display

    from langchain.vectorstores import DeepLake
    from langchain.document_loaders import TextLoader
    from langchain_text_splitters import (
        Language,
        RecursiveCharacterTextSplitter,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain.chains import RetrievalQA
    import deeplake
    return (
        ChatGoogleGenerativeAI,
        DeepLake,
        GoogleGenerativeAIEmbeddings,
        Language,
        Markdown,
        RecursiveCharacterTextSplitter,
        RetrievalQA,
        display,
        glob,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Set up your API key

        To run the following cell, your API key must be stored it in a Kaggle secret named GOOGLE_API_KEY.

        If you don't already have an API key, you can grab one from AI Studio. You can find detailed instructions in the docs.

        To make the key available through Kaggle secrets, choose Secrets from the Add-ons menu and follow the instructions to add your key or enable it for this notebook.
        """
    )
    return


@app.cell
def _(api_key):
    import os
    import dotenv


    os.environ["GOOGLE_API_KEY"] = api_key
    return


@app.cell
def _(glob):
    smells_match = "smells/content/smells/**/*.md"
    all_smells = glob(smells_match, recursive=True)
    # Print the first 10 files
    print(f"First 5 of {len(all_smells)} markdown files:")
    for file in all_smells[:5]:
        print(f"  {file}")
    return (all_smells,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Each file with a matching path will be loaded and split by RecursiveCharacterTextSplitter. only Markdown files with structured content will be processed
        """
    )
    return


@app.cell
def _(Language, RecursiveCharacterTextSplitter):
    # common seperators used for Python files
    RecursiveCharacterTextSplitter.get_separators_for_language(Language.MARKDOWN)
    return


@app.cell
def _():
    # '%pip install python-frontmatter' command supported automatically in marimo
    return


@app.cell
def _(all_smells):
    from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
    from langchain.schema import Document
    headers_to_split_on = [('#', 'Title'), ('##', 'Section'), ('###', 'Subsection')]
    docs = []
    for file_1 in all_smells:
        try:
            with open(file_1, 'r', encoding='utf-8') as f:
                content = f.read()
            if '---' in content:
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    markdown_content = '---' + parts[2]
                else:
                    markdown_content = content
            else:
                markdown_content = content
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            header_splits = markdown_splitter.split_text(markdown_content)
            for doc in header_splits:
                doc.metadata['source'] = file_1
            docs.extend(header_splits)
        except Exception as e:
            print(f'Error processing {file_1}: {e}')
    print(f'Created {len(docs)} chunks from {len(all_smells)} files')
    return RecursiveCharacterTextSplitter, docs


@app.cell
def _():
    # define path to database
    dataset_path = 'mem://deeplake/smells'
    return (dataset_path,)


@app.cell
def _(GoogleGenerativeAIEmbeddings):
    # define the embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return (embeddings,)


@app.cell
def _(DeepLake, dataset_path, docs, embeddings):
    smell_db = DeepLake.from_documents(docs, embeddings, dataset_path=dataset_path)
    return (smell_db,)


@app.cell
def _(smell_db):
    retriever = smell_db.as_retriever()
    retriever.search_kwargs['distance_metric'] = 'cos'
    retriever.search_kwargs['k'] = 20 # number of documents to return
    return (retriever,)


@app.cell
def _(ChatGoogleGenerativeAI):
    # define the chat model
    llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")
    return (llm,)


@app.cell
def _(RetrievalQA, llm, retriever):
    qa = RetrievalQA.from_llm(llm, retriever=retriever)
    return (qa,)


@app.cell
def _(Markdown, display, qa):
    # a helper function for calling retrival chain
    def call_qa_chain(prompt):
      response = qa.invoke(prompt)
      display(Markdown(response["result"]))
    return (call_qa_chain,)


@app.cell
def _(call_qa_chain):
    call_qa_chain("List a recommendations on how to get rid of 'God class' smell, rating from more simple to more sophisticated strategies.")
    return


@app.cell
def _(all_smells):
    import enum
    from pathlib import Path

    # Get all markdown files in the content/smells directory
    smell_files = all_smells

    # Create enum class dynamically
    def create_code_smell_enum():
        # Process each filename to create enum-friendly names
        enum_entries = {}
    
        for file_path in smell_files:
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

    # Display the enum members
    print(f"Created enum with {len(CodeSmell)} code smells:")
    for smell in CodeSmell:
        print(f"{smell.name} = '{smell.value}'")

    # Example usage
    print("\nExample usage:")
    print(f"CodeSmell.DEAD_CODE = '{CodeSmell.DEAD_CODE.value}'")
    print(f"CodeSmell.FEATURE_ENVY = '{CodeSmell.FEATURE_ENVY.value}'")

    # You can also look up an enum by value
    def get_smell_by_name(name):
        for smell in CodeSmell:
            if smell.value == name:
                return smell
        return None

    print("\nLooking up by name:")
    print(f"get_smell_by_name('dead-code') = {get_smell_by_name('dead-code')}")
    return CodeSmell, enum


@app.cell
def _():
    # '%pip install langchain_community langchain-text-splitters pypdf' command supported automatically in marimo
    return


@app.cell
def _(enum):
    from langchain.output_parsers import PydanticOutputParser
    from langchain.prompts import PromptTemplate
    from pydantic import BaseModel, Field
    from typing import List, Optional, Literal

    # Define your structured output schema using Pydantic
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

    # Create a prompt template that includes formatting instructions
    code_analysis_prompt = PromptTemplate(
        template="""
    You are an expert code analyst. Analyze the following code for code smells:

    ```java
    {code}
    {format_instructions}

    Only identify code smells from this list: {valid_smells} """, input_variables=["code", "valid_smells"], partial_variables={"format_instructions": parser.get_format_instructions()} )
    return BaseModel, Field, PromptTemplate, PydanticOutputParser, parser


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
    return (analyze_code_with_structure,)


@app.cell
def _(analyze_code_with_structure):
    def display_code_analysis(file_path): # Read the file content 
        with open(file_path, 'r') as f: code_content = f.read()
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
    return (display_code_analysis,)


@app.cell
def _(display_code_analysis):
    # Analyze a Java file from the pixel-dungeon repository
    java_file = "pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java"
    display_code_analysis(java_file)
    return


@app.cell
def _(
    BaseModel,
    Field,
    List_1,
    Optional_1,
    PromptTemplate,
    PydanticOutputParser,
    analyze_code_with_structure,
    enum,
):
    from typing import List, Optional, Dict, Any
    import json

    class EvaluationScore(str, enum.Enum):
        EXCELLENT = 'EXCELLENT'
        GOOD = 'GOOD'
        ACCEPTABLE = 'ACCEPTABLE'
        POOR = 'POOR'
        INCORRECT = 'INCORRECT'

    class SmellEvaluation(BaseModel):
        detected_smell: str = Field(description='The detected code smell type')
        location: str = Field(description='Where the smell was detected')
        ground_truth_match: Optional_1[str] = Field(None, description='The matching ground truth smell if any')
        score: EvaluationScore = Field(description='Evaluation score for this detection')
        justification: str = Field(description='Explanation for the rating')

    class CodeSmellEvaluationResult(BaseModel):
        overall_score: float = Field(description='Overall evaluation score out of 5')
        precision: float = Field(description='Ratio of correctly identified smells to all detections')
        recall: float = Field(description='Ratio of correctly identified smells to all actual smells')
        evaluations: List_1[SmellEvaluation] = Field(description='Individual smell evaluations')
        summary: str = Field(description='Summary of evaluation results')
    eval_template = '\n# Instruction\nYou are an expert evaluator specializing in code smell detection. Your task is to evaluate the quality of code smell detections by comparing them with ground truth data.\n\n# Evaluation\n## Metric Definition\nYou will be assessing code smell detection quality, which measures how accurately the system identifies:\n1. The correct type of code smell\n2. The correct location of the smell (file, line numbers, method)\n\n## Criteria\nSmell Type Accuracy: The detected smell type matches the actual smell type in the code.\nLocation Accuracy: The location identified for the smell (line numbers, method, class) matches where the smell actually exists.\nJustification Quality: The explanation provided for the smell makes sense and correctly describes the issue.\nRefactoring Relevance: The suggested refactoring is appropriate for the identified smell.\n\n## Rating Rubric\nEXCELLENT: Perfect match of smell type and exact location (score: 5).\nGOOD: Correct smell type with minor location imprecision (score: 4).\nACCEPTABLE: Partial match (either correct smell type or approximate location) (score: 3).\nPOOR: Wrong smell type but area of concern correctly identified (score: 2).\nINCORRECT: Completely incorrect detection (wrong smell type and location) (score: 1).\n\n## Evaluation Steps\nSTEP 1: For each detected smell, find any matching ground truth smells.\nSTEP 2: Evaluate the accuracy of the detected smell type.\nSTEP 3: Evaluate the precision of the location identified.\nSTEP 4: Assign a score based on the rating rubric.\nSTEP 5: Calculate overall precision and recall metrics.\n\n# Input Data\n## Ground Truth Smells\n{ground_truth}\n\n## Detected Smells\n{detected_smells}\n\n{format_instructions}\n'
    eval_parser = PydanticOutputParser(pydantic_object=CodeSmellEvaluationResult)
    eval_prompt = PromptTemplate(template=eval_template, input_variables=['ground_truth', 'detected_smells'], partial_variables={'format_instructions': eval_parser.get_format_instructions()})

    def evaluate_smell_detection(ground_truth_data, detected_smells, llm):
        """
        Evaluate the quality of code smell detection by comparing with ground truth.
    
        Args:
            ground_truth_data: Dictionary with ground truth smells
            detected_smells: Dictionary with detected smells
            llm: LangChain LLM instance
    
        Returns:
            Evaluation results with scores and metrics
        """
        ground_truth_str = json.dumps(ground_truth_data, indent=2)
        detected_str = json.dumps(detected_smells, indent=2)
        formatted_prompt = eval_prompt.format(ground_truth=ground_truth_str, detected_smells=detected_str)
        try:
            output = llm.invoke(formatted_prompt)
            parsed_output = eval_parser.parse(output.content)
            return parsed_output
        except Exception as e:
            print(f'Error during evaluation: {e}')
            print(f"Raw output: {(output.content if 'output' in locals() else 'No output')}")
            return None

    def create_ground_truth(file_path, manual_annotations):
        """Create ground truth data structure"""
        return {'file_path': file_path, 'smells': manual_annotations}

    def evaluate_code_analysis(file_path, manual_annotations, llm):
        """Run a full evaluation on a code file"""
        with open(file_path, 'r') as f:
            code_content = f.read()
        analysis_result = analyze_code_with_structure(code_content)
        if not analysis_result:
            print('Failed to analyze code')
            return None
        detected_smells = {'file_path': file_path, 'smells': []}
        for smell in analysis_result.smells_detected:
            detected_smells['smells'].append({'smell_type': smell.smell_type, 'location': smell.location, 'description': smell.description, 'severity': str(smell.severity), 'refactoring': smell.refactoring_suggestion})
        ground_truth = create_ground_truth(file_path, manual_annotations)
        evaluation = evaluate_smell_detection(ground_truth, detected_smells, llm)
        if evaluation:
            print(f'Overall Score: {evaluation.overall_score:.2f}/5.0')
            print(f'Precision: {evaluation.precision:.2f}')
            print(f'Recall: {evaluation.recall:.2f}')
            print(f'\nSummary: {evaluation.summary}\n')
            print('Individual Evaluations:')
            for i, eval_item in enumerate(evaluation.evaluations, 1):
                print(f'\n{i}. {eval_item.detected_smell} ({eval_item.location})')
                print(f'   Score: {eval_item.score.value}')
                print(f"   Matched with: {(eval_item.ground_truth_match if eval_item.ground_truth_match else 'No match')}")
                print(f'   Justification: {eval_item.justification}')
        return evaluation
    return (evaluate_code_analysis,)


@app.cell
def _(evaluate_code_analysis, llm):
    evaluation_files = {'pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java': [{'smell_type': 'MAGIC_NUMBER', 'location': 'multiple locations: lines 25-26, 37, 52-53, 85, 89', 'severity': 'MEDIUM'}, {'smell_type': 'LONG_METHOD', 'location': 'decorate() method, lines 74-101', 'severity': 'MEDIUM'}, {'smell_type': 'CONDITIONAL_COMPLEXITY', 'location': 'decorate() method, lines 76-89', 'severity': 'LOW'}, {'smell_type': 'DUPLICATED_CODE', 'location': 'tileName() and tileDesc() methods, lines 104-138', 'severity': 'LOW'}, {'smell_type': 'DEAD_CODE', 'location': 'map[i] == 63 condition in addVisuals method, line 151', 'severity': 'MEDIUM'}, {'smell_type': 'FEATURE_ENVY', 'location': 'addVisuals method, lines 149-153', 'severity': 'LOW'}, {'smell_type': 'PRIMITIVE_OBSESSION', 'location': 'throughout class, using boolean arrays and ints for terrain', 'severity': 'LOW'}]}
    java_file_1 = 'pixel-dungeon/src/com/watabou/pixeldungeon/levels/HallsLevel.java'
    evaluation = evaluate_code_analysis(java_file_1, evaluation_files, llm)
    return (evaluation_files,)


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
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
