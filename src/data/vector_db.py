"""
Vector database operations for code smell knowledge base.

This module provides functions to create and query a DeepLake vector database
containing structured information about code smells. The knowledge base is built
from markdown documentation files and used for Retrieval-Augmented Generation (RAG)
during smell detection.

**Implementation based on**: pipeline_reference/pipeline.py
"""

from glob import glob
from typing import List

from langchain.schema import Document
from langchain.vectorstores import DeepLake
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_smell_documents(smell_files: List[str]) -> List[Document]:
    """
    Load code smell documentation from markdown files and split by headers.

    Processes markdown files containing code smell definitions, examples, and
    refactoring guidelines. Handles frontmatter removal and splits documents
    by markdown headers for optimal retrieval.

    **Based on**: pipeline_reference/pipeline.py lines 193-218

    Args:
        smell_files: List of file paths to markdown documents (e.g., from glob)

    Returns:
        List of Document objects with content and metadata

    Example:
        >>> smell_files = glob("smells/content/smells/**/*.md", recursive=True)
        >>> documents = load_smell_documents(smell_files)
        >>> print(f"Loaded {len(documents)} document chunks")

    Notes:
        - Skips YAML frontmatter (content between `---` markers)
        - Splits on headers: # (Title), ## (Section), ### (Subsection)
        - Each document includes source file path in metadata
    """
    # Define markdown headers for splitting
    headers_to_split_on = [("#", "Title"), ("##", "Section"), ("###", "Subsection")]

    # Initialize markdown splitter
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    docs = []

    for file_path in smell_files:
        try:
            # Read file content
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Remove frontmatter if present (YAML between --- markers)
            if "---" in content:
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    # Skip frontmatter, keep content after second ---
                    markdown_content = "---" + parts[2]
                else:
                    markdown_content = content
            else:
                markdown_content = content

            # Split by markdown headers
            header_splits = markdown_splitter.split_text(markdown_content)

            # Add source file to metadata for each chunk
            for doc in header_splits:
                doc.metadata["source"] = file_path

            docs.extend(header_splits)

        except Exception as e:
            print(f"⚠ Error processing {file_path}: {e}")
            continue

    print(f"✓ Loaded {len(docs)} document chunks from {len(smell_files)} files")
    return docs


def create_smell_vector_db(
    documents: List[Document], dataset_path: str = "./data/deeplake/smells"
) -> DeepLake:
    """
    Create DeepLake vector database from smell documentation.

    Initializes a vector database with Google's text-embedding-004 model for
    semantic search over code smell documentation. Supports both in-memory
    (mem://) and persistent file system storage.

    **Based on**: pipeline_reference/pipeline.py lines 229-238

    Args:
        documents: List of Document objects (from load_smell_documents)
        dataset_path: Path for DeepLake dataset storage
                     - File system: "./data/deeplake/smells"
                     - In-memory (testing): "mem://deeplake/smells"

    Returns:
        DeepLake vector store instance

    Raises:
        ValueError: If GOOGLE_API_KEY environment variable not set
        Exception: If DeepLake initialization fails

    Example:
        >>> docs = load_smell_documents(smell_files)
        >>> vector_db = create_smell_vector_db(docs, "./data/deeplake/smells")
        >>> print(f"Vector DB created with {len(docs)} documents")

    Notes:
        - Requires GOOGLE_API_KEY environment variable
        - Uses Google's text-embedding-004 (768 dimensions)
        - First run creates dataset; subsequent runs load existing
    """
    import os

    # Verify API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError(
            "GOOGLE_API_KEY environment variable not set. "
            "Please add it to your .env file."
        )

    print(f"Initializing DeepLake vector database at: {dataset_path}")

    # Initialize Google embeddings model
    # text-embedding-004 is recommended for semantic search (768 dimensions)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    try:
        # Create DeepLake vector store from documents
        vector_db = DeepLake.from_documents(
            documents, embeddings, dataset_path=dataset_path
        )

        print(f"✓ Vector database created with {len(documents)} documents")
        return vector_db

    except Exception as e:
        raise Exception(
            f"Failed to create DeepLake vector database: {e}. "
            "Check GOOGLE_API_KEY and dataset_path permissions."
        ) from e


def get_retriever(vector_db: DeepLake, k: int = 20):
    """
    Configure and return a retriever from the vector database.

    Creates a retriever configured with cosine similarity for semantic search
    over code smell documentation. The retriever is used in RAG pipelines to
    provide context for LLM-based smell detection.

    **Based on**: pipeline_reference/pipeline.py lines 242-246

    Args:
        vector_db: DeepLake vector database instance
        k: Number of documents to retrieve per query (default: 20)

    Returns:
        Configured retriever instance

    Example:
        >>> vector_db = create_smell_vector_db(documents)
        >>> retriever = get_retriever(vector_db, k=20)
        >>> # Use in RAG chain
        >>> relevant_docs = retriever.get_relevant_documents("complex method smell")

    Notes:
        - Uses cosine distance metric for similarity
        - k=20 provides good context without overwhelming the LLM
        - Retriever returns Document objects with content and metadata
    """
    # Create retriever from vector database
    retriever = vector_db.as_retriever()

    # Configure search parameters
    retriever.search_kwargs["distance_metric"] = "cos"  # Cosine similarity
    retriever.search_kwargs["k"] = k  # Number of documents to return

    print(f"✓ Retriever configured (cosine similarity, k={k})")
    return retriever


def load_and_create_vector_db(
    smells_pattern: str = "smells/content/smells/**/*.md",
    dataset_path: str = "./data/deeplake/smells",
    k: int = 20,
) -> tuple[DeepLake, any]:
    """
    Convenience function to load documents, create vector DB, and get retriever.

    Combines all vector database setup steps into a single function call.
    Useful for quick initialization and testing.

    Args:
        smells_pattern: Glob pattern for smell markdown files
        dataset_path: Path for DeepLake dataset
        k: Number of documents for retriever to return

    Returns:
        Tuple of (vector_db, retriever)

    Example:
        >>> vector_db, retriever = load_and_create_vector_db()
        >>> docs = retriever.get_relevant_documents("long method")
        >>> print(f"Found {len(docs)} relevant documents")
    """
    # Find all smell documentation files
    smell_files = glob(smells_pattern, recursive=True)

    if not smell_files:
        raise FileNotFoundError(
            f"No files found matching pattern: {smells_pattern}. "
            "Make sure the smells repository is cloned."
        )

    # Load documents
    documents = load_smell_documents(smell_files)

    # Create vector database
    vector_db = create_smell_vector_db(documents, dataset_path)

    # Get retriever
    retriever = get_retriever(vector_db, k)

    return vector_db, retriever
