#!/usr/bin/env python3
"""Data ingestion script for documentation into vector database."""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


class DocumentationIngestion:
    """Handle ingestion of documentation into vector stores."""

    def __init__(self):
        """Initialize with embeddings model."""
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore_path = Path("data/vectorstores/documentation")
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)

    def load_documentation(self) -> List[Document]:
        """Load documentation files from README.md and docs/ directory.

        Returns:
            List of Document objects for vector storage
        """
        documents = []

        # Load README.md
        readme_path = Path("README.md")
        if readme_path.exists():
            print("📖 Loading README.md...")
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "title": "README",
                        "source": "README.md",
                        "type": "documentation",
                        "file_path": str(readme_path),
                    },
                )
                documents.append(doc)
                print(f"✅ Loaded README.md ({len(content)} characters)")
            except Exception as e:
                print(f"⚠️  Error loading README.md: {e}")

        # Load docs/*.md files
        docs_dir = Path("docs")
        if docs_dir.exists():
            md_files = list(docs_dir.glob("*.md"))
            print(f"📂 Found {len(md_files)} documentation files in docs/")

            for md_file in md_files:
                print(f"📖 Loading {md_file.name}...")
                try:
                    with open(md_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    # Extract title from first line (usually # Title)
                    lines = content.split("\n", 1)
                    if lines[0].startswith("#"):
                        title = lines[0].strip("# ").strip()
                    else:
                        title = md_file.stem

                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={
                            "title": title,
                            "source": md_file.name,
                            "type": "documentation",
                            "file_path": str(md_file),
                        },
                    )
                    documents.append(doc)
                    print(f"✅ Loaded {md_file.name} " f"({len(content)} characters)")
                except Exception as e:
                    print(f"⚠️  Error loading {md_file.name}: {e}")
        else:
            print("⚠️  docs/ directory not found")

        return documents

    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """Create FAISS vector store from documents.

        Args:
            documents: List of Document objects

        Returns:
            FAISS vector store
        """
        if not documents:
            raise ValueError("No documents provided for vector store creation")

        print(f"🏗️  Creating vector store with {len(documents)} documents...")

        # Create vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)

        print("✅ Vector store created successfully")
        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS) -> None:
        """Save vector store to disk.

        Args:
            vectorstore: FAISS vector store to save
        """
        print(f"💾 Saving vector store to {self.vectorstore_path}...")
        vectorstore.save_local(str(self.vectorstore_path))
        print("✅ Vector store saved successfully")

    def run_ingestion(self) -> None:
        """Run the complete documentation ingestion process."""
        print("🚀 Starting documentation ingestion...")

        # Load documentation
        documents = self.load_documentation()

        if not documents:
            print(
                "❌ No documentation files found. "
                "Please ensure README.md and docs/ exist."
            )
            return

        # Create vector store
        vectorstore = self.create_vectorstore(documents)

        # Save vector store
        self.save_vectorstore(vectorstore)

        print("🎉 Documentation ingestion completed!")
        print(f"📊 Total documents processed: {len(documents)}")


def main():
    """Main entry point."""
    ingestor = DocumentationIngestion()
    ingestor.run_ingestion()


if __name__ == "__main__":
    main()
