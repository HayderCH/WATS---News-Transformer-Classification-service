#!/usr/bin/env python3
"""Data ingestion script for news articles into vector database."""

import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS


class NewsDataIngestion:
    """Handle ingestion of news data into vector stores."""

    def __init__(self):
        """Initialize with embeddings model."""
        self.embeddings = SentenceTransformerEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore_path = Path("data/vectorstores/news_articles")
        self.vectorstore_path.parent.mkdir(parents=True, exist_ok=True)

    def load_huffpost_data(
        self, data_path: str = "data/raw/huffpost"
    ) -> List[Document]:
        """Load HuffPost news data from JSON Lines files.

        Args:
            data_path: Path to the raw data directory

        Returns:
            List of Document objects for vector storage
        """

        documents = []
        data_dir = Path(data_path)

        if not data_dir.exists():
            print(
                f"⚠️  Data directory {data_path} not found. Skipping HuffPost data loading."
            )
            return documents

        # Look for JSON files
        json_files = list(data_dir.glob("*.json"))
        if not json_files:
            print(f"⚠️  No JSON files found in {data_path}")
            return documents

        print(f"📂 Found {len(json_files)} JSON files in {data_path}")

        for json_file in json_files:
            print(f"📖 Loading {json_file.name}...")
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            article = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"⚠️  JSON decode error at line {line_num + 1}: {e}")
                            continue

                        # Extract fields
                        headline = str(article.get("headline", ""))
                        category = str(article.get("category", ""))
                        authors = str(article.get("authors", ""))
                        date = str(article.get("date", ""))
                        description = str(article.get("short_description", ""))
                        link = str(article.get("link", ""))

                        if not headline:
                            continue  # Skip articles without headlines

                        # Combine headline and description
                        full_content = f"{headline}"
                        if description:
                            full_content += f". {description}"

                        # Create metadata
                        metadata = {
                            "headline": headline,
                            "category": category,
                            "authors": authors,
                            "date": date,
                            "link": link,
                            "source": "huffpost",
                            "file": json_file.name,
                        }

                        # Create Document object
                        doc = Document(page_content=full_content, metadata=metadata)
                        documents.append(doc)

                        # Removed limit for full dataset ingestion

                print(f"✅ Loaded {len(documents)} articles from {json_file.name}")

            except Exception as e:
                print(f"❌ Error loading {json_file.name}: {e}")
                continue

        print(f"📊 Total documents loaded: {len(documents)}")
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

        # Create FAISS vector store
        vectorstore = FAISS.from_documents(documents, self.embeddings)

        print("✅ Vector store created successfully!")
        return vectorstore

    def save_vectorstore(self, vectorstore: FAISS, path: str = None) -> None:
        """Save vector store to disk.

        Args:
            vectorstore: FAISS vector store to save
            path: Path to save to (default: self.vectorstore_path)
        """
        save_path = Path(path) if path else self.vectorstore_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"💾 Saving vector store to {save_path}...")
        vectorstore.save_local(str(save_path))
        print("✅ Vector store saved!")

    def load_vectorstore(self, path: str = None) -> FAISS:
        """Load vector store from disk.

        Args:
            path: Path to load from (default: self.vectorstore_path)

        Returns:
            FAISS vector store
        """
        load_path = Path(path) if path else self.vectorstore_path

        if not load_path.exists():
            raise FileNotFoundError(f"Vector store not found at {load_path}")

        print(f"📂 Loading vector store from {load_path}...")
        vectorstore = FAISS.load_local(
            str(load_path), self.embeddings, allow_dangerous_deserialization=True
        )
        print("✅ Vector store loaded!")
        return vectorstore

    def run_ingestion(
        self, data_path: str = "data/raw/huffpost", force_recreate: bool = False
    ):
        """Run the complete data ingestion pipeline.

        Args:
            data_path: Path to raw data
            force_recreate: Whether to force recreation of vector store
        """
        print("🚀 Starting news data ingestion pipeline...")

        # Check if vector store already exists
        if self.vectorstore_path.exists() and not force_recreate:
            print(
                "ℹ️  Vector store already exists. Use force_recreate=True to recreate."
            )
            return

        # Load documents
        documents = self.load_huffpost_data(data_path)

        if not documents:
            print("❌ No documents loaded. Cannot create vector store.")
            return

        # Create vector store
        vectorstore = self.create_vectorstore(documents)

        # Save vector store
        self.save_vectorstore(vectorstore)

        print("🎉 News data ingestion completed successfully!")
        print(f"📊 Vector store contains {len(documents)} documents")


def main():
    """Main function for command line usage."""
    import argparse

    parser = argparse.ArgumentParser(description="News Data Ingestion for RAG Chatbot")
    parser.add_argument(
        "--data-path", default="data/raw/huffpost", help="Path to raw news data"
    )
    parser.add_argument(
        "--force-recreate", action="store_true", help="Force recreation of vector store"
    )

    args = parser.parse_args()

    # Run ingestion
    ingestion = NewsDataIngestion()
    ingestion.run_ingestion(args.data_path, args.force_recreate)


if __name__ == "__main__":
    main()
