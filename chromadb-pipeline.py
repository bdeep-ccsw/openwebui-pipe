"""
title: ChromaDB RAG Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a local ChromaDB instance.
requirements: chromadb, openai
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os
from openai import OpenAI


class Pipeline:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.openai_client = OpenAI()

    async def on_startup(self):
        """Initialize ChromaDB client and collection on startup."""
        os.environ["OPENAI_API_KEY"] = "NONE"

        # Initialize ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="/chroma_db")

        # Create or get the collection with an OpenAI embedding function
        self.collection = self.chroma_client.get_or_create_collection(
            name="rag_documents",
            embedding_function=OpenAIEmbeddingFunction()
        )

    async def on_shutdown(self):
        """Handle cleanup when the server stops."""
        pass

    def retrieve_documents(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant documents from ChromaDB."""
        results = self.collection.query(query_texts=[query], n_results=top_k)
        return results["documents"][0] if results["documents"] else []

    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate a response using OpenAI's model with retrieved context."""
        prompt = f"Context:\n{'\n\n'.join(context)}\n\nQuestion: {query}\nAnswer:"
        response = self.openai_client.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"].strip()

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline logic for processing queries."""
        context = self.retrieve_documents(user_message)
        response = self.generate_response(user_message, context)
        return response
