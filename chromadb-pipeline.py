"""
title: ChromaDB RAG Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a local ChromaDB instance.
requirements: chromadb, sentence-transformers
"""

from typing import List, Union, Generator, Iterator
import chromadb

class Pipeline:
    def __init__(self):
        self.client = None
        self.collection = None

    async def on_startup(self):
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path="/home/bdeep/code/llm/pipelines/chroma_db")
        self.collection = self.client.get_or_create_collection(name="document_collection")

    async def on_shutdown(self):
        # Shutdown logic if needed
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Ensure ChromaDB is ready
        if not self.collection:
            return "Error: ChromaDB collection not initialized."

        # Retrieve most relevant documents
        results = self.collection.query(
            
            query_texts=[user_message], n_results=3
        )

        # Combine results into a response
        #count = self.collection.count()
        #retrieved_docs = [doc for doc in results["documents"][0]]
        #response_text = "Relevant Info: \n" + "\n\n".join(retrieved_docs)
        #response_text += f"[{retrieved_docs}]\n"
        #response_text += f"DOCS:[{retrieved_docs}]\n"
        #response_text = f"COUNT:[{count}]\n"
        #DOCS:[{retrieved_docs}]\n"
        query_engine = self.index.as_query_engine(streaming=True)
        response_text = query_engine.query(retrieved_docs)
        return response_text
