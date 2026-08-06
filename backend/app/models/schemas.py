from typing import List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: str = Field(description="Role of the message author (system, user, assistant)")
    content: str = Field(description="Content of the message")

class ChatRequest(BaseModel):
    prompt: str = Field(description="The user query or prompt")
    collection_name: Optional[str] = Field(default=None, description="The Qdrant collection to query")
    history: Optional[List[ChatMessage]] = Field(default=None, description="Conversation history")
    top_k: Optional[int] = Field(default=4, description="Number of final context blocks to feed the LLM")

class DocumentInfo(BaseModel):
    file_name: str = Field(description="Name of the file")
    chunk_count: int = Field(description="Number of chunks stored for this file")

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class CreateCollectionRequest(BaseModel):
    name: str = Field(description="Name of the collection to create")
