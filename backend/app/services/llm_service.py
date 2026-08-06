import logging
from typing import List, Dict, AsyncGenerator
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.config import settings

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Initialize LangChain's ChatOpenAI pointing to vLLM's local endpoint
        logger.info(f"Connecting ChatOpenAI to vLLM server at: {settings.VLLM_URL} using model: {settings.MODEL_NAME}")
        self.chat = ChatOpenAI(
            model=settings.MODEL_NAME,
            base_url=settings.VLLM_URL,
            api_key="not-needed",  # Prevents OpenAI SDK from throwing a missing credentials error
            temperature=0.1,       # Highly deterministic for tech support QA
            max_tokens=1024,
            streaming=True
        )


    async def stream_chat(
        self, 
        prompt: str, 
        context: str, 
        history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Formats a system prompt + context + query using LangChain message structures,
        then streams the generated tokens asynchronously from vLLM.
        """
        # 1. Initialize with system instruction prompt
        messages = [
            SystemMessage(content=settings.SYSTEM_INSTRUCTION)
        ]

        # 2. Append conversation history
        if history:
            for msg in history:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # 3. Format the final user message with RAG context
        user_prompt_with_context = (
            f"Reference Documents Context:\n"
            f"-----------------------------\n"
            f"{context}\n"
            f"-----------------------------\n\n"
            f"User Question: {prompt}"
        )
        messages.append(HumanMessage(content=user_prompt_with_context))

        logger.info(f"Streaming prompt response from local vLLM...")

        try:
            # 4. Stream response asynchronously from ChatOpenAI
            async for chunk in self.chat.astream(messages):
                token = chunk.content
                if token:
                    yield token
        except Exception as e:
            logger.error(f"Error streaming response via ChatOpenAI: {e}", exc_info=True)
            yield f"Error during streaming execution: {str(e)}"

# Initialize a singleton instance
llm_service = LLMService()
