import asyncio
import logging

logger = logging.getLogger(__name__)

class ChatGate:
    def __init__(self):
        self._block_chat = False
        self._active_chats = 0
        self._cond = asyncio.Condition()

    def block_new_chats(self):
        logger.info("ChatGate: Blocking incoming chat requests.")
        self._block_chat = True

    def allow_chats(self):
        logger.info("ChatGate: Unblocking incoming chat requests.")
        self._block_chat = False
        # Schedule notification to wake up any blocked tasks
        asyncio.create_task(self._notify_all())

    async def _notify_all(self):
        async with self._cond:
            self._cond.notify_all()

    async def enter_chat(self):
        async with self._cond:
            while self._block_chat:
                logger.info("ChatGate: Chat request blocked due to active GPU maintenance. Waiting...")
                await self._cond.wait()
            self._active_chats += 1
            logger.debug(f"ChatGate: Entered chat. Active chats: {self._active_chats}")

    async def exit_chat(self):
        async with self._cond:
            self._active_chats -= 1
            logger.debug(f"ChatGate: Exited chat. Active chats: {self._active_chats}")
            if self._active_chats == 0:
                self._cond.notify_all()

    async def wait_for_active_chats(self):
        logger.info("ChatGate: Waiting for currently running chat requests to drain...")
        async with self._cond:
            while self._active_chats > 0:
                logger.info(f"ChatGate: Active chats remaining: {self._active_chats}. Waiting...")
                await self._cond.wait()
        logger.info("ChatGate: All active chat requests drained.")

chat_gate = ChatGate()
