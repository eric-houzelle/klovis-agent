"""Discord bot integration for the autonomous agent.

Provides a PerceptionSource backed by a Discord bot that receives messages
via DMs or @mentions and feeds them as REQUEST events into the daemon loop.
After the agent processes a request, the result can be sent back to the
originating channel/DM through :meth:`send_reply`.

Setup:
  1. Create a bot at https://discord.com/developers/applications
  2. Enable the "Message Content" privileged intent
  3. Invite the bot (OAuth2 → bot scope, Send Messages + Read Message History)
  4. Set DISCORD_BOT_TOKEN in your .env

The integration is fully optional — if discord.py is not installed or
DISCORD_BOT_TOKEN is not set, no source is registered and no error is raised.
"""

from __future__ import annotations

import asyncio
import os
from collections import deque
from typing import TYPE_CHECKING, Any

import structlog

from klovis_agent.perception.base import Event, EventKind, PerceptionSource

if TYPE_CHECKING:
    from klovis_agent.result import AgentResult

logger = structlog.get_logger(__name__)

_MAX_MESSAGE_LEN = 2000


def _extract_direct_user_response(result: AgentResult) -> str:
    """Extract the best user-facing response from step outputs.

    Preference order:
    1) Explicit direct responses ("response")
    2) Text payloads from successful content-producing tools ("content")
    """
    for step in reversed(result.steps):
        if step.status != "success":
            continue

        outputs = step.outputs
        resp = outputs.get("response")
        if isinstance(resp, str) and resp.strip():
            return resp.strip()

        content = outputs.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    return ""


def format_discord_reply(result: AgentResult) -> str:
    """Build a Discord message from an AgentResult.

    Shows the summary as a quoted "thinking" block, followed by the
    direct response extracted from the last step (if any).
    """
    parts: list[str] = []

    direct = _extract_direct_user_response(result)

    if result.summary:
        parts.append(f"> 💭 *{result.summary}*")

    if direct and direct != result.summary:
        parts.append(direct)

    if not parts:
        parts.append("Tâche terminée.")

    return "\n\n".join(parts)


def _chunk_message(text: str, limit: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split a long message into Discord-safe chunks."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    return chunks


class DiscordPerceptionSource(PerceptionSource):
    """Discord bot that acts as a perception source for the daemon.

    Messages are captured in two situations:
    - Direct messages (DMs) to the bot
    - @mentions in a server channel

    When ``allowed_user_ids`` is set, only those users can interact with the
    bot.  Leave empty/None to allow everyone (not recommended in public servers).
    """

    def __init__(
        self,
        token: str,
        *,
        allowed_user_ids: list[int] | None = None,
    ) -> None:
        self._token = token
        self._allowed_users: set[int] = set(allowed_user_ids or [])
        self._pending: deque[Event] = deque()
        self._client: Any = None
        self._ready = asyncio.Event()
        self._started = False

    @property
    def name(self) -> str:
        return "discord"

    async def start(self) -> None:
        """Start the Discord bot in the background.

        Must be called once before the daemon loop begins.  The method
        returns as soon as the bot has connected and is ready.
        """
        if self._started:
            return

        try:
            import discord
        except ImportError:
            logger.error(
                "discord_import_failed",
                hint="Install discord.py: pip install 'discord.py>=2.3.0'",
            )
            raise

        intents = discord.Intents.default()
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready() -> None:
            logger.info("discord_ready", user=str(self._client.user))
            self._ready.set()

        @self._client.event
        async def on_message(message: discord.Message) -> None:
            if message.author == self._client.user:
                return
            if not self._should_process(message):
                return

            self._pending.append(Event(
                source="discord",
                kind=EventKind.REQUEST,
                title=message.content[:200],
                detail=message.content,
                metadata={
                    "message_id": message.id,
                    "channel_id": message.channel.id,
                    "author_id": message.author.id,
                    "author_name": str(message.author),
                    "guild_id": message.guild.id if message.guild else None,
                },
            ))
            logger.info(
                "discord_message_received",
                author=str(message.author),
                length=len(message.content),
            )

        asyncio.create_task(self._client.start(self._token))
        await asyncio.wait_for(self._ready.wait(), timeout=30)
        self._started = True
        logger.info("discord_bot_started", user=str(self._client.user))

    def _should_process(self, message: Any) -> bool:
        import discord

        if self._allowed_users and message.author.id not in self._allowed_users:
            return False

        if isinstance(message.channel, discord.DMChannel):
            return True

        if self._client.user in message.mentions:
            return True

        return False

    async def poll(self) -> list[Event]:
        events = list(self._pending)
        self._pending.clear()
        return events

    async def send_reply(
        self, event_metadata: dict[str, Any], content: str,
    ) -> None:
        """Send a response back to the channel/DM that originated the event."""
        if not self._client:
            return

        channel_id = event_metadata.get("channel_id")
        if not channel_id:
            return

        channel = self._client.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self._client.fetch_channel(channel_id)
            except Exception as exc:
                logger.warning("discord_fetch_channel_failed", error=str(exc))
                return

        for chunk in _chunk_message(content):
            try:
                await channel.send(chunk)
            except Exception as exc:
                logger.warning("discord_send_failed", error=str(exc))
                break

    async def stop(self) -> None:
        """Gracefully disconnect the bot."""
        if self._client and not self._client.is_closed():
            await self._client.close()
            logger.info("discord_bot_stopped")


def load_discord_config() -> dict[str, Any]:
    """Load Discord config from environment variables.

    Returns a dict with ``token`` and optionally ``allowed_user_ids``.
    Returns an empty dict if DISCORD_BOT_TOKEN is not set.
    """
    token = os.environ.get("DISCORD_BOT_TOKEN", "")
    if not token:
        return {}

    config: dict[str, Any] = {"token": token}

    raw_users = os.environ.get("DISCORD_ALLOWED_USERS", "")
    if raw_users:
        config["allowed_user_ids"] = [
            int(uid.strip()) for uid in raw_users.split(",") if uid.strip()
        ]

    return config
