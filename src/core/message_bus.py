#!/usr/bin/env python3
"""
Message Bus for agent communication using Redis Pub/Sub.
Handles message routing, persistence, and delivery guarantees.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import asdict
import aioredis
from enum import Enum

from .gemini_client import MessageEnvelope, MessageType

logger = logging.getLogger(__name__)

class ChannelType(Enum):
    """Channel types for message routing."""
    AGENT_BROADCAST = "agent.broadcast"
    COORDINATOR = "coordinator"
    INFO_REQUESTS = "info.requests"
    INFO_RESPONSES = "info.responses"
    SYSTEM_OPS = "system.ops"
    OPERATION_AGENT = "operation.agent"
    AGENT_LOGS = "agent.logs"
    AGENT_HEARTBEATS = "agent.heartbeats"

class MessageBus:
    """Redis-based message bus for agent communication."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, 
                 redis_db: int = 0, redis_password: Optional[str] = None):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_password = redis_password
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.subscribers: Dict[str, Set[Callable]] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.running = False
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
    async def connect(self):
        """Connect to Redis and initialize message bus."""
        try:
            self.redis = aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}",
                password=self.redis_password,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await self.redis.ping()
            logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
            
            # Initialize pubsub
            self.pubsub = self.redis.pubsub()
            
            # Start message processing
            self.running = True
            asyncio.create_task(self._process_messages())
            asyncio.create_task(self._process_queue())
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self):
        """Disconnect from Redis and cleanup."""
        self.running = False
        
        if self.pubsub:
            await self.pubsub.close()
        
        if self.redis:
            await self.redis.close()
        
        logger.info("Disconnected from Redis")
    
    def subscribe(self, channel: str, handler: Callable[[MessageEnvelope], None]):
        """Subscribe to a channel."""
        if channel not in self.subscribers:
            self.subscribers[channel] = set()
        self.subscribers[channel].add(handler)
        logger.debug(f"Subscribed to channel: {channel}")
    
    def unsubscribe(self, channel: str, handler: Callable[[MessageEnvelope], None]):
        """Unsubscribe from a channel."""
        if channel in self.subscribers:
            self.subscribers[channel].discard(handler)
            if not self.subscribers[channel]:
                del self.subscribers[channel]
        logger.debug(f"Unsubscribed from channel: {channel}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable[[MessageEnvelope], None]):
        """Register a handler for a specific message type."""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        logger.debug(f"Registered handler for message type: {message_type}")
    
    async def publish(self, message: MessageEnvelope, channel: Optional[str] = None) -> bool:
        """Publish a message to a channel."""
        try:
            if not self.redis:
                logger.error("Redis not connected")
                return False
            
            # Use specified channel or determine from message
            target_channel = channel or self._get_channel_for_message(message)
            
            # Serialize message
            message_data = json.dumps(asdict(message), default=str)
            
            # Publish to Redis
            await self.redis.publish(target_channel, message_data)
            
            logger.debug(f"Published message {message.id} to {target_channel}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False
    
    async def send_message(self, to: str, message_type: MessageType, payload: Dict[str, Any], 
                          reply_to: Optional[str] = None, timeout: float = 30.0) -> Optional[MessageEnvelope]:
        """Send a message and wait for response."""
        message_id = str(uuid.uuid4())
        
        # Create message envelope
        message = MessageEnvelope(
            id=message_id,
            from_agent="message_bus",
            to=to,
            type=message_type,
            timestamp=datetime.now().isoformat(),
            reply_to=reply_to,
            payload=payload
        )
        
        # Create future for response
        response_future = asyncio.Future()
        self.pending_responses[message_id] = response_future
        
        try:
            # Publish message
            success = await self.publish(message)
            if not success:
                return None
            
            # Wait for response with timeout
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Message {message_id} timed out")
            return None
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return None
        finally:
            # Cleanup
            self.pending_responses.pop(message_id, None)
    
    async def broadcast(self, message_type: MessageType, payload: Dict[str, Any], 
                       from_agent: str = "system") -> bool:
        """Broadcast a message to all agents."""
        message = MessageEnvelope(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to="*",
            type=message_type,
            timestamp=datetime.now().isoformat(),
            payload=payload
        )
        
        return await self.publish(message, ChannelType.AGENT_BROADCAST.value)
    
    def _get_channel_for_message(self, message: MessageEnvelope) -> str:
        """Determine the appropriate channel for a message."""
        if message.to == "*":
            return ChannelType.AGENT_BROADCAST.value
        elif message.to == "coordinator":
            return ChannelType.COORDINATOR.value
        elif message.type == MessageType.INFO_REQUEST:
            return ChannelType.INFO_REQUESTS.value
        elif message.type == MessageType.INFO_REQUEST:
            return ChannelType.INFO_RESPONSES.value
        elif message.type in [MessageType.OPERATION_REQUEST, MessageType.OPERATION_RESPONSE]:
            return ChannelType.OPERATION_AGENT.value
        elif message.type == MessageType.LOG:
            return ChannelType.AGENT_LOGS.value
        elif message.type == MessageType.HEARTBEAT:
            return ChannelType.AGENT_HEARTBEATS.value
        else:
            return ChannelType.SYSTEM_OPS.value
    
    async def _process_messages(self):
        """Process incoming messages from Redis."""
        if not self.pubsub:
            return
        
        try:
            # Subscribe to all channels
            for channel in self.subscribers.keys():
                await self.pubsub.subscribe(channel)
            
            # Listen for messages
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    await self._handle_incoming_message(message["channel"], message["data"])
                    
        except Exception as e:
            logger.error(f"Error processing messages: {e}")
    
    async def _handle_incoming_message(self, channel: str, data: str):
        """Handle an incoming message."""
        try:
            # Parse message
            message_data = json.loads(data)
            message = MessageEnvelope(**message_data)
            
            # Check if this is a response to a pending request
            if message.id in self.pending_responses:
                future = self.pending_responses.pop(message.id)
                if not future.done():
                    future.set_result(message)
                return
            
            # Route to channel subscribers
            if channel in self.subscribers:
                for handler in self.subscribers[channel]:
                    try:
                        await self._call_handler(handler, message)
                    except Exception as e:
                        logger.error(f"Error in channel handler: {e}")
            
            # Route to message type handlers
            if message.type in self.message_handlers:
                for handler in self.message_handlers[message.type]:
                    try:
                        await self._call_handler(handler, message)
                    except Exception as e:
                        logger.error(f"Error in message type handler: {e}")
            
            # Add to processing queue
            await self.message_queue.put(message)
            
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
    
    async def _call_handler(self, handler: Callable, message: MessageEnvelope):
        """Call a message handler safely."""
        if asyncio.iscoroutinefunction(handler):
            await handler(message)
        else:
            handler(message)
    
    async def _process_queue(self):
        """Process messages from the internal queue."""
        while self.running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                # Additional processing can be added here
                logger.debug(f"Processed message: {message.id}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message queue: {e}")
    
    async def get_message_history(self, channel: str, limit: int = 100) -> List[MessageEnvelope]:
        """Get message history for a channel (if using Redis Streams)."""
        # This would require Redis Streams implementation
        # For now, return empty list
        return []
    
    async def store_message(self, message: MessageEnvelope) -> bool:
        """Store a message for persistence."""
        try:
            if not self.redis:
                return False
            
            # Store in Redis with TTL
            key = f"message:{message.id}"
            message_data = json.dumps(asdict(message), default=str)
            await self.redis.setex(key, 3600, message_data)  # 1 hour TTL
            return True
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return False

# Global message bus instance
_message_bus: Optional[MessageBus] = None

def get_message_bus() -> MessageBus:
    """Get the global message bus instance."""
    global _message_bus
    if _message_bus is None:
        from src.config.settings import settings
        _message_bus = MessageBus(
            redis_host=settings.REDIS_HOST,
            redis_port=settings.REDIS_PORT,
            redis_db=settings.REDIS_DB,
            redis_password=settings.REDIS_PASSWORD
        )
    return _message_bus

async def initialize_message_bus():
    """Initialize the global message bus."""
    bus = get_message_bus()
    await bus.connect()
    return bus
