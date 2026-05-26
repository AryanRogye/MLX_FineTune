from models import Message
from datetime import datetime, timedelta
import dataclasses
from models import Segment

APPLE_EPOCH = datetime(2001, 1, 1)

"""
Conversation-aware model for segmenting message logs into distinct conversations.
"""
class ConversationSegmenter:
    def __init__(self, messages: list[Message] = []) -> list[list[Message]]:
        self.messages = messages

    def begin_conversation_segmentation(self):
        segments: list[Segment] = []

        start = 0

        while start < len(self.messages):
            end: int = self.determine_boundary(start)
            if end == -1:
                print(f"No boundary found starting from index {start}, stopping.")
                break
            segments.append(Segment(start, end - 1))
            start = end

        print("Total Segments:", len(segments))

        messages_by_segment: list[list[Message]] = []
        for segment in segments:
            segment_messages = self.messages[segment.start_index:segment.end_index + 1]
            filtered_messages: list[Message] = []
            for message in segment_messages:
                if message.text and message.text != "[Attachment/No Text]":
                    filtered_messages.append(message)
            if len(filtered_messages) > 0 and filtered_messages[-1].is_from_me:
                messages_by_segment.append(filtered_messages)
        
        return messages_by_segment
        

    """
    Determines the boundaries of conversations based on message content and metadata like dates, and gaps
    Returns -1 if no boundary is found, otherwise returns the index of the message that is determined to be a boundary
    """
    def determine_boundary(self, start_index: int) -> int:

        # Boundary checks
        if start_index < 0 or start_index >= len(self.messages):
            return -1

        for i in range(start_index + 1, len(self.messages)):
            
            previous = self.messages[i - 1]
            current = self.messages[i]

            if self._is_boundary(previous, current):
                return i
        return -1
    
    """
    Helpers
    """
    
    """
    Determines if there is a boundary between two messages based on time gap and other heuristics.
    """
    def _is_boundary(self, previous: Message, current: Message) -> bool:
        time_gap = self.apple_to_datetime(current.date) - self.apple_to_datetime(previous.date)
    
        # hard boundary: 4+ hours is always a new conversation
        if time_gap > timedelta(hours=4):
            return True
    
        # soft boundary: 30 min gap only if not a direct reply
        if time_gap > timedelta(minutes=30):
            # if same sender continuing, less likely to be a new convo
            if previous.is_from_me == current.is_from_me:
                return False
            return True
    
        return False

    """
    Apple's epoch starts on January 1, 2001, so we need to convert the timestamps to human-readable format
    """
    def apple_to_datetime(self, ts):
        return APPLE_EPOCH + timedelta(seconds=ts / 1e9)