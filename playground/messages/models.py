from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Handle:
    rowid: int
    id: str
    service: str

APPLE_EPOCH = datetime(2001, 1, 1)

@dataclass
class Message:
    rowid: int
    text: str
    is_from_me: bool
    date: int
    service: str
    """
    Generate a Display
    [0] 2026-05-26 02:18 Me: [Attachment]
    [1] 2026-05-26 02:18 Friend: Why is he asking consent
    [2] 2026-05-26 02:18 Me: [Attachment]
    [3] 2026-05-26 02:18 Friend: 😭😭😭😭
    [4] 2026-05-26 02:25 Friend: Do u have all the messages stored tho
    """
    def apple_to_datetime(self, ts: int) -> datetime:
        return APPLE_EPOCH + timedelta(seconds=ts / 1e9)


    def display(self, index: int) -> str:
        date_str = self.apple_to_datetime(self.date).strftime("%Y-%m-%d %H:%M:%S")
        from_str = "Me" if self.is_from_me else "Friend"
        text_str = self.text if self.text else "[Attachment]"
        return f"[{index}] {date_str} {from_str}: {text_str}"


@dataclass
class Segment:
    start_index: int
    end_index: int
