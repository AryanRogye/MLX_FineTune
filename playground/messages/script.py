"""
-- all handles
SELECT ROWID, id, service FROM handle LIMIT ?;

-- last message date for a handle
SELECT date FROM message WHERE handle_id = ? ORDER BY date DESC LIMIT 1;

-- last message text for a handle
SELECT text, attributedBody FROM message WHERE handle_id = ? ORDER BY date DESC LIMIT 1;

-- messages with a specific person
SELECT * FROM message WHERE handle_id = ? ORDER BY date DESC LIMIT ?;

-- change detection
SELECT guid, date, is_from_me FROM message ORDER BY date DESC LIMIT 1;
"""


from models import Handle, Message
from contacts import load_contacts

from jsonl_formatter import JsonLFormatter
from conversation_segmenter import ConversationSegmenter
import argparse

import sqlite3
import os
from pprint import pprint
from rich.table import Table
from rich.console import Console
from datetime import datetime, timedelta
from pathlib import Path

"""
Apple's epoch starts on January 1, 2001, so we need to convert the timestamps to human-readable format
"""
APPLE_EPOCH = datetime(2001, 1, 1)
def apple_to_datetime(ts: int) -> datetime:
    return APPLE_EPOCH + timedelta(seconds=ts / 1e9)

"""
Filter Handles to where the matching.id's exist in the id's of the handles
"""
def normalize(s: str) -> str:
    return ''.join(filter(str.isdigit, s))


def print_messages(selected_handle: Handle, messages_data: list[Message], console: Console):
    messages_table = Table(title=f"Messages for {selected_handle.id}")
    messages_table.add_column("Date", style="cyan")
    messages_table.add_column("From", style="magenta")
    messages_table.add_column("Text", style="green")
    for message in messages_data:
        date_str = apple_to_datetime(message.date).strftime("%Y-%m-%d %H:%M:%S")
        from_str = "Me" if message.is_from_me else selected_handle.id
        messages_table.add_row(date_str, from_str, message.text)
    console.print(messages_table)

output_path = Path(f"outputs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_boundaries.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, help="ROWID of the handle to load messages for")
    parser.add_argument("--system-prompt", type=str, help="System prompt to use for the conversation segmentation")
    return parser.parse_args()

def get_id() -> int:
    args = get_args()
    if args.id is not None:
        return args.id

    return int(console.input("Enter the ROWID of the handle you want to see messages for: "))

def get_system_prompt() -> str:
    args = get_args()
    if args.system_prompt is not None:
        return args.system_prompt

    return console.input("Enter the system prompt for conversation segmentation: ")

def clean_text(text: str) -> str:
    return (
        text
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .strip()
    )


console = Console()

# Handle Tables
handleTables = Table(title="Messages Handles")
handleTables.add_column("ROWID", justify="right", style="cyan", no_wrap=True)
handleTables.add_column("ID", style="magenta")
handleTables.add_column("Service", style="green")

# This is where the messages app stores its data on macOS
db_path = os.path.expanduser("~/Library/Messages/chat.db")
conn = sqlite3.connect(db_path)
handleCursor = conn.cursor()

handleCursor.execute("SELECT ROWID, id, service FROM handle;")

handles: list[Handle] = []
for row in handleCursor.fetchall():
    handle = Handle(rowid=row[0], id=row[1], service=row[2])
    handles.append(handle)
    handleTables.add_row(str(row[0]), row[1], row[2])

# Display Handles
console.print(handleTables)
# Make user pick a id to see the messages

selected_id = get_id()

selected_handle = next((handle for handle in handles if handle.rowid == int(selected_id)), None)
if selected_handle is None:
    print(f"No handle found with ROWID {selected_id}")
    exit(0)

print(f"Selected handle: {selected_handle}")

# Load Contacts and find the matching contact for the selected handle.id
print("Loading contacts...")
contacts = load_contacts()
print(f"Loaded {len(contacts)} contacts:")

selected = selected_handle.id.lower()
selected_digits = normalize(selected_handle.id)

matching = [
    key
    for key in contacts.keys()
    if key.lower() == selected
    or normalize(key) == selected_digits
    or selected_digits.endswith(normalize(key))
    or normalize(key).endswith(selected_digits)
]

if not len(matching) > 0:
    print("No matching contacts found for this handle.")
    exit(0)


filtered_handles = [
    handle
    for handle in handles
    if any(normalize(handle.id).endswith(k) for k in matching)
]
pprint(filtered_handles)

messages_data: list[Message] = []
for handle in filtered_handles:
    temp_cursor = conn.cursor()

    temp_cursor.execute(
        """
        SELECT
            ROWID,
            text,
            is_from_me,
            date,
            service
        FROM message
        WHERE handle_id = ?
        ORDER BY date ASC;
        """,
        (handle.rowid,)
    )

    messages = temp_cursor.fetchall()
    for rowid, text, is_from_me, date, service in messages:
        text = text or "[Attachment/No Text]"

        messages_data.append(Message(rowid=rowid, text=text, is_from_me=bool(is_from_me), date=date, service=service))

# Sort messages by date ascending
messages_data.sort(key=lambda m: m.date)


# Begin Segmentation
segmenter = ConversationSegmenter(messages_data)
messages_by_segment: list[list[Message]] = segmenter.begin_conversation_segmentation()

system_prompt = get_system_prompt()
formatter = JsonLFormatter(messages_by_segment, system_prompt)
jsonl_output = clean_text(formatter.format_to_jsonl())

lines = jsonl_output.splitlines()

lines_count = len(lines)
print(f"Generated JSONL with {lines_count} lines.")

valid_jsonl_size = int(lines_count / 7.25)
test_jsonl_size = int(lines_count / 7.25)
train_jsonl_size = lines_count - valid_jsonl_size - test_jsonl_size

print(f"train.jsonl size: {train_jsonl_size}")
print(f"valid.jsonl size: {valid_jsonl_size}")
print(f"test.jsonl  size: {test_jsonl_size}")

train_lines = lines[:train_jsonl_size]
valid_lines = lines[train_jsonl_size:train_jsonl_size + valid_jsonl_size]
test_lines = lines[train_jsonl_size + valid_jsonl_size:]

date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_file = f"outputs/{date}/train.jsonl"
valid_file = f"outputs/{date}/valid.jsonl"
test_file = f"outputs/{date}/test.jsonl"

os.makedirs(os.path.dirname(train_file), exist_ok=True)
with open(train_file, "w", encoding="utf-8") as f:
    f.write("\n".join(train_lines))

os.makedirs(os.path.dirname(valid_file), exist_ok=True)
with open(valid_file, "w", encoding="utf-8") as f:
    f.write("\n".join(valid_lines))

os.makedirs(os.path.dirname(test_file), exist_ok=True)
with open(test_file, "w", encoding="utf-8") as f:
    f.write("\n".join(test_lines))
