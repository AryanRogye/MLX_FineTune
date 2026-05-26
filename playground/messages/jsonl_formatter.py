import json
from models import Message

"""
Example:
{
  "messages": [
    {
      "role": "system",
      "content": "You label text message conversations. Return only valid JSON."
    },
    {
      "role": "user",
      "content": "[65793] hey\n[65794] wyd\n[65795] are you free later?"
    },
    {
      "role": "assistant",
      "content": "{\"topic\":\"making plans\",\"summary\":\"They discussed whether they were free later.\"}"
    }
  ]
}
"""
class JsonLFormatter:
    """
    Formatter for converting conversations into JSONL format suitable for fine-tuning language models.
    """
    def __init__(self, conversations: list[list[Message]], system_prompt: str):
        self.system_prompt = system_prompt
        self.conversations = conversations

    """
    Converts the segmented conversations into JSONL format.
    Each conversation is represented as a JSON object with a "conversation" key containing a list of messages.
    """
    def format_to_jsonl(self) -> str:
        rows: list[str] = []

        for conversation in self.conversations:
            messages = self._conversation_to_chat_messages(conversation)

            # Skip weak examples
            if len(messages) < 2:
                continue

            # Only keep examples where your side appears
            if not any(m["role"] == "assistant" for m in messages):
                continue

            row = {"messages": messages}
            rows.append(json.dumps(row, ensure_ascii=False))

        return "\n".join(rows)

    def _conversation_to_chat_messages(self, conversation: list[Message]) -> list[dict[str, str]]:
        chat_messages: list[dict[str, str]] = []

        for msg in conversation:
            if not msg.text or msg.text == "[Attachment/No Text]":
                continue

            text = msg.text.strip()
            if not text:
                continue

            role = "assistant" if msg.is_from_me else "user"

            # Merge consecutive messages from same speaker
            if chat_messages and chat_messages[-1]["role"] == role:
                chat_messages[-1]["content"] += "\n" + text
            else:
                text = text.replace("\n", "\\n")
                chat_messages.append({
                    "role": role,
                    "content": text
                })

        while chat_messages and chat_messages[0]["role"] == "assistant":
            chat_messages.pop(0)

        if not any(m["role"] == "user" for m in chat_messages):
            return []

        if not any(m["role"] == "assistant" for m in chat_messages):
            return []

        if chat_messages[-1]["role"] != "assistant":
            return []

        return [
            {
                "role": "system",
                "content": self.system_prompt
            },
            *chat_messages
        ]
