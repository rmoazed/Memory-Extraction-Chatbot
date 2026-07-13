from typing import Literal
from pydantic import BaseModel


class MemoryExtractionResult(BaseModel):
    importance: Literal[1, 2, 3, 4, 5]
    category: Literal[
        "critical_constraint",
        "stable_preference_goal",
        "useful_recurring_context",
        "temporary_context",
        "do_not_store"
    ]
    memory_text: str


class MemoryRelationshipResult(BaseModel):
    relationship: Literal[
        "unrelated",
        "duplicate",
        "compatible",
        "conflict"
    ]
    existing_memory_id: str | None
    existing_memory_text: str
    memory_to_store: str
    action: Literal[
        "store",
        "skip",
        "replace"
    ]