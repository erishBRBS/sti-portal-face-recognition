import json
import os
from typing import Any

from app.core.config import settings


class StorageService:
    def __init__(self) -> None:
        self.file_path = settings.embeddings_file
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def load_embeddings(self) -> list[dict[str, Any]]:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                return []
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def save_embeddings(self, items: list[dict[str, Any]]) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def upsert_student_embedding(
        self,
        student_id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        items = self.load_embeddings()

        existing_index = next(
            (i for i, item in enumerate(items) if item.get("student_id") == student_id),
            None,
        )

        record = {
            "student_id": student_id,
            "embedding": embedding,
            "metadata": metadata or {},
        }

        if existing_index is None:
            items.append(record)
        else:
            items[existing_index] = record

        self.save_embeddings(items)