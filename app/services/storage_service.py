import json
import os
from typing import Any

from app.core.config import settings


class StorageService:
    def delete_student_embedding(self, student_no: str) -> bool:
      items = self.load_embeddings()

      filtered_items = [
          item for item in items
          if str(item.get("student_no")) != str(student_no)
      ]

      deleted = len(filtered_items) != len(items)

      if deleted:
          self.save_embeddings(filtered_items)

      return deleted

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

    def upsert_student_embeddings(
        self,
        student_no: str,
        embeddings: list[dict[str, Any]],
    ) -> None:
        items = self.load_embeddings()

        existing_index = next(
            (i for i, item in enumerate(items) if item.get("student_no") == student_no),
            None,
        )

        record = {
            "student_no": student_no,
            "embeddings": embeddings,
        }

        if existing_index is None:
            items.append(record)
        else:
            items[existing_index] = record

        self.save_embeddings(items)