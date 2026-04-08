from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile
from insightface.app import FaceAnalysis

from app.core.config import settings
from app.services.storage_service import StorageService


@dataclass
class FaceMatchResult:
    matched: bool
    student_id: str | None
    similarity: float | None
    metadata: dict[str, Any] | None


class InsightFaceService:
    def __init__(self) -> None:
        self.storage = StorageService()

        # CPU-only provider
        self.providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=settings.face_model_name,
            providers=self.providers,
        )

        # ctx_id = -1 for CPU
        self.app.prepare(
            ctx_id=-1,
            det_size=(settings.face_det_width, settings.face_det_height),
        )

    async def read_image(self, file: UploadFile) -> np.ndarray:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file uploaded.")

        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        return image

    def get_largest_face(self, image: np.ndarray) -> Any:
        faces = self.app.get(image)

        if not faces:
            raise HTTPException(status_code=404, detail="No face detected.")

        largest_face = max(
            faces,
            key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]),
        )
        return largest_face

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        emb1 = self.normalize_embedding(emb1)
        emb2 = self.normalize_embedding(emb2)
        return float(np.dot(emb1, emb2))

    async def enroll_face(
        self,
        student_id: str,
        file: UploadFile,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        image = await self.read_image(file)
        face = self.get_largest_face(image)

        embedding = face.embedding.astype(np.float32).tolist()

        self.storage.upsert_student_embedding(
            student_id=student_id,
            embedding=embedding,
            metadata=metadata,
        )

        return {
            "student_id": student_id,
            "metadata": metadata or {},
        }

    async def recognize_face(self, file: UploadFile) -> FaceMatchResult:
        image = await self.read_image(file)
        face = self.get_largest_face(image)
        probe_embedding = np.array(face.embedding, dtype=np.float32)

        candidates = self.storage.load_embeddings()
        if not candidates:
            return FaceMatchResult(
                matched=False,
                student_id=None,
                similarity=None,
                metadata=None,
            )

        best_student_id: str | None = None
        best_similarity: float = -1.0
        best_metadata: dict[str, Any] | None = None

        for candidate in candidates:
            stored_embedding = np.array(candidate["embedding"], dtype=np.float32)
            similarity = self.cosine_similarity(probe_embedding, stored_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_student_id = candidate.get("student_id")
                best_metadata = candidate.get("metadata", {})

        matched = best_similarity >= settings.face_similarity_threshold

        return FaceMatchResult(
            matched=matched,
            student_id=best_student_id if matched else None,
            similarity=best_similarity if best_similarity >= 0 else None,
            metadata=best_metadata if matched else None,
        )


insightface_service = InsightFaceService()