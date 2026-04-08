from typing import Any

from fastapi import APIRouter, File, Form, UploadFile, HTTPException

from app.core.config import settings
from app.schemas.recognition import EnrollResponse, HealthResponse, RecognizeResponse
from app.services.insightface_service import insightface_service
from app.services.laravel_service import laravel_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        success=True,
        message="Face recognition API is running.",
    )


@router.post("/enroll-face", response_model=EnrollResponse)
async def enroll_face(
    student_id: str = Form(...),
    full_name: str | None = Form(None),
    course: str | None = Form(None),
    section: str | None = Form(None),
    image: UploadFile = File(...),
) -> EnrollResponse:
    metadata: dict[str, Any] = {
        "full_name": full_name,
        "course": course,
        "section": section,
        "filename": image.filename,
    }

    result = await insightface_service.enroll_face(
        student_id=student_id,
        file=image,
        metadata=metadata,
    )

    return EnrollResponse(
        success=True,
        message="Face enrolled successfully.",
        student_id=result["student_id"],
        metadata=result["metadata"],
    )


@router.post("/recognize-face", response_model=RecognizeResponse)
async def recognize_face(
    image: UploadFile = File(...),
    camera_id: str | None = Form(None),
    gate_id: str | None = Form(None),
) -> RecognizeResponse:
    result = await insightface_service.recognize_face(file=image)

    if not result.matched or not result.student_id or result.similarity is None:
        return RecognizeResponse(
            success=True,
            matched=False,
            student_id=None,
            similarity=result.similarity,
            message="No matching student found.",
            laravel_response=None,
            metadata=None,
        )

    laravel_response = None
    if settings.notify_laravel:
        laravel_response = await laravel_service.notify_attendance(
            student_id=result.student_id,
            similarity=result.similarity,
            camera_id=camera_id,
            gate_id=gate_id,
            raw_result={
                "student_id": result.student_id,
                "similarity": result.similarity,
                "metadata": result.metadata or {},
            },
        )

    return RecognizeResponse(
        success=True,
        matched=True,
        student_id=result.student_id,
        similarity=result.similarity,
        message="Student matched successfully.",
        laravel_response=laravel_response,
        metadata=result.metadata,
    )

@router.delete("/delete-face/{student_id}")
async def delete_face(student_id: str) -> dict[str, str | bool]:
    deleted = insightface_service.storage.delete_student_embedding(student_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Student face record not found.")

    return {
        "success": True,
        "message": "Student face record deleted successfully.",
        "student_id": student_id,
    }