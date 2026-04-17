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
    student_no: str = Form(...),
    images: list[UploadFile] = File(...),
) -> EnrollResponse:
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images uploaded.")

    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Maximum of 5 images only.")

    result = await insightface_service.enroll_faces(
        student_no=student_no,
        files=images,
    )

    return EnrollResponse(
        success=True,
        message="Face(s) enrolled successfully.",
        student_no=result["student_no"],
        metadata=result["metadata"],
    )


@router.post("/recognize-face", response_model=RecognizeResponse)
async def recognize_face(
    image: UploadFile = File(...),
) -> RecognizeResponse:
    result = await insightface_service.recognize_face(file=image)

    if not result.matched or not result.student_no or result.similarity is None:
        return RecognizeResponse(
            success=True,
            matched=False,
            student_no=None,
            similarity=result.similarity,
            message="No matching student found.",
            laravel_response=None,
            metadata=None,
        )

    metadata = result.metadata or {}

    laravel_response = None
    if settings.notify_laravel:
        laravel_response = await laravel_service.notify_attendance_by_student_no(
            student_no=result.student_no,
        )

    return RecognizeResponse(
        success=True,
        matched=True,
        student_no=result.student_no,
        similarity=result.similarity,
        message="Student matched successfully.",
        laravel_response=laravel_response,
        metadata=metadata,
    )


@router.delete("/delete-face/{student_no}")
async def delete_face(student_no: str) -> dict[str, str | bool]:
    deleted = insightface_service.storage.delete_student_embedding(student_no)

    if not deleted:
        raise HTTPException(status_code=404, detail="Student face record not found.")

    return {
        "success": True,
        "message": "Student face record deleted successfully.",
        "student_no": student_no,
    }
