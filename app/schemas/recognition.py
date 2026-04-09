from pydantic import BaseModel
from typing import Any


class HealthResponse(BaseModel):
    success: bool
    message: str


class EnrollResponse(BaseModel):
    success: bool
    message: str
    student_no: str
    metadata: dict[str, Any] | None = None


class RecognizeResponse(BaseModel):
    success: bool
    matched: bool
    student_no: str | None = None
    similarity: float | None = None
    message: str
    laravel_response: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None