from typing import Any

import httpx

from app.core.config import settings


class LaravelService:
    def __init__(self) -> None:
        self.base_url = settings.laravel_api_base_url.rstrip("/")
        self.timeout = settings.request_timeout_seconds

    def build_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
        }

        if settings.laravel_api_token:
            headers["Authorization"] = f"Bearer {settings.laravel_api_token}"

        return headers

    async def notify_attendance(
        self,
        student_id: str,
        similarity: float,
        camera_id: str | None = None,
        gate_id: str | None = None,
        raw_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/attendance/gate-face-scan"

        payload = {
            "student_id": student_id,
            "similarity": similarity,
            "camera_id": camera_id,
            "gate_id": gate_id,
            "provider": "insightface",
            "raw_result": raw_result or {},
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                json=payload,
                headers=self.build_headers(),
            )

            try:
                data = response.json()
            except Exception:
                data = {
                    "status_code": response.status_code,
                    "text": response.text,
                }

            return {
                "status_code": response.status_code,
                "data": data,
            }


laravel_service = LaravelService()