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
        student_no: str,
        full_name: str,
        course: str,
        section: str,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/process-scan/gate-monitoring"

        payload = {
            "student_no": student_no,
            "full_name": full_name,
            "course": course,
            "section": section,
        }

        try:
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
                "success": response.is_success,
                "status_code": response.status_code,
                "data": data,
                "request_payload": payload,
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "status_code": 504,
                "data": {
                    "message": "Request to Laravel timed out."
                },
                "request_payload": payload,
            }

        except httpx.RequestError as e:
            return {
                "success": False,
                "status_code": 500,
                "data": {
                    "message": "Failed to connect to Laravel API.",
                    "error": str(e),
                },
                "request_payload": payload,
            }


laravel_service = LaravelService()