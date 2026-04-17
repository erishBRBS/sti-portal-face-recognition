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
            "Content-Type": "application/json",
        }

        if settings.laravel_api_token:
            headers["Authorization"] = f"Bearer {settings.laravel_api_token}"

        return headers

    async def get_student_by_student_no(self, student_no: str) -> dict[str, Any]:
        """
        Fetch student details from Laravel using student_no.
        Important: student_no must stay as string to preserve leading zeros.
        """
        url = f"{self.base_url}/get/student/search/by-student-no"
        params = {
            "student_no": student_no,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    url,
                    params=params,
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
                "request_params": params,
            }

        except httpx.TimeoutException:
            return {
                "success": False,
                "status_code": 504,
                "data": {
                    "message": "Request to Laravel timed out while fetching student."
                },
                "request_params": params,
            }

        except httpx.RequestError as e:
            return {
                "success": False,
                "status_code": 500,
                "data": {
                    "message": "Failed to connect to Laravel API while fetching student.",
                    "error": str(e),
                },
                "request_params": params,
            }

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

    async def notify_attendance_by_student_no(self, student_no: str) -> dict[str, Any]:
        """
        1. Fetch student by student_no
        2. Build full_name, course, section
        3. Send gate monitoring scan
        """
        student_response = await self.get_student_by_student_no(student_no)

        if not student_response["success"]:
            return {
                "success": False,
                "status_code": student_response["status_code"],
                "data": {
                    "message": "Failed to fetch student before attendance notification.",
                    "student_lookup": student_response["data"],
                },
                "request_payload": {
                    "student_no": student_no,
                },
            }

        student_wrapper = student_response.get("data", {})
        student_data = student_wrapper.get("data")

        if not student_data:
            return {
                "success": False,
                "status_code": 404,
                "data": {
                    "message": "Student not found.",
                    "student_lookup": student_wrapper,
                },
                "request_payload": {
                    "student_no": student_no,
                },
            }

        first_name = student_data.get("first_name") or ""
        middle_name = student_data.get("middle_name") or ""
        last_name = student_data.get("last_name") or ""

        full_name = " ".join(
            part.strip()
            for part in [first_name, middle_name, last_name]
            if part and part.strip()
        ).strip()

        course = (
            student_data.get("course", {}).get("course_name")
            if isinstance(student_data.get("course"), dict)
            else ""
        ) or ""

        section = (
            student_data.get("section", {}).get("section_name")
            if isinstance(student_data.get("section"), dict)
            else ""
        ) or ""

        return await self.notify_attendance(
            student_no=student_no,
            full_name=full_name,
            course=course,
            section=section,
        )


laravel_service = LaravelService()