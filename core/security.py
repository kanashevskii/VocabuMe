"""Security headers compatible with the Telegram Mini App runtime."""

from __future__ import annotations

from collections.abc import Callable

from django.http import HttpRequest, HttpResponse

CONTENT_SECURITY_POLICY = "; ".join(
    (
        "default-src 'self'",
        "base-uri 'self'",
        "object-src 'none'",
        "script-src 'self' 'unsafe-inline' https://telegram.org https://*.telegram.org https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net",
        "img-src 'self' data: blob: https:",
        "media-src 'self' blob:",
        "connect-src 'self' https://api.telegram.org",
        "frame-src https://web.telegram.org https://*.telegram.org",
        "frame-ancestors 'self' https://web.telegram.org https://*.telegram.org",
    )
)
PERMISSIONS_POLICY = "camera=(self), microphone=(self), geolocation=()"


class SecurityHeadersMiddleware:
    """Attach a conservative CSP and browser permission policy to every response."""

    def __init__(self, get_response: Callable[[HttpRequest], HttpResponse]) -> None:
        self.get_response = get_response

    def __call__(self, request: HttpRequest) -> HttpResponse:
        response = self.get_response(request)
        response.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        response.setdefault("Permissions-Policy", PERMISSIONS_POLICY)
        return response
