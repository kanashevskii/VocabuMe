"""OpenAPI schema and Swagger UI delivery endpoints."""

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_GET

from vocab.openapi import OPENAPI_SCHEMA


@require_GET
def openapi_schema(_: HttpRequest) -> JsonResponse:
    return JsonResponse(OPENAPI_SCHEMA)


@require_GET
def api_docs(_: HttpRequest) -> HttpResponse:
    return HttpResponse(
        """<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">\
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\
<title>VocabuMe API</title>\
<link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css\">\
</head><body><div id=\"swagger-ui\"></div>\
<script src=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js\"></script>\
<script>SwaggerUIBundle({url:'/api/openapi.json',dom_id:'#swagger-ui',persistAuthorization:true});</script>\
</body></html>""",
        content_type="text/html; charset=utf-8",
    )
