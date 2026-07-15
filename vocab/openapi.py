"""Hand-maintained OpenAPI contract for the public VocabuMe HTTP API.

The backend intentionally uses Django function views instead of DRF/Ninja. Keeping
the contract here avoids adding a second web framework only to generate a schema.
"""

from typing import Any

OPENAPI_SCHEMA: dict[str, Any] = {
    "openapi": "3.1.0",
    "info": {
        "title": "VocabuMe API",
        "version": "1.0.0",
        "description": (
            "API for the VocabuMe Telegram bot, Mini App, and website. "
            "Mutating endpoints require a Telegram-authenticated identity."
        ),
    },
    "tags": [
        {"name": "Auth"},
        {"name": "Learning"},
        {"name": "Dictionary"},
        {"name": "Audio"},
        {"name": "Packs"},
        {"name": "Billing"},
    ],
    "components": {
        "securitySchemes": {
            "TelegramInitData": {
                "type": "apiKey",
                "in": "header",
                "name": "X-Telegram-Init-Data",
                "description": "Signed Telegram Web App initData. Takes precedence over a session cookie.",
            },
            "SessionCookie": {
                "type": "apiKey",
                "in": "cookie",
                "name": "sessionid",
            },
        },
        "schemas": {
            "Error": {
                "type": "object",
                "required": ["ok", "error"],
                "properties": {
                    "ok": {"type": "boolean", "const": False},
                    "error": {"type": "string"},
                    "code": {"type": "string"},
                },
            },
            "AudioPreparation": {
                "type": "object",
                "required": ["ok", "ready", "job_id"],
                "properties": {
                    "ok": {"type": "boolean"},
                    "ready": {"type": "boolean"},
                    "job_id": {"type": ["integer", "null"]},
                },
            },
        },
        "responses": {
            "ErrorResponse": {
                "description": "A request validation, authentication, or rate-limit error.",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/Error"}
                    }
                },
            },
            "AudioPreparationResponse": {
                "description": "The existing or newly queued audio-preparation job.",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/AudioPreparation"}
                    }
                },
            },
        },
    },
    "paths": {
        "/api/app-config": {
            "get": {
                "tags": ["Auth"],
                "summary": "Read public Mini App configuration",
                "responses": {"200": {"description": "Public configuration"}},
            }
        },
        "/api/auth/me": {
            "get": {
                "tags": ["Auth"],
                "summary": "Read current authenticated Telegram identity and progress",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {"200": {"description": "Authentication state"}},
            }
        },
        "/api/auth/telegram/webapp": {
            "post": {
                "tags": ["Auth"],
                "summary": "Authenticate a Telegram Mini App initData payload",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["init_data"],
                                "properties": {"init_data": {"type": "string"}},
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "Authenticated Telegram user"}},
            }
        },
        "/api/dashboard": {
            "get": {
                "tags": ["Learning"],
                "summary": "Read dashboard and current progress",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {"200": {"description": "Dashboard payload"}},
            }
        },
        "/api/words": {
            "get": {
                "tags": ["Dictionary"],
                "summary": "List the current user's words",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {"200": {"description": "Dictionary items"}},
            },
            "post": {
                "tags": ["Dictionary"],
                "summary": "Create word drafts or a word",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {
                    "200": {"description": "Created item or draft flow"},
                    "402": {"$ref": "#/components/responses/ErrorResponse"},
                    "429": {"$ref": "#/components/responses/ErrorResponse"},
                },
            },
        },
        "/api/learn/question": {
            "get": {
                "tags": ["Learning"],
                "summary": "Issue a server-authoritative learning question",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {
                    "200": {"description": "Question with opaque question_id"}
                },
            }
        },
        "/api/learn/answer": {
            "post": {
                "tags": ["Learning"],
                "summary": "Submit an answer to an issued question",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {
                    "200": {"description": "Idempotent answer result"},
                    "400": {"$ref": "#/components/responses/ErrorResponse"},
                    "429": {"$ref": "#/components/responses/ErrorResponse"},
                },
            }
        },
        "/api/audio/{word_id}": {
            "get": {
                "tags": ["Audio"],
                "summary": "Read an already prepared word audio asset",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "parameters": [{"$ref": "#/components/parameters/WordId"}],
                "responses": {
                    "200": {"description": "MPEG audio"},
                    "404": {"$ref": "#/components/responses/ErrorResponse"},
                },
            }
        },
        "/api/audio/{word_id}/prepare": {
            "post": {
                "tags": ["Audio"],
                "summary": "Queue idempotent audio preparation",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "parameters": [{"$ref": "#/components/parameters/WordId"}],
                "responses": {
                    "200": {"$ref": "#/components/responses/AudioPreparationResponse"},
                    "202": {"$ref": "#/components/responses/AudioPreparationResponse"},
                    "429": {"$ref": "#/components/responses/ErrorResponse"},
                },
            }
        },
        "/api/packs": {
            "get": {
                "tags": ["Packs"],
                "summary": "List packs for the active course",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {"200": {"description": "Pack catalog and state"}},
            }
        },
        "/api/packs/prepare": {
            "post": {
                "tags": ["Packs"],
                "summary": "Queue pack preparation",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {
                    "202": {"description": "Preparation job accepted"},
                    "429": {"$ref": "#/components/responses/ErrorResponse"},
                },
            }
        },
        "/api/billing": {
            "get": {
                "tags": ["Billing"],
                "summary": "Read subscription and entitlement state",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {"200": {"description": "Billing state"}},
            }
        },
        "/api/billing/checkout": {
            "post": {
                "tags": ["Billing"],
                "summary": "Create a Telegram checkout attempt",
                "security": [{"TelegramInitData": []}, {"SessionCookie": []}],
                "responses": {
                    "200": {"description": "Checkout link or invoice"},
                    "400": {"$ref": "#/components/responses/ErrorResponse"},
                    "429": {"$ref": "#/components/responses/ErrorResponse"},
                },
            }
        },
    },
}

OPENAPI_SCHEMA["components"]["parameters"] = {
    "WordId": {
        "in": "path",
        "name": "word_id",
        "required": True,
        "schema": {"type": "integer", "minimum": 1},
    }
}
