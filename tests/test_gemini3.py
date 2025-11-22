import base64
import json
from types import SimpleNamespace

from app.main import app
from app.routers.gemini3 import get_gemini3_client


class FakeUsage:
    def model_dump(self):
        return {"prompt_token_count": 10, "candidates_token_count": 20}


def _candidate(parts):
    return SimpleNamespace(content=SimpleNamespace(parts=parts))


class FakeResponse:
    def __init__(self, text: str, signatures):
        parts = [SimpleNamespace(text=text, thought_signature=s) for s in signatures]
        self.text = text
        self.candidates = [_candidate(parts)]
        self.usage_metadata = FakeUsage()

    def to_json_dict(self):
        return {"text": self.text}


class FakeGemini3Client:
    def __init__(self, text="ok", structured=None, signatures=None):
        self.text = text
        self.structured = structured or {"status": "ok"}
        self.signatures = signatures if signatures is not None else ["sig-default"]
        self.last_kwargs = {}

    def _response(self, text, signatures):
        return FakeResponse(text, signatures)

    def generate_text(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response(self.text, self.signatures)

    def generate_multimodal(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response(self.text, self.signatures)

    def generate_structured(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response(json.dumps(self.structured), self.signatures)


def _override_client(fake_client):
    app.dependency_overrides[get_gemini3_client] = lambda: fake_client


def _reset_client_override():
    app.dependency_overrides.pop(get_gemini3_client, None)


def test_text_only_generation(client):
    fake_client = FakeGemini3Client(text="hello world", signatures=["alpha-signature"])
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [
                {"role": "user", "parts": [{"text": "hi"}]},
            ]
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["text"] == "hello world"
    assert payload["thought_signatures"] == ["alpha-signature"]
    assert payload["usage"]["prompt_token_count"] == 10


def test_image_media_resolution_defaults_to_high(client):
    fake_client = FakeGemini3Client(text="image-response")
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [{"role": "user", "parts": [{"text": "see image"}]}],
            "media": [
                {
                    "mime_type": "image/png",
                    "base64_data": base64.b64encode(b"img-bytes").decode(),
                }
            ],
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    assert fake_client.last_kwargs["media_resolution"].name == "MEDIA_RESOLUTION_HIGH"


def test_text_field_backwards_compat(client):
    fake_client = FakeGemini3Client(text="fallback", signatures=["sig-text"])
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [
                {"role": "user", "text": "legacy format"},
            ]
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    assert resp.json()["text"] == "fallback"


def test_parts_string_is_accepted(client):
    fake_client = FakeGemini3Client(text="string-parts")
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [
                {"role": "user", "parts": "hello as string"},
            ]
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    assert resp.json()["text"] == "string-parts"


def test_media_with_conversation_persists(client):
    fake_client = FakeGemini3Client(text="media persisted")
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "conversation_id": "conv-media",
            "messages": [{"role": "user", "parts": [{"text": "see file"}]}],
            "media": [
                {
                    "mime_type": "image/png",
                    "base64_data": base64.b64encode(b"img-bytes").decode(),
                }
            ],
        },
    )
    _reset_client_override()

    assert resp.status_code == 200


def test_binary_thought_signature_is_encoded(client):
    fake_client = FakeGemini3Client(text="bin", signatures=[b"\x01\x02binary"])
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [{"role": "user", "parts": [{"text": "hi"}]}],
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    encoded = resp.json()["thought_signatures"][0]
    assert isinstance(encoded, str)
    assert encoded != ""


def test_pdf_media_resolution_defaults_to_medium(client):
    fake_client = FakeGemini3Client(text="pdf-response")
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "messages": [{"role": "user", "parts": [{"text": "pdf"}]}],
            "media": [
                {
                    "mime_type": "application/pdf",
                    "base64_data": base64.b64encode(b"pdf-bytes").decode(),
                }
            ],
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    assert fake_client.last_kwargs["media_resolution"].name == "MEDIA_RESOLUTION_MEDIUM"


def test_structured_output_parsing(client):
    structured_payload = {"answer": "42"}
    fake_client = FakeGemini3Client(structured=structured_payload)
    _override_client(fake_client)

    resp = client.post(
        "/api/v1/chat/gemini3/structured",
        json={
            "prompt": "answer in json",
            "jsonSchema": {"type": "object"},
            "tools": {"google_search": True, "url_context": False, "code_execution": False},
        },
    )
    _reset_client_override()

    assert resp.status_code == 200
    assert resp.json()["data"] == structured_payload


def test_thought_signature_round_trip(client):
    conversation_id = "conv-123"
    first_client = FakeGemini3Client(text="first", signatures=["sig-one"])
    _override_client(first_client)

    resp = client.post(
        "/api/v1/chat/gemini3",
        json={
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "parts": [{"text": "keep this"}]}],
        },
    )
    assert resp.status_code == 200
    _reset_client_override()

    second_client = FakeGemini3Client(text="second")
    _override_client(second_client)
    resp2 = client.post(
        "/api/v1/chat/gemini3",
        json={
            "conversation_id": conversation_id,
            "messages": [{"role": "user", "parts": [{"text": "again"}]}],
        },
    )
    _reset_client_override()

    assert resp2.status_code == 200
    injected = second_client.last_kwargs["messages"][0]
    assert injected["role"] == "system"
    assert injected["parts"][0]["thought_signature"] == "sig-one"

