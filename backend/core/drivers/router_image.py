"""
Router Image Driver — imágenes y edición a través del gateway Router de Olares.

Habla la API OpenAI-compatible que Router expone en su entrada pública, con una
clave emitida por `olares-cli router key issue`. Funciona tanto con modelos en la
nube (openai/chatgpt-image-latest) como con filas locales servidas por FlowStudio
(Olares/<uuid>), porque ambos se direccionan igual: <provider>/<model>.

Configuración, por orden de prioridad:
    1. variables de entorno ROUTER_BASE_URL / ROUTER_API_KEY
    2. assets/settings.json (pantalla de ajustes del estudio)
"""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import aiohttp

from .base import (
    DriverCategory,
    GenerationStatus,
    ImageDriver,
    ImageGenerationRequest,
    ImageGenerationResponse,
)

DEFAULT_BASE_URL = "https://router.ataillepierre.olares.com/v1"

# Tamaños que acepta la API de imágenes de OpenAI. Router no admite otros.
_SIZE_LANDSCAPE = "1536x1024"
_SIZE_PORTRAIT = "1024x1536"
_SIZE_SQUARE = "1024x1024"


class RouterImageDriver(ImageDriver):
    """Genera y edita imágenes llamando a Router."""

    MODEL_INFO = {
        "gpt_image": {
            "name": "GPT Image (Router)",
            "model": "openai/chatgpt-image-latest",
        },
        "router_image": {
            "name": "GPT Image (Router)",
            "model": "openai/chatgpt-image-latest",
        },
        "qwen_image_router": {
            "name": "Qwen-Image 2.1 Edit (Router)",
            "model": "Olares/67740d83-45f9-43e5-bd8b-e2b40d78443e",
        },
        "router_vision": {
            "name": "GPT-4.1 visión (Router)",
            "model": "openai/gpt-4.1",
        },
    }

    def __init__(self, model_id: str = "gpt_image"):
        info = self.MODEL_INFO.get(model_id, self.MODEL_INFO["gpt_image"])
        self._model_id = model_id
        self._model_name = info["name"]
        self._model = info["model"]
        self._jobs: Dict[str, dict] = {}
        self._vault = Path(__file__).parent.parent.parent / "assets"

    # --- identidad -----------------------------------------------------------

    @property
    def driver_id(self) -> str:
        return self._model_id

    @property
    def driver_name(self) -> str:
        return self._model_name

    @property
    def category(self) -> DriverCategory:
        return DriverCategory.CLOUD

    @property
    def supported_features(self) -> List[str]:
        return ["text_to_image", "image_to_image", "multi_reference"]

    # --- configuración -------------------------------------------------------

    @staticmethod
    def _settings() -> dict:
        path = Path(__file__).parent.parent.parent / "assets" / "settings.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8-sig"))
            except Exception:
                return {}
        return {}

    def _api_base(self) -> str:
        base = os.getenv("ROUTER_BASE_URL") or self._settings().get("ROUTER_BASE_URL") or DEFAULT_BASE_URL
        return base.rstrip("/")

    def _api_key(self) -> Optional[str]:
        return os.getenv("ROUTER_API_KEY") or self._settings().get("ROUTER_API_KEY")

    def _headers(self) -> dict:
        key = self._api_key()
        if not key:
            raise RuntimeError("ROUTER_API_KEY no configurada (ajustes del estudio o variable de entorno)")
        return {"Authorization": f"Bearer {key}"}

    # --- utilidades ----------------------------------------------------------

    def _resolve_local(self, url: str) -> Optional[Path]:
        if url.startswith("/assets/"):
            candidate = self._vault / url[len("/assets/"):]
            return candidate if candidate.exists() else None
        p = Path(url)
        return p if p.exists() else None

    @staticmethod
    def _size(width: Optional[int], height: Optional[int]) -> str:
        w = width or 1024
        h = height or 1024
        if not h:
            return _SIZE_SQUARE
        ratio = w / h
        if ratio > 1.2:
            return _SIZE_LANDSCAPE
        if ratio < 0.8:
            return _SIZE_PORTRAIT
        return _SIZE_SQUARE

    def _store(self, job_id: str, index: int, data: bytes) -> str:
        out_dir = self._vault / "generated"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{job_id}_{index}.png"
        (out_dir / name).write_bytes(data)
        return f"/assets/generated/{name}"

    async def _decode(self, session: aiohttp.ClientSession, item: dict, job_id: str, index: int) -> Optional[str]:
        if item.get("b64_json"):
            return self._store(job_id, index, base64.b64decode(item["b64_json"]))
        if item.get("url"):
            async with session.get(item["url"]) as resp:
                if resp.status == 200:
                    return self._store(job_id, index, await resp.read())
        return None

    # --- generación ----------------------------------------------------------

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {"status": GenerationStatus.PENDING, "images": [], "error": None}
        print(f"[RouterImageDriver] generate: model={self._model}, prompt={request.prompt[:80]}..., refs={len(request.reference_image_paths)}")
        asyncio.create_task(self._run(job_id, request))
        return ImageGenerationResponse(job_id=job_id, status=GenerationStatus.PENDING)

    async def _run(self, job_id: str, request: ImageGenerationRequest) -> None:
        job = self._jobs[job_id]
        try:
            job["status"] = GenerationStatus.PROCESSING
            size = self._size(request.width, request.height)
            timeout = aiohttp.ClientTimeout(total=420)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                if request.reference_image_paths:
                    # Edición: multipart con la imagen de referencia
                    form = aiohttp.FormData()
                    form.add_field("model", self._model)
                    form.add_field("prompt", request.prompt)
                    form.add_field("size", size)
                    form.add_field("n", str(max(1, min(4, request.num_images))))
                    attached = 0
                    for ref in request.reference_image_paths[:4]:
                        local = self._resolve_local(ref)
                        if not local:
                            print(f"[RouterImageDriver]   no encuentro {ref}")
                            continue
                        mime = mimetypes.guess_type(str(local))[0] or "image/png"
                        field = "image[]" if attached else "image"
                        form.add_field(
                            field,
                            local.read_bytes(),
                            filename=local.name,
                            content_type=mime,
                        )
                        attached += 1
                    if not attached:
                        raise RuntimeError("ninguna imagen de referencia se pudo leer")
                    url = f"{self._api_base()}/images/edits"
                    async with session.post(url, headers=self._headers(), data=form) as resp:
                        payload = await resp.json(content_type=None)
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: {str(payload)[:300]}")
                else:
                    body = {
                        "model": self._model,
                        "prompt": request.prompt,
                        "size": size,
                        "n": max(1, min(4, request.num_images)),
                    }
                    url = f"{self._api_base()}/images/generations"
                    async with session.post(url, headers=self._headers(), json=body) as resp:
                        payload = await resp.json(content_type=None)
                        if resp.status >= 400:
                            raise RuntimeError(f"HTTP {resp.status}: {str(payload)[:300]}")

                images: List[str] = []
                for i, item in enumerate(payload.get("data") or []):
                    saved = await self._decode(session, item, job_id, i)
                    if saved:
                        images.append(saved)

            if not images:
                raise RuntimeError(f"respuesta sin imágenes: {str(payload)[:200]}")

            job["images"] = images
            job["status"] = GenerationStatus.COMPLETED
            print(f"[RouterImageDriver] completado: {images}")
        except Exception as e:  # noqa: BLE001 — el mensaje se propaga al estudio
            job["status"] = GenerationStatus.FAILED
            job["error"] = str(e)
            print(f"[RouterImageDriver] FALLÓ: {e}")

    async def check_status(self, job_id: str) -> ImageGenerationResponse:
        job = self._jobs.get(job_id)
        if not job:
            return ImageGenerationResponse(
                job_id=job_id,
                status=GenerationStatus.FAILED,
                error_message="trabajo desconocido",
            )
        return ImageGenerationResponse(
            job_id=job_id,
            status=job["status"],
            image_paths=job["images"],
            image_urls=job["images"],
            error_message=job["error"],
            metadata={"model": self._model, "provider": "router"},
        )

    # --- visión: análisis de personaje ---------------------------------------

    async def analyze_character(self, image_path: str) -> ImageGenerationResponse:
        """Describe un personaje a partir de su imagen usando un modelo con visión."""
        job_id = str(uuid.uuid4())
        local = self._resolve_local(image_path)
        if not local:
            return ImageGenerationResponse(
                job_id=job_id,
                status=GenerationStatus.FAILED,
                error_message=f"no encuentro la imagen {image_path}",
            )
        mime = mimetypes.guess_type(str(local))[0] or "image/png"
        b64 = base64.b64encode(local.read_bytes()).decode()
        body = {
            "model": "openai/gpt-4.1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this character in detail for a turnaround sheet. Focus on: "
                                "gender, approximate age, clothing (color, style, material), hair (color, "
                                "length, style), body type, distinguishing features (scars, tattoos, "
                                "accessories), footwear, and any props. Write as a single concise "
                                "paragraph, no bullet points."
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }
            ],
        }
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self._api_base()}/chat/completions", headers=self._headers(), json=body
                ) as resp:
                    payload = await resp.json(content_type=None)
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}: {str(payload)[:300]}")
            text = payload["choices"][0]["message"]["content"].strip()
            return ImageGenerationResponse(
                job_id=job_id,
                status=GenerationStatus.COMPLETED,
                metadata={"description": text, "model": "openai/gpt-4.1"},
            )
        except Exception as e:  # noqa: BLE001
            print(f"[RouterImageDriver] visión FALLÓ: {e}")
            return ImageGenerationResponse(
                job_id=job_id,
                status=GenerationStatus.FAILED,
                error_message=str(e),
            )

    async def check_analysis_status(self, job_id: str) -> ImageGenerationResponse:
        return await self.check_status(job_id)

    # --- hoja de personaje ---------------------------------------------------

    async def generate_asset_sheet(
        self,
        asset_image_path: str,
        asset_type: str,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> ImageGenerationResponse:
        """Hoja de diseño de varias vistas, editando la imagen del personaje."""
        sheet_prompt = (
            f"Character design sheet for a {asset_type or 'character'}. Show the same character "
            "in four consistent views: front, three-quarter, profile and back, on a plain light "
            "neutral background, same lighting and same clothing in every view, full body, "
            "professional animation model sheet layout. "
            f"{prompt}".strip()
        )
        request = ImageGenerationRequest(
            prompt=sheet_prompt,
            width=1536,
            height=1024,
            num_images=1,
            reference_image_paths=[asset_image_path],
        )
        return await self.generate(request)
