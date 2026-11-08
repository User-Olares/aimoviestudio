"""
Fallback Image Driver — primario en la nube, respaldo local.

Envuelve dos drivers de imagen con la misma interfaz. Intenta el primario y, si
falla al enviar o termina en FAILED, repite el trabajo con el secundario. El
resultado indica siempre qué driver atendió, para que la interfaz pueda avisar.

Se usa como "GPT Image con respaldo Qwen": si Router no responde, está sin
saldo o caducó la clave, el estudio sigue produciendo en local sin que el
usuario tenga que cambiar nada.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .base import (
    DriverCategory,
    GenerationStatus,
    ImageDriver,
    ImageGenerationRequest,
    ImageGenerationResponse,
)


class FallbackImageDriver(ImageDriver):
    """Encadena un driver primario con un respaldo ante cualquier fallo."""

    def __init__(
        self,
        primary: ImageDriver,
        secondary: ImageDriver,
        model_id: str = "image_auto",
        name: Optional[str] = None,
    ):
        self._primary = primary
        self._secondary = secondary
        self._model_id = model_id
        self._name = name or f"{primary.driver_name} → {secondary.driver_name} (respaldo)"
        # job_id -> driver que lo atiende
        self._route: Dict[str, ImageDriver] = {}
        self._used: Dict[str, str] = {}
        self._last_request: Optional[ImageGenerationRequest] = None

    # --- identidad -----------------------------------------------------------

    @property
    def driver_id(self) -> str:
        return self._model_id

    @property
    def driver_name(self) -> str:
        return self._name

    @property
    def category(self) -> DriverCategory:
        return self._primary.category

    @property
    def supported_features(self) -> List[str]:
        merged = list(dict.fromkeys(self._primary.supported_features + self._secondary.supported_features))
        return merged

    # --- generación ----------------------------------------------------------

    async def _dispatch(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """Intenta el primario; si falla de inmediato, usa el secundario."""
        try:
            response = await self._primary.generate(request)
        except Exception as e:  # noqa: BLE001
            print(f"[FallbackImageDriver] primario lanzó excepción: {e}; paso al respaldo")
            response = ImageGenerationResponse(
                job_id="", status=GenerationStatus.FAILED, error_message=str(e)
            )

        if response.status != GenerationStatus.FAILED:
            self._route[response.job_id] = self._primary
            self._used[response.job_id] = self._primary.driver_name
            return response

        print(f"[FallbackImageDriver] primario falló al enviar ({response.error_message}); usando respaldo")
        backup = await self._secondary.generate(request)
        self._route[backup.job_id] = self._secondary
        self._used[backup.job_id] = self._secondary.driver_name
        meta = dict(backup.metadata or {})
        meta["fallback"] = True
        meta["fallback_reason"] = response.error_message
        backup.metadata = meta
        return backup

    async def generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        self._last_request = request
        return await self._dispatch(request)

    async def check_status(self, job_id: str) -> ImageGenerationResponse:
        driver = self._route.get(job_id)
        if driver is None:
            return ImageGenerationResponse(
                job_id=job_id, status=GenerationStatus.FAILED, error_message="trabajo desconocido"
            )
        response = await driver.check_status(job_id)

        # El primario puede fallar ya en marcha: se reintenta una sola vez en el respaldo
        if (
            response.status == GenerationStatus.FAILED
            and driver is self._primary
            and self._last_request is not None
        ):
            print(f"[FallbackImageDriver] primario falló en curso ({response.error_message}); reintento en respaldo")
            self._route.pop(job_id, None)
            retry = await self._secondary.generate(self._last_request)
            self._route[retry.job_id] = self._secondary
            self._used[retry.job_id] = self._secondary.driver_name
            meta = dict(retry.metadata or {})
            meta["fallback"] = True
            meta["fallback_reason"] = response.error_message
            retry.metadata = meta
            return retry

        meta = dict(response.metadata or {})
        meta.setdefault("served_by", self._used.get(job_id))
        response.metadata = meta
        return response

    async def generate_asset_sheet(
        self,
        asset_image_path: str,
        asset_type: str,
        prompt: str,
        negative_prompt: str = "",
        seed: Optional[int] = None,
    ) -> ImageGenerationResponse:
        self._last_request = ImageGenerationRequest(
            prompt=prompt,
            width=1536,
            height=1024,
            num_images=1,
            reference_image_paths=[asset_image_path],
        )
        try:
            response = await self._primary.generate_asset_sheet(
                asset_image_path, asset_type, prompt, negative_prompt, seed
            )
        except Exception as e:  # noqa: BLE001
            print(f"[FallbackImageDriver] hoja: primario lanzó excepción: {e}")
            response = ImageGenerationResponse(job_id="", status=GenerationStatus.FAILED, error_message=str(e))

        if response.status != GenerationStatus.FAILED and response.job_id:
            self._route[response.job_id] = self._primary
            self._used[response.job_id] = self._primary.driver_name
            return response

        print("[FallbackImageDriver] hoja: usando respaldo")
        backup = await self._secondary.generate_asset_sheet(
            asset_image_path, asset_type, prompt, negative_prompt, seed
        )
        self._route[backup.job_id] = self._secondary
        self._used[backup.job_id] = self._secondary.driver_name
        meta = dict(backup.metadata or {})
        meta["fallback"] = True
        meta["fallback_reason"] = response.error_message
        backup.metadata = meta
        return backup

    async def analyze_character(self, image_path: str) -> ImageGenerationResponse:
        if hasattr(self._primary, "analyze_character"):
            response = await self._primary.analyze_character(image_path)
            if response.status != GenerationStatus.FAILED:
                return response
            print(f"[FallbackImageDriver] análisis: primario falló ({response.error_message})")
        if hasattr(self._secondary, "analyze_character"):
            return await self._secondary.analyze_character(image_path)
        return ImageGenerationResponse(
            job_id="", status=GenerationStatus.FAILED, error_message="ningún driver analiza imágenes"
        )

    async def check_analysis_status(self, job_id: str) -> ImageGenerationResponse:
        for driver in (self._primary, self._secondary):
            if hasattr(driver, "check_analysis_status"):
                response = await driver.check_analysis_status(job_id)
                if response.status != GenerationStatus.FAILED:
                    return response
        return ImageGenerationResponse(
            job_id=job_id, status=GenerationStatus.FAILED, error_message="trabajo desconocido"
        )
