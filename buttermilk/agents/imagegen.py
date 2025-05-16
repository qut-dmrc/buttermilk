# Multi-model text-to-image generation

import asyncio
import json
import os
import random
import uuid
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import aiohttp
import httpx
import replicate
from cloudpathlib import CloudPath
from google import genai
from google.genai import types
from huggingface_hub import AsyncInferenceClient, login
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from shortuuid import ShortUUID

from buttermilk._core.image import ImageRecord, read_image
from buttermilk._core.log import logger
from buttermilk._core.retry import RetryWrapper


class TextToImageClient(RetryWrapper):
    client: object = None
    model: str = Field(
        ...,
        description="The model to use for text-to-image generation.",
    )
    prefix: str

    async def generate(
        self,
        text: str,
        save_path: str = "",
        negative_prompt: str = "",
        filetype: Literal["png"] = "png",
        **kwargs,
    ) -> ImageRecord:
        if not save_path:
            save_path = f"{bm.save_dir}/{self.prefix}{uuid.uuid4()}.{filetype}"
        if negative_prompt:
            msg = f"Generating image with {self.model} using prompt: ```{text}```, negative prompt: ```{negative_prompt}```, saving to `{save_path}`"
        else:
            msg = f"Generating image with {self.model} using prompt: ```{text}```, saving to `{save_path}`"
        logger.info(msg)

        parameters = dict(
            text=text,
            negative_prompt=negative_prompt,
            **kwargs,
        )

        try:
            image = await self._execute_with_retry(
                self.generate_image,
                **parameters,
            )
            image.uri = image.save(save_path)
            image.prompt = text
            image.negative_prompt = negative_prompt
            return image

        except Exception as e:
            err = f"Error generating image for {text} using {self.model}: {e or type(e)} {e.args=}"
            args = [str(x) for x in e.args]
            error_dict = dict(
                prompt=text,
                model=self.model,
                message=str(e),
                args=args,
                type=type(e).__name__,
            )

        logger.error(err)
        image = ImageRecord(
            image=None,
            model=self.model,
            parameters=parameters,
            error=error_dict,
        )

        return image

    async def generate_image(
        self,
        text: str,
        negative_prompt: str = "",
        **kwargs,
    ) -> ImageRecord:
        """Subclasses must implement this method."""
        raise NotImplementedError


class Imagegen3(TextToImageClient):
    model: str = "imagen-3.0-generate-002"
    prefix: str = "imagen3_"
    client: genai.Client = Field(
        default_factory=lambda: genai.Client(
            vertexai=True,
            project=bm.credentials["GOOGLE_CLOUD_PROJECT"],
            location=bm.credentials["GOOGLE_CLOUD_LOCATION"],
        ),
    )  # type: ignore
    fast: bool = Field(default=False, description="Use the faster, lower quality model")

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        aspect_ratio="3:4",
        **kwargs,
    ) -> ImageRecord:
        parameters = dict(
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            safety_filter_level="BLOCK_ONLY_HIGH",
            person_generation="ALLOW_ADULT",
            negative_prompt=negative_prompt,
        )

        # Imagen 3 image generation
        image = self.client.models.generate_images(
            model=self.model,
            prompt=text,
            config=types.GenerateImagesConfig(
                **parameters,
            ),
        )
        if not image.generated_images:
            raise ValueError(f"No image generated. Response: {image.model_dump()}")
        result = image.generated_images[0]
        pil_image = result.image._pil_image
        enhanced_prompt = result.enhanced_prompt

        image = ImageRecord(
            image=pil_image,
            model=self.model,
            parameters=parameters,
            prompt=prompt,
            enhanced_prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
        )
        return image


class SD35Large(TextToImageClient):
    # Stable Diffusion 3 using Azure
    model: str = "stabilityai/stable-diffusion-3.5-large"
    prefix: str = "sd35_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        size: str = "1024x1024",
        output_format: str = "png",
        seed: int = 0,
        **kwargs,
    ) -> ImageRecord:
        request_data = {
            "prompt": text,
            "negative_prompt": negative_prompt,
            "size": size,
            "output_format": output_format,
            "seed": seed,
        }

        url = bm.credentials["AZURE_STABILITY35_URL"]
        api_key = bm.credentials["AZURE_STABILITY35_API_KEY"]

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": ("Bearer " + api_key),
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                headers=headers,
                json=request_data,
                timeout=600,
            )
        response.raise_for_status()
        image = read_image(image_b64=response.json().get("image"))
        image.model = self.model
        image.parameters = request_data

        return image


class Imagegen3Fast(Imagegen3):
    mode: str = "imagen-3.0-fast-generate-001"


class SD3(TextToImageClient):
    # Stable Diffusion 3 using api.stability.ai
    model: str = "stabilityai/stable-diffusion-3"
    prefix: str = "sd3_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        output_format: str = "png",
        **kwargs,
    ) -> ImageRecord:
        parameters = dict(prompt=text, output_format=output_format, **kwargs)

        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt

        form = aiohttp.FormData()
        for k, v in parameters.items():
            form.add_field(k, v, content_type="multipart/form-data")

        timeout = aiohttp.ClientTimeout(total=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": f"Bearer {os.environ['STABILITY_AI_KEY']}",
                    "accept": "image/*",
                },
                data=form,
            )
            response.raise_for_status()
            data = await response.read()
            img = Image.open(BytesIO(data))

        # add extra keys not passed to model after generation
        parameters.update(**kwargs)

        image = ImageRecord(
            image=img,
            model=self.model,
            parameters=parameters,
            prompt=text,
            negative_prompt=negative_prompt,
        )
        return image


class SDXL(TextToImageClient):
    # Stable Diffusion XL using HuggingFace Hub
    model: str = "stabilityai/sdxl-base-refiner-1.0"
    prefix: str = "sdxl_"
    image_params: dict[str, Any] = {"num_inference_steps": 50}
    refiner: Any = None

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> ImageRecord:
        if self.client is None:
            # access token with permission to access the model and PRO subscription
            login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"], new_session=False)
            self.client = AsyncInferenceClient(
                "stabilityai/stable-diffusion-xl-base-1.0",
                timeout=600,
            )
            self.refiner = AsyncInferenceClient(
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                timeout=600,
            )
        parameters = dict(prompt=text, negative_prompt=negative_prompt, **self.image_params)
        img = await self.client.text_to_image(**parameters)

        # Use both stages of the model to refine the image
        # refined_img = await self.refiner.image_to_image(
        #     prompt=text, denoising_start=0.8, image=img.tobytes()
        # )
        from huggingface_hub import InferenceClient

        self.refiner = InferenceClient(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            timeout=600,
        )
        img_to_refine = ImageRecord(image=img).as_bytestream()
        refined_img = self.refiner.image_to_image(img_to_refine, text)
        # add extra keys not passed to model after generation
        parameters.update(**kwargs)

        image = ImageRecord(
            image=refined_img,
            model=self.model,
            parameters=parameters,
            prompt=text,
            negative_prompt=negative_prompt,
        )

        return image


class SDXLReplicate(TextToImageClient):
    # Stable Diffusion XL using Replicate
    model: str = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"
    prefix: str = "sdxl_replicate_"
    image_params: dict[str, Any] = {
        "width": 768,
        "height": 768,
        "refine": "expert_ensemble_refiner",
        "apply_watermark": True,
        "num_inference_steps": 50,
    }
    refiner: Any = None

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> ImageRecord:
        parameters = dict(prompt=text, negative_prompt=negative_prompt, **self.image_params)
        response = await replicate.async_run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input=parameters,
        )
        output_uri = response[0]

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(output_uri)
            response.raise_for_status()

            img = Image.open(BytesIO(await response.content.read()))
        # add extra keys not passed to model after generation
        parameters.update(**kwargs)

        image = ImageRecord(
            image=img,
            model=self.model,
            parameters=parameters,
            prompt=text,
            negative_prompt=negative_prompt,
        )

        return image


class SD(TextToImageClient):
    # Stable Diffusion 2.1 using Replicate
    model: str = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
    prefix: str = "sd21_"
    image_params: dict[str, Any] = {
        "width": 1024,
        "height": 1024,
        "scheduler": "DPMSolverMultistep",
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
    }

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> ImageRecord:
        parameters = self.image_params.copy()
        parameters["prompt"] = text
        if negative_prompt:
            parameters["negative_prompt"] = negative_prompt
        output = await replicate.async_run(self.model, input=parameters)

        # add extra keys not passed to model after generation
        parameters.update(**kwargs)

        output_uri = output[0]
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(output_uri)
        response.raise_for_status()
        data = await response.read()
        img = Image.open(BytesIO(data))
        image = ImageRecord(
            image=img,
            model=self.model,
            parameters=parameters,
            prompt=text,
            negative_prompt=negative_prompt,
        )
        return image


class DALLE(TextToImageClient):
    model: str = "dall-e-3"
    prefix: str = "dalle3_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = None,
        size="1024x1024",
        style="natural",
        quality="standard",
        **kwargs,
    ) -> ImageRecord:
        parameters = dict(
            model=self.model,
            size=size,
            style=style,
            quality=quality,
            **kwargs,
        )
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=bm.credentials["OPENAI_API_KEY"],
                timeout=600,
            )
        if negative_prompt:
            parameters["prompt"] = f"{text} \nDO NOT INCLUDE: {negative_prompt}"
        else:
            parameters["prompt"] = text

        response = await self.client.images.generate(
            **parameters,
        )

        # add extra keys not passed to model after generation
        parameters.update(**kwargs)

        output_uri = response.data[0].url
        if revised_prompt := response.data[0].revised_prompt:
            parameters["revised_prompt"] = revised_prompt
        async with httpx.AsyncClient() as client:
            response = await client.get(output_uri)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        image = ImageRecord(
            image=img,
            model=self.model,
            parameters=parameters,
            prompt=text,
            negative_prompt=negative_prompt,
        )
        return image


ImageClients = [DALLE, SD35Large, Imagegen3, Imagegen3Fast]


class BatchImageGenerator(BaseModel):
    """Generates images from text using a variety of models."""

    generators: Sequence[Any] = Field(
        default=ImageClients,
        description="The image generators to use.",
    )
    save_path: CloudPath | Path = Field(
        default=None,
        description="The path to save the generated images to.",
    )
    _clients: dict[str, TextToImageClient] = {}
    _tasks: Sequence[asyncio.Task] = []

    @field_validator("save_path")
    def _convert_save_path(
        cls,
        v: str | CloudPath | Path,
    ) -> CloudPath | Path:
        if not v:
            # By default, save to the current BM run path
            v = bm.save_dir
        if isinstance(v, str):
            return CloudPath(v)
        if isinstance(v, (CloudPath, Path)):
            return v
        raise ValueError(
            f"save_path must be a string, Path, or CloudPath, got {type(v)}",
        )

    def init_clients(self):
        self._clients = {gen.__name__: gen() for gen in self.generators}

    async def abatch(self, input: Sequence[str | dict[str, str]], n=1):
        """Generate images from a list of texts using multiple models asynchronously."""
        self.init_clients()
        batch_path = self.save_path / ShortUUID().uuid()[:12]
        info_path = batch_path / "info.json"
        records = []
        summary = []

        prompts = []
        for i, prompt in enumerate(input):
            if isinstance(prompt, str):
                prompts.append(dict(text=prompt, id=i))
            elif isinstance(prompt, dict):
                prompts.append(
                    dict(
                        text=prompt["text"],
                        id=prompt.get("id", i),
                        negative_prompt=prompt.get("negative_prompt"),
                    ),
                )
            else:
                raise ValueError(
                    f"Invalid prompt type, expected str or dict, got {type(prompt)}",
                )

        # shuffle prompts
        random.shuffle(prompts)

        try:
            for input in prompts:
                for j in range(n):
                    for name, model in self._clients.items():
                        # await asyncio.sleep(0.1)
                        try:
                            img_path = (batch_path / f"{input['id']}_{model.prefix}{j}_{ShortUUID().uuid()[:12]}.png").as_uri()
                            self._tasks.append(
                                model.generate(**input, save_path=img_path),
                            )

                        except Exception as e:
                            logger.error(
                                f"Error adding task to generate image from {name}, attempt {j}, prompt {input}: {e} {e.args=}",
                            )
                            continue

            for t in asyncio.as_completed(self._tasks):
                try:
                    await asyncio.sleep(0.1)
                    result = await t
                    # if not using as_completed(): result = t.result()
                    yield result
                    info = dict(
                        parameters=result.parameters,
                        model=result.model,
                        uri=result.uri,
                    )
                    summary.append(info)

                except Exception as e:
                    logger.error(
                        f"Error collecting result from task {t}: {e} {e.args=}",
                    )
                    continue

        except ExceptionGroup as eg:
            for e in eg.exceptions:
                logger.exception(
                    f"Received unhandled exception  (in ExceptionGroup)! {e}",
                    extra={"traceback": e.__traceback__},
                )

        info_path.write_text(json.dumps(summary))
        logger.info(
            f"Generated {len(records)} images from {len(input)} prompts, using {self.generators}. Saved summary to {info_path.as_uri()}",
        )
