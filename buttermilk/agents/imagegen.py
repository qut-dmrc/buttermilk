# Multi-model text-to-image generation

import asyncio
import json
import os
import random
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
import httpx
import replicate
from cloudpathlib import CloudPath
from datatools.exceptions import RateLimit
from datatools.gcloud import GCloud
from datatools.types.images import ImageRecord
from huggingface_hub import AsyncInferenceClient, login
from openai import APIStatusError, AsyncOpenAI
from PIL import Image
from promptflow.tracing import trace
from pydantic import BaseModel
from requests import ConnectTimeout, HTTPError
from shortuuid import ShortUUID
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

uuid = ShortUUID()


from pydantic import Field, field_validator

gc = GCloud()
logger = gc.logger


class TextToImageClient(BaseModel):
    client: Any | None = None
    model: str = Field(
        ...,
        description="The model to use for text-to-image generation.",
    )
    prefix: str
    image_params: dict[str, Any] = {}

    @trace
    async def generate(
        self,
        text,
        save_path,
        negative_prompt: str = "",
        **kwargs,
    ) -> ImageRecord:
        if negative_prompt:
            msg = f"Generating image with {self.model} using prompt: ```{text}```, negative prompt: ```{negative_prompt}```, saving to `{save_path}`"
        else:
            msg = f"Generating image with {self.model} using prompt: ```{text}```, saving to `{save_path}`"

        params = dict(prompt=text, save_path=save_path, negative_prompt=negative_prompt, **self.image_params)
        params.update(**kwargs)

        try:
            image = await self.generate_with_retry(
                **params,
            )
            image.uri = image.save(save_path)
            return image

        except (ConnectTimeout, TimeoutError) as e:
            err = f"Timeout error {msg}: {e or type(e)} {e.args=}"
            args = [str(x) for x in e.args]
            error_dict = dict(prompt=text, model=self.model, message=str(e), args=args, type=type(e).__name__)
        except APIStatusError as e:
            err = f"Error generating image for {text} using {self.model}: {e or type(e)} {e.args=}"
            error_dict = dict(
                prompt=text,
                model=self.model,
                status_code=e.status_code,
                request_id=e.request_id,
                type=type(e).__name__,
            )
            error_dict.update(e.body)
        except replicate.exceptions.ModelError as e:
            err = f"Error generating image for {text} using {self.model}: {e or type(e)} {e.args=}"
            error_dict = dict(prompt=text, model=self.model, message=str(e), type=type(e).__name__)
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
            params=params,
            error=error_dict,
        )

        return image

    @retry(
        retry=retry_if_exception_type((RateLimit, HTTPError)),
        wait=wait_random_exponential(multiplier=0.5, max=60),
        stop=stop_after_attempt(7),
    )
    async def generate_with_retry(
        self,
        prompt: str,
        negative_prompt: str = "",
        **kwargs,
    ) -> ImageRecord:
        return await self.generate_image(
            prompt,
            negative_prompt=negative_prompt,
            **kwargs,
        )

    async def generate_image(
        self,
        text: str,
        negative_prompt: str = "",
        **kwargs,
    ) -> ImageRecord:
        """Subclasses must implement this method."""
        raise NotImplementedError


class SD3(TextToImageClient):
    # Stable Diffusion 3 using api.stability.ai
    model: str = "stabilityai/stable-diffusion-3"
    prefix: str = "sd3_"
    image_params: dict[str, Any] = {"output_format": "png"}

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        **kwargs,
    ) -> ImageRecord:
        params = self.image_params.copy()

        params["prompt"] = text
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        form = aiohttp.FormData()
        for k, v in params.items():
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
        params.update(**kwargs)

        image = ImageRecord(
            image=img,
            model=self.model,
            params=params,
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
        params = dict(prompt=text, negative_prompt=negative_prompt, **self.image_params)
        img = await self.client.text_to_image(**params)

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
        params.update(**kwargs)

        image = ImageRecord(
            image=refined_img,
            model=self.model,
            params=params,
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
        params = dict(prompt=text, negative_prompt=negative_prompt, **self.image_params)
        response = await replicate.async_run(
            "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
            input=params,
        )
        output_uri = response[0]

        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            response = await session.get(output_uri)
            response.raise_for_status()

            img = Image.open(BytesIO(await response.content.read()))
        # add extra keys not passed to model after generation
        params.update(**kwargs)

        image = ImageRecord(
            image=img,
            model=self.model,
            params=params,
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
        params = self.image_params.copy()
        params["prompt"] = text
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        output = await replicate.async_run(self.model, input=params)

        # add extra keys not passed to model after generation
        params.update(**kwargs)

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
            params=params,
        )
        return image


class DALLE(TextToImageClient):
    model: str = "dall-e-3"
    image_params: dict[str, Any] = dict(size="1024x1024", quality="standard")
    prefix: str = "dalle3_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = None,
        **kwargs,
    ) -> ImageRecord:
        params = self.image_params.copy()
        if self.client is None:
            self.client = AsyncOpenAI()
        if negative_prompt:
            params["prompt"] = f"{text} \nDO NOT INCLUDE: {negative_prompt}"
        else:
            params["prompt"] = text

        response = await self.client.images.generate(model=self.model, **params)

        # add extra keys not passed to model after generation
        params.update(**kwargs)

        output_uri = response.data[0].url
        if revised_prompt := response.data[0].revised_prompt:
            params["revised_prompt"] = revised_prompt
        async with httpx.AsyncClient() as client:
            response = await client.get(output_uri)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        image = ImageRecord(image=img, model=self.model, params=params)
        return image


ImageClients = [
    DALLE,
    SD,
    SD3,
    SDXL,
    SDXLReplicate,
]


class BatchImageGenerator(BaseModel):
    """Generates images from text using a variety of models."""

    generators: Sequence[Any] = Field(
        default=ImageClients,
        description="The image generators to use.",
    )
    save_path: CloudPath | Path = Field(
        ...,
        description="The path to save the generated images to.",
    )
    _clients: dict[str, TextToImageClient] = {}
    _tasks: Sequence[asyncio.Task] = []

    @field_validator("save_path")
    def convert_save_path(
        cls,
        v: str | CloudPath | Path,
    ) -> CloudPath | Path:
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
                            img_path = (
                                batch_path
                                / f"{input['id']}_{model.prefix}{j}_{ShortUUID().uuid()[:12]}.png"
                            ).as_uri()
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
                        params=result.params,
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
