"""Multi-model text-to-image generation using various APIs and services.

This module provides a base client `TextToImageClient` with retry capabilities
and several concrete implementations for different image generation models/services:
- Google's Imagen 3 (via Vertex AI)
- Stable Diffusion 3.5 Large (via Azure)
- Stable Diffusion 3 (via Stability AI API)
- Stable Diffusion XL (via HuggingFace Hub and Replicate)
- Stable Diffusion 2.1 (via Replicate)
- DALL-E 3 (via OpenAI API)

It also includes `BatchImageGenerator` for generating images from multiple prompts
using a selection of these clients asynchronously.
"""

import asyncio
import json
import os # For accessing environment variables (e.g., API keys)
import random
import uuid # For generating unique IDs for save paths
from collections.abc import Sequence # For type hinting sequences
from io import BytesIO # For handling image data in memory
from pathlib import Path # For local path manipulation
from typing import Any, Literal, Type # For type hinting

import aiohttp # Asynchronous HTTP client (used by SD3, SDXLReplicate, SD)
import httpx # Asynchronous HTTP client (used by SD35Large, DALLE)
import replicate # Client for Replicate API
from cloudpathlib import CloudPath # For handling cloud storage paths
from google import genai # Google Generative AI client
from google.genai import types as google_genai_types # Specific types from google.genai
from huggingface_hub import AsyncInferenceClient, login # HuggingFace Hub client
from openai import AsyncOpenAI # OpenAI client
from PIL import Image # Pillow library for image manipulation
from pydantic import BaseModel, Field, PrivateAttr, field_validator # Pydantic components
from shortuuid import ShortUUID # For generating short unique IDs

from buttermilk import buttermilk as bm  # Global Buttermilk instance for accessing config/credentials
from buttermilk._core.image import ImageRecord, read_image # Buttermilk ImageRecord model
from buttermilk._core.log import logger # Centralized logger
from buttermilk._core.retry import RetryWrapper # Base class for retry logic


class TextToImageClient(RetryWrapper):
    """Base class for text-to-image generation clients, incorporating retry logic.

    This class extends `RetryWrapper` to provide a common interface and
    functionality for various image generation model clients. Subclasses must
    implement the `generate_image` method to interact with their specific API.

    The `generate` method handles logging, parameter preparation, calling the
    subclass's `generate_image` with retry logic, and saving the resulting image.

    Attributes:
        client (object): The actual API client object used by the subclass.
            Initialized to None, subclasses should populate this.
        model (str): The specific model identifier to be used by this client
            (e.g., "dall-e-3", "imagen-3.0-generate-002"). This is a mandatory field.
        prefix (str): A short prefix string used for naming generated image files,
            helping to identify which model created them.
    """
    client: object = Field(default=None, description="The underlying API client object for the image generation service.")
    model: str = Field(
        ..., # Ellipsis indicates a required field
        description="The specific model identifier to use for text-to-image generation.",
    )
    prefix: str = Field(description="A short prefix for naming generated image files (e.g., 'dalle3_', 'sdxl_').")

    async def generate(
        self,
        text: str,
        save_path: str = "",
        negative_prompt: str = "",
        filetype: Literal["png"] = "png", # Currently only supports PNG
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image based on a text prompt using the configured model and client.

        This method prepares parameters, logs the request, invokes the subclass's
        `generate_image` method (with retries via `_execute_with_retry`),
        saves the generated image to the specified `save_path` (or a default
        location), and returns an `ImageRecord` containing the image, its URI,
        and metadata. If an error occurs, it logs the error and returns an
        `ImageRecord` with error details.

        Args:
            text: The textual prompt to generate the image from.
            save_path: Optional. The full path (local or cloud) where the generated
                image should be saved. If empty, a default path is constructed
                using `bm.save_dir`, the client's `prefix`, a UUID, and `filetype`.
            negative_prompt: Optional. A textual prompt describing elements to avoid
                in the generated image.
            filetype: The desired filetype for the saved image. Currently defaults
                to "png" and is the only explicitly supported type in save path generation.
            **kwargs: Additional keyword arguments to be passed to the subclass's
                `generate_image` method and included in the `ImageRecord.parameters`.

        Returns:
            ImageRecord: An `ImageRecord` instance containing the generated PIL Image,
            its URI after saving, model details, parameters, prompt, and any error
            information if generation failed.
        """
        final_save_path = save_path
        if not final_save_path:
            # Ensure bm.save_dir is available and valid
            if not bm.save_dir or not Path(bm.save_dir).is_dir(): # Check if local dir, cloud paths need different check
                 # Fallback if bm.save_dir is not set or invalid, consider a temporary directory
                import tempfile
                temp_dir = tempfile.mkdtemp()
                logger.warning(f"bm.save_dir not available or invalid, using temporary directory: {temp_dir}")
                final_save_path = f"{temp_dir}/{self.prefix}{uuid.uuid4()}.{filetype}"
            else:
                final_save_path = f"{bm.save_dir}/{self.prefix}{uuid.uuid4()}.{filetype}"
        
        log_msg_parts = [
            f"Generating image with {self.model} using prompt: ```{text}```"
        ]
        if negative_prompt:
            log_msg_parts.append(f", negative prompt: ```{negative_prompt}```")
        log_msg_parts.append(f", saving to `{final_save_path}`")
        logger.info("".join(log_msg_parts))

        parameters_for_generation = {
            "text": text,
            "negative_prompt": negative_prompt,
            **kwargs, # Include any additional model-specific parameters
        }

        try:
            # _execute_with_retry calls self.generate_image (the subclass implementation)
            generated_image_record = await self._execute_with_retry(
                self.generate_image, # The method to call with retry
                **parameters_for_generation, # Arguments for generate_image
            )
            
            # Ensure generated_image_record is an ImageRecord and has an image
            if not isinstance(generated_image_record, ImageRecord) or not generated_image_record.image:
                raise ValueError("generate_image implementation did not return a valid ImageRecord with an image.")

            # Save the image and update the URI in the record
            # The save method is part of ImageRecord
            generated_image_record.uri = generated_image_record.save(final_save_path)
            
            # Ensure prompt, negative_prompt, and other relevant params are in the final record
            generated_image_record.prompt = text
            generated_image_record.negative_prompt = negative_prompt
            # Parameters used for generation should already be in generated_image_record.parameters
            # by the subclass's generate_image method.

            return generated_image_record

        except Exception as e:
            error_message = f"Error generating image for prompt '{text[:100]}...' using {self.model}: {type(e).__name__} - {e!s}"
            logger.error(error_message, exc_info=True) # Log with stack trace
            
            error_details = {
                "prompt": text,
                "model": self.model,
                "message": str(e),
                "args": [str(arg) for arg in e.args], # Convert args to string
                "type": type(e).__name__,
            }
            # Return an ImageRecord indicating failure
            return ImageRecord(
                image=None, # No image generated
                model=self.model,
                parameters=parameters_for_generation, # Parameters that were attempted
                error=error_details,
                prompt=text,
                negative_prompt=negative_prompt,
            )

    async def generate_image(
        self,
        text: str,
        negative_prompt: str = "",
        **kwargs: Any,
    ) -> ImageRecord:
        """Abstract method for generating an image based on text.
        
        Subclasses must implement this method to provide the specific logic for
        interacting with their respective image generation APIs. The implementation
        should return an `ImageRecord` containing the generated PIL Image and
        relevant metadata (model, parameters, prompt, etc.). The `uri` field
        of the `ImageRecord` will be populated by the `generate` method after saving.

        Args:
            text: The textual prompt.
            negative_prompt: Optional negative prompt.
            **kwargs: Additional model-specific parameters.

        Returns:
            ImageRecord: An `ImageRecord` containing the generated image and metadata.

        Raises:
            NotImplementedError: If a subclass does not implement this method.
        """
        raise NotImplementedError("Subclasses must implement the `generate_image` method.")


class Imagegen3(TextToImageClient):
    """Client for Google's Imagen 3 model via Vertex AI.

    Uses `google-cloud-aiplatform` (implicitly via `google.genai` configured for Vertex).
    Requires `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` to be available
    in `bm.credentials`.

    Attributes:
        model (str): Defaults to "imagen-3.0-generate-002".
        prefix (str): File prefix defaults to "imagen3_".
        client (genai.GenerativeModel): Initialized Google Generative AI client for Imagen 3.
        fast (bool): If True, uses a faster, potentially lower-quality variant of the model.
                     (Note: Current implementation always uses `self.model`, `fast` field
                      might be for future use or needs integration into model selection).
    """
    model: str = "imagen-3.0-generate-002"
    prefix: str = "imagen3_"
    client: genai.GenerativeModel = Field( # Corrected type to GenerativeModel
        default_factory=lambda: genai.GenerativeModel(
            model_name=Imagegen3.model, # Use the class's model attribute
            # Assumes genai client is configured globally or bm.credentials provide necessary setup
            # For Vertex, client initialization might need project/location.
            # genai.configure(credentials=bm.gcp_credentials) might be needed elsewhere.
            # This factory might need adjustment if genai.Client requires explicit project/location.
            # The original had genai.Client(vertexai=True, project=..., location=...)
            # but genai.GenerativeModel is the typical interface for specific models.
        ),
        description="Google Generative AI client for Imagen 3."
    )
    fast: bool = Field(default=False, description="If True, use a faster model variant (currently illustrative).")
    # TODO: The 'fast' attribute currently doesn't change the model string.
    # If 'fast' is intended to select a different model like "imagen-3.0-fast-generate-001",
    # the model string should be dynamically chosen in __init__ or generate_image.

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        aspect_ratio: str ="3:4", # Default aspect ratio
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using Google's Imagen 3 model.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            aspect_ratio: Desired aspect ratio (e.g., "1:1", "16:9", "3:4").
            **kwargs: Additional parameters for `types.GenerateImagesConfig`.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.

        Raises:
            ValueError: If the API returns no generated images.
        """
        # Consolidate parameters for GenerateImagesConfig
        # Parameters passed in kwargs will override defaults here if names match
        generation_params = {
            "number_of_images": 1, # Default to 1 image
            "aspect_ratio": aspect_ratio,
            "safety_filter_level": "BLOCK_ONLY_HIGH", # Example safety setting
            "person_generation": "ALLOW_ADULT", # Example setting for person generation
            "negative_prompt": negative_prompt or "", # Ensure empty string if None
            **kwargs, # Allow overrides and additional params
        }
        
        # Ensure client is initialized with correct project and location for Vertex AI
        # This might be better handled in a root_validator or __init__ if bm is always available
        # For now, assuming bm.credentials are accessible as in original.
        if self.client._client is None: # type: ignore # Accessing private member for check
             # This re-initialization logic might be problematic if client is shared.
             # Ideally, client configuration is finalized at instantiation.
            genai.configure(
                credentials=bm.gcp_credentials, # Assumes bm.gcp_credentials is set
                project=bm.credentials["GOOGLE_CLOUD_PROJECT"],
                location=bm.credentials["GOOGLE_CLOUD_LOCATION"]
            )
            self.client = genai.GenerativeModel(self.model)


        # Actual API call to generate images
        # Note: The google.genai library's async capabilities might differ.
        # The original used a synchronous client.generate_images.
        # If an async version is available, it should be used.
        # For now, assuming a blocking call or that the client handles async internally.
        # If client.generate_images is not async, it should be wrapped with asyncio.to_thread
        
        # Simulating async call if client.generate_images is sync
        loop = asyncio.get_event_loop()
        api_response = await loop.run_in_executor(
            None, # Uses default ThreadPoolExecutor
            self.client.generate_images,
            text, # prompt argument for generate_images
            google_genai_types.GenerateImagesConfig(**generation_params) # config argument
        )

        if not api_response.generated_images:
            raise ValueError(f"No image generated by Imagen 3. Response: {api_response}") # Or handle differently
        
        first_generated_image = api_response.generated_images[0]
        pil_image = first_generated_image.image._pil_image # Accessing private member, might be fragile
        enhanced_prompt = first_generated_image.enhanced_prompt

        return ImageRecord(
            image=pil_image,
            model=self.model,
            parameters=generation_params, # Log the actual parameters used
            prompt=text, # Original prompt
            enhanced_prompt=enhanced_prompt, # Prompt as understood/enhanced by model
            negative_prompt=negative_prompt,
        )


class SD35Large(TextToImageClient):
    """Client for Stability AI's Stable Diffusion 3.5 Large model via Azure.

    Requires `AZURE_STABILITY35_URL` and `AZURE_STABILITY35_API_KEY` in `bm.credentials`.

    Attributes:
        model (str): Defaults to "stabilityai/stable-diffusion-3.5-large".
        prefix (str): File prefix defaults to "sd35_".
    """
    model: str = "stabilityai/stable-diffusion-3.5-large"
    prefix: str = "sd35_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        size: str = "1024x1024", # Default size
        output_format: str = "png", # Default output format
        seed: int = 0, # Default seed
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using Stable Diffusion 3.5 Large via Azure.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            size: Image dimensions (e.g., "1024x1024").
            output_format: Image format (e.g., "png", "jpeg").
            seed: Seed for generation. Use 0 for a random seed if API supports it,
                  or manage random seed externally if API requires specific non-zero seed.
            **kwargs: Additional parameters for the API request.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.
        """
        request_data = {
            "prompt": text,
            "negative_prompt": negative_prompt or "", # Ensure empty string if None
            "size": size,
            "output_format": output_format,
            "seed": seed,
            **kwargs, # Allow overrides and additional params
        }

        azure_url = bm.credentials.get("AZURE_STABILITY35_URL")
        azure_api_key = bm.credentials.get("AZURE_STABILITY35_API_KEY")

        if not azure_url or not azure_api_key:
            raise ValueError("Azure Stability 3.5 URL or API Key not configured in bm.credentials.")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json", # Or "image/png" etc. if API supports direct image response
            "Authorization": f"Bearer {azure_api_key}",
        }
        
        async with httpx.AsyncClient() as http_client: # Renamed to avoid conflict
            response = await http_client.post(
                azure_url,
                headers=headers,
                json=request_data,
                timeout=600.0, # Increased timeout
            )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Assuming API returns JSON with base64 image data under "image" key
        response_json = response.json()
        image_b64 = response_json.get("image")
        if not image_b64:
            raise ValueError("Azure Stability 3.5 API response did not include an 'image' field.")

        # read_image utility is expected to handle base64 string and return ImageRecord
        image_record = read_image(image_b64=image_b64)
        image_record.model = self.model
        image_record.parameters = request_data # Log parameters used
        image_record.prompt = text
        image_record.negative_prompt = negative_prompt

        return image_record


class Imagegen3Fast(Imagegen3):
    """Client for the faster variant of Google's Imagen 3 model.

    Inherits from `Imagegen3` but overrides the model identifier.

    Attributes:
        model (str): Overridden to "imagen-3.0-fast-generate-001".
                     (Note: The original had 'mode', changed to 'model' for consistency).
    """
    model: str = "imagen-3.0-fast-generate-001" # Corrected from 'mode' to 'model'


class SD3(TextToImageClient):
    """Client for Stability AI's Stable Diffusion 3 model via `api.stability.ai`.

    Requires `STABILITY_AI_KEY` environment variable to be set.

    Attributes:
        model (str): Defaults to "stabilityai/stable-diffusion-3".
        prefix (str): File prefix defaults to "sd3_".
    """
    model: str = "stabilityai/stable-diffusion-3"
    prefix: str = "sd3_"

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        output_format: str = "png", # Default output format
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using Stable Diffusion 3 via Stability AI's API.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            output_format: Desired output image format (e.g., "png", "jpeg").
            **kwargs: Additional parameters for the Stability AI API.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.
        
        Raises:
            KeyError: If `STABILITY_AI_KEY` environment variable is not set.
        """
        stability_api_key = os.environ.get("STABILITY_AI_KEY")
        if not stability_api_key:
            raise KeyError("STABILITY_AI_KEY environment variable not set for SD3 client.")

        parameters = {
            "prompt": text,
            "output_format": output_format,
            **kwargs, # Allow overrides and additional params
        }
        if negative_prompt: # Add negative_prompt only if provided
            parameters["negative_prompt"] = negative_prompt

        form = aiohttp.FormData()
        for key, value in parameters.items():
            if value is not None: # Only add fields if they have a value
                form.add_field(key, str(value)) # Ensure value is string for form data

        timeout = aiohttp.ClientTimeout(total=600.0) # Increased timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                "https://api.stability.ai/v2beta/stable-image/generate/sd3",
                headers={
                    "authorization": f"Bearer {stability_api_key}",
                    "accept": "image/*", # Accepts any image format
                },
                data=form,
            ) as response:
                response.raise_for_status() # Raise HTTPError for bad responses
                image_data = await response.read()
        
        pil_image = Image.open(BytesIO(image_data))

        return ImageRecord(
            image=pil_image,
            model=self.model,
            parameters=parameters, # Log actual parameters sent
            prompt=text,
            negative_prompt=negative_prompt,
        )


class SDXL(TextToImageClient):
    """Client for Stability AI's Stable Diffusion XL (SDXL) model using HuggingFace Hub.

    Uses `AsyncInferenceClient` from `huggingface_hub`. Requires
    `HUGGINGFACEHUB_API_TOKEN` environment variable. This implementation uses
    both the base SDXL model and a refiner model.

    Attributes:
        model (str): Defaults to "stabilityai/sdxl-base-refiner-1.0" (this seems
            to combine base and refiner in its name, but the code uses separate clients).
            Consider renaming to reflect the base model, e.g.,
            "stabilityai/stable-diffusion-xl-base-1.0".
        prefix (str): File prefix defaults to "sdxl_".
        image_params (dict[str, Any]): Default parameters for image generation
            (e.g., `num_inference_steps`).
        _refiner_client (AsyncInferenceClient | None): Asynchronous client for the SDXL refiner model.
        _sync_refiner_client (InferenceClient | None): Synchronous client for the SDXL refiner model,
            used in the original refinement step.
    """
    model: str = "stabilityai/stable-diffusion-xl-base-1.0" # Changed to base model
    prefix: str = "sdxl_"
    image_params: dict[str, Any] = Field(default_factory=lambda: {"num_inference_steps": 50})
    
    # client is inherited from TextToImageClient for the base model
    # refiner client needs to be initialized
    _refiner_client: AsyncInferenceClient | None = PrivateAttr(default=None) # For async refiner
    _sync_refiner_client: Any | None = PrivateAttr(default=None) # For sync refiner used in original

    async def _initialize_clients_if_needed(self) -> None:
        """Initializes HuggingFace clients if not already done."""
        if self.client is None: # client for base model
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
            if not hf_token:
                raise KeyError("HUGGINGFACEHUB_API_TOKEN environment variable not set for SDXL client.")
            try:
                login(token=hf_token, new_session=False) # Login to HuggingFace Hub
            except Exception as e:
                logger.warning(f"HuggingFace Hub login failed (token might be invalid or already logged in): {e!s}")

            self.client = AsyncInferenceClient(
                "stabilityai/stable-diffusion-xl-base-1.0", # Base model
                token=hf_token, # Pass token explicitly
                timeout=600.0,
            )
        
        # Initialize refiner client (example uses sync, should ideally be async)
        # The original code mixed sync and async refiner client initialization.
        # Sticking to async for consistency if possible, or noting the sync usage.
        if self._sync_refiner_client is None: # Using the sync client as per original code's refine step
            from huggingface_hub import InferenceClient as SyncInferenceClient # Local import for clarity
            hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN") # Get token again just in case
            self._sync_refiner_client = SyncInferenceClient(
                 "stabilityai/stable-diffusion-xl-refiner-1.0",
                 token=hf_token,
                 timeout=600.0,
            )


    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using SDXL base and then refines it.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            **kwargs: Additional parameters, merged with `self.image_params`.

        Returns:
            ImageRecord: An `ImageRecord` with the refined image and metadata.
        
        Raises:
            KeyError: If `HUGGINGFACEHUB_API_TOKEN` is not set.
            RuntimeError: If client initialization fails or image generation/refinement fails.
        """
        await self._initialize_clients_if_needed() # Ensure clients are ready
        if self.client is None or self._sync_refiner_client is None: # Check after init
             raise RuntimeError("SDXL clients (base or refiner) not properly initialized.")

        # Combine default image_params with any runtime kwargs
        final_parameters = {**self.image_params, "prompt": text, "negative_prompt": negative_prompt or "", **kwargs}

        # Generate base image
        base_image_pil = await self.client.text_to_image(**final_parameters) # type: ignore # client is AsyncInferenceClient
        if not isinstance(base_image_pil, Image.Image):
            raise RuntimeError(f"SDXL base model did not return a PIL Image. Got: {type(base_image_pil)}")

        # Refine the image (using the synchronous client as per original code)
        # For a fully async pipeline, the refiner should also be async and awaited.
        # This requires converting PIL image to bytes for the sync client.
        img_bytes_io = BytesIO()
        base_image_pil.save(img_bytes_io, format="PNG") # Save to BytesIO in a common format
        img_to_refine_bytes = img_bytes_io.getvalue()

        try:
            # The sync client's image_to_image might be blocking.
            # To run it in an async context without blocking the event loop:
            loop = asyncio.get_event_loop()
            refined_image_pil = await loop.run_in_executor(
                None, # Default ThreadPoolExecutor
                self._sync_refiner_client.image_to_image, # type: ignore
                img_to_refine_bytes,
                prompt=text # Refiner also takes a prompt
                # Add other refiner-specific parameters if needed, e.g., denoising_strength
            )
        except Exception as e:
            logger.error(f"Error during SDXL image refinement: {e!s}")
            raise RuntimeError(f"SDXL image refinement failed: {e!s}") from e
            
        if not isinstance(refined_image_pil, Image.Image):
            raise RuntimeError(f"SDXL refiner did not return a PIL Image. Got: {type(refined_image_pil)}")

        return ImageRecord(
            image=refined_image_pil,
            model=self.model, # Should ideally be specific, e.g., "stabilityai/sdxl-base-1.0 + refiner"
            parameters=final_parameters, # Log parameters used for base generation
            prompt=text,
            negative_prompt=negative_prompt,
        )


class SDXLReplicate(TextToImageClient):
    """Client for Stability AI's Stable Diffusion XL (SDXL) model using Replicate.

    Requires Replicate API token to be configured (usually via `REPLICATE_API_TOKEN`
    environment variable).

    Attributes:
        model (str): Replicate model identifier for SDXL.
        prefix (str): File prefix defaults to "sdxl_replicate_".
        image_params (dict[str, Any]): Default parameters for the Replicate SDXL API call.
    """
    model: str = "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc"
    prefix: str = "sdxl_replicate_"
    image_params: dict[str, Any] = Field(default_factory=lambda: { # Use factory for mutable default
        "width": 768,
        "height": 768,
        "refine": "expert_ensemble_refiner",
        "apply_watermark": True, # Replicate API default might differ, this is explicit
        "num_inference_steps": 50,
    })
    # refiner attribute was present but not used; removed for clarity unless it has a purpose.

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using SDXL via Replicate's API.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            **kwargs: Additional parameters to merge with `self.image_params` for
                the Replicate API call.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.
        
        Raises:
            RuntimeError: If Replicate API call fails or returns unexpected data.
        """
        # Combine default image_params with runtime kwargs and essential prompts
        final_parameters = {
            **self.image_params,
            "prompt": text,
            "negative_prompt": negative_prompt or "", # Ensure empty string if None
            **kwargs,
        }

        try:
            # replicate.async_run returns a list of output URLs or direct data
            replicate_response = await replicate.async_run(self.model, input=final_parameters)
        except Exception as e:
            logger.error(f"Error calling Replicate API for SDXL: {e!s}")
            raise RuntimeError(f"Replicate API call failed for SDXL: {e!s}") from e

        if not replicate_response or not isinstance(replicate_response, list) or not replicate_response[0]:
            raise RuntimeError(f"Replicate API for SDXL returned an unexpected response: {replicate_response}")
        
        output_image_url = replicate_response[0] # Assuming the first item is the image URL

        # Fetch the image from the URL
        timeout = aiohttp.ClientTimeout(total=300.0) # Timeout for fetching image
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(output_image_url) as image_response:
                image_response.raise_for_status() # Ensure download was successful
                image_data = await image_response.read()
        
        pil_image = Image.open(BytesIO(image_data))

        return ImageRecord(
            image=pil_image,
            model=self.model, # Replicate model identifier
            parameters=final_parameters, # Log parameters sent to Replicate
            prompt=text,
            negative_prompt=negative_prompt,
            uri=output_image_url, # Store the Replicate output URL as URI
        )


class SD(TextToImageClient):
    """Client for Stability AI's Stable Diffusion 2.1 model using Replicate.

    Requires Replicate API token to be configured.

    Attributes:
        model (str): Replicate model identifier for Stable Diffusion 2.1.
        prefix (str): File prefix defaults to "sd21_".
        image_params (dict[str, Any]): Default parameters for the Replicate API call.
    """
    model: str = "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4"
    prefix: str = "sd21_"
    image_params: dict[str, Any] = Field(default_factory=lambda: { # Use factory
        "width": 1024, # Default SD 2.1 dimensions
        "height": 1024,
        "scheduler": "DPMSolverMultistep", # Common scheduler
        "num_outputs": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
    })

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "",
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using Stable Diffusion 2.1 via Replicate's API.

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional negative prompt.
            **kwargs: Additional parameters to merge with `self.image_params`.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.
        
        Raises:
            RuntimeError: If Replicate API call fails or returns unexpected data.
        """
        final_parameters = {
            **self.image_params,
            "prompt": text,
            "negative_prompt": negative_prompt or "",
            **kwargs,
        }

        try:
            replicate_output = await replicate.async_run(self.model, input=final_parameters)
        except Exception as e:
            logger.error(f"Error calling Replicate API for SD 2.1: {e!s}")
            raise RuntimeError(f"Replicate API call failed for SD 2.1: {e!s}") from e

        if not replicate_output or not isinstance(replicate_output, list) or not replicate_output[0]:
            raise RuntimeError(f"Replicate API for SD 2.1 returned an unexpected response: {replicate_output}")

        output_image_url = replicate_output[0]

        timeout = aiohttp.ClientTimeout(total=300.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(output_image_url) as image_response:
                image_response.raise_for_status()
                image_data = await image_response.read()
        
        pil_image = Image.open(BytesIO(image_data))

        return ImageRecord(
            image=pil_image,
            model=self.model,
            parameters=final_parameters,
            prompt=text,
            negative_prompt=negative_prompt,
            uri=output_image_url,
        )


class DALLE(TextToImageClient):
    """Client for OpenAI's DALL-E 3 model.

    Uses the `openai` Python library. Requires `OPENAI_API_KEY` in `bm.credentials`.

    Attributes:
        model (str): Defaults to "dall-e-3".
        prefix (str): File prefix defaults to "dalle3_".
    """
    model: str = "dall-e-3"
    prefix: str = "dalle3_"
    # Client will be initialized in generate_image if None

    async def generate_image(
        self,
        text: str,
        negative_prompt: str | None = "", # Note: DALL-E 3 doesn't directly support negative_prompt via API
        size: str = "1024x1024", # Supported DALL-E 3 sizes
        style: str = "natural", # "natural" or "vivid"
        quality: str = "standard", # "standard" or "hd"
        **kwargs: Any,
    ) -> ImageRecord:
        """Generates an image using OpenAI's DALL-E 3 model.

        Note: DALL-E 3 API does not have a direct `negative_prompt` parameter.
        If `negative_prompt` is provided, it's prepended to the main `text` prompt
        with instructions like "DO NOT INCLUDE: ...".

        Args:
            text: The prompt for image generation.
            negative_prompt: Optional. Text describing elements to avoid.
            size: Image dimensions (e.g., "1024x1024", "1792x1024", "1024x1792").
            style: The style of the generated images ("natural" or "vivid").
            quality: The quality of the image to generate ("standard" or "hd").
            **kwargs: Additional parameters for the OpenAI images API.

        Returns:
            ImageRecord: An `ImageRecord` with the generated image and metadata.
        
        Raises:
            KeyError: If `OPENAI_API_KEY` is not in `bm.credentials`.
            RuntimeError: If the OpenAI API call fails or returns unexpected data.
        """
        if self.client is None or not isinstance(self.client, AsyncOpenAI):
            openai_api_key = bm.credentials.get("OPENAI_API_KEY")
            if not openai_api_key:
                raise KeyError("OPENAI_API_KEY not found in bm.credentials for DALL-E client.")
            self.client = AsyncOpenAI(
                api_key=openai_api_key,
                timeout=600.0, # httpx.Timeout
            )
        
        prompt_for_api = text
        if negative_prompt: # Simulate negative prompt by instruction
            prompt_for_api = f"{text} \n\nIMPORTANT: DO NOT INCLUDE the following elements: {negative_prompt}"

        # Consolidate all parameters for the API call
        api_parameters = {
            "model": self.model,
            "prompt": prompt_for_api,
            "size": size,
            "style": style,
            "quality": quality,
            "n": 1, # DALL-E 3 currently supports n=1
            "response_format": "url", # Get a URL to download the image
            **kwargs, # Allow other valid DALL-E parameters
        }

        try:
            response = await self.client.images.generate(**api_parameters) # type: ignore # client is AsyncOpenAI
        except Exception as e:
            logger.error(f"Error calling OpenAI DALL-E 3 API: {e!s}")
            raise RuntimeError(f"OpenAI DALL-E 3 API call failed: {e!s}") from e

        if not response.data or not response.data[0].url:
            raise RuntimeError(f"OpenAI DALL-E 3 API returned no image data or URL. Response: {response}")

        output_image_url = response.data[0].url
        revised_prompt_from_api = response.data[0].revised_prompt
        
        # Store the actual prompt sent and any revised prompt
        api_parameters["prompt_sent_to_api"] = prompt_for_api # Store the exact prompt sent
        if revised_prompt_from_api:
            api_parameters["revised_prompt_by_api"] = revised_prompt_from_api

        # Fetch the image from the URL provided by OpenAI
        async with httpx.AsyncClient() as http_client: # Renamed to avoid conflict
            image_http_response = await http_client.get(output_image_url, timeout=300.0)
        image_http_response.raise_for_status() # Ensure download was successful
        
        pil_image = Image.open(BytesIO(image_http_response.content))

        return ImageRecord(
            image=pil_image,
            model=self.model,
            parameters=api_parameters, # Log all parameters used, including revisions
            prompt=text, # Original user prompt
            negative_prompt=negative_prompt, # Original user negative prompt
            enhanced_prompt=revised_prompt_from_api, # DALL-E often revises prompts
            uri=output_image_url, # Store the temporary OpenAI URL
        )


ImageClients: list[Type[TextToImageClient]] = [DALLE, SD35Large, Imagegen3, Imagegen3Fast, SD3, SDXL, SDXLReplicate]
"""A list of available `TextToImageClient` classes that can be used by `BatchImageGenerator`.
This list allows for easy iteration or selection of different image generation models.
"""


class BatchImageGenerator(BaseModel):
    """Generates images from multiple text prompts using a variety of models asynchronously.

    This class orchestrates image generation across several configured `TextToImageClient`
    instances. It takes a sequence of prompts, distributes them among the available
    generator clients, and runs the generation tasks concurrently. Results (as
    `ImageRecord` objects) are yielded as they complete. A summary of all generated
    images and their metadata is saved to a JSON file.

    Attributes:
        generators (Sequence[Type[TextToImageClient]]): A sequence of `TextToImageClient`
            classes to be used for generation. Defaults to `ImageClients` list.
        save_path (CloudPath | Path): The base directory path (local or cloud)
            where generated images and the summary JSON file will be saved.
            A unique subdirectory will be created under this path for each batch run.
            Defaults to `bm.save_dir` if not provided.
        _clients (dict[str, TextToImageClient]): Private dictionary to store
            instantiated client objects, keyed by client class name.
        _tasks (list[asyncio.Task[ImageRecord]]): Private list to hold asyncio tasks for
            concurrent image generation. Corrected type hint for task list.
    """

    generators: Sequence[Type[TextToImageClient]] = Field( # Type hint for list of classes
        default_factory=lambda: ImageClients, # Use factory for mutable default
        description="A sequence of TextToImageClient classes to use for generation.",
    )
    save_path: CloudPath | Path | None = Field( # Allow None initially, will default in validator
        default=None,
        description="Base path to save generated images. Defaults to current Buttermilk run's save directory.",
    )
    _clients: dict[str, TextToImageClient] = PrivateAttr(default_factory=dict)
    _tasks: list[asyncio.Task[ImageRecord]] = PrivateAttr(default_factory=list) # More specific task type

    @field_validator("save_path", mode="before")
    @classmethod
    def _validate_and_default_save_path( # Renamed for clarity
        cls,
        v: str | CloudPath | Path | None, # Allow None as input
    ) -> CloudPath | Path:
        """Validates `save_path` and defaults to `bm.save_dir` if not provided.

        Args:
            v: The input value for `save_path`.

        Returns:
            CloudPath | Path: The validated and resolved save path.
        
        Raises:
            ValueError: If `v` is not a string, Path, or CloudPath.
            RuntimeError: If `bm.save_dir` is not available when `v` is None.
        """
        if v is None:
            if bm.save_dir:
                return CloudPath(bm.save_dir) if isinstance(bm.save_dir, str) and bm.save_dir.startswith(("gs://", "s3://", "az://")) else Path(bm.save_dir) # type: ignore
            else:
                # Fallback to a temporary directory if bm.save_dir is also None
                # This ensures save_path is always set.
                temp_dir = Path(mkdtemp(prefix="buttermilk_imagegen_batch_"))
                logger.warning(f"No save_path provided and bm.save_dir not set. Defaulting to temporary directory: {temp_dir}")
                return temp_dir
        if isinstance(v, str):
            return CloudPath(v) if v.startswith(("gs://", "s3://", "az://")) else Path(v)
        if isinstance(v, (CloudPath, Path)):
            return v
        raise ValueError(
            f"save_path must be a string, Path, or CloudPath, got {type(v)}",
        )

    def init_clients(self) -> None:
        """Initializes instances of the `TextToImageClient` classes specified in `self.generators`.

        Populates `self._clients` with instantiated clients, keyed by their class name.
        """
        self._clients = {gen_class.__name__: gen_class() for gen_class in self.generators}
        logger.info(f"Initialized image generator clients: {list(self._clients.keys())}")

    async def abatch(
        self, 
        inputs: Sequence[str | dict[str, str]], # Changed 'input' to 'inputs' to avoid builtin clash
        n: int = 1
    ) -> AsyncGenerator[ImageRecord, None]:
        """Generates images asynchronously from a sequence of prompts using multiple models.

        For each prompt in `inputs`, this method schedules `n` image generation tasks
        for each configured client in `self.generators`. Images are saved to a unique
        subdirectory within `self.save_path`. A summary JSON (`info.json`) containing
        metadata of all generated images is also saved in this subdirectory.

        Args:
            inputs: A sequence of prompts. Each prompt can be a string (the text
                prompt) or a dictionary containing "text" (prompt), optional "id"
                (for tracking), and optional "negative_prompt".
            n: The number of images to generate for each prompt per model. Defaults to 1.

        Yields:
            ImageRecord: An `ImageRecord` for each successfully generated image as
            it becomes available.

        Raises:
            ValueError: If a prompt in `inputs` has an invalid type.
        """
        if not self._clients: # Ensure clients are initialized
            self.init_clients()
        if not self._clients:
            logger.error("No image generator clients initialized for batch generation. Aborting.")
            return

        # Create a unique subdirectory for this batch run
        # Ensure self.save_path is not None (handled by validator now)
        batch_save_path = self.save_path / f"batch_{ShortUUID().uuid()[:8]}" # type: ignore
        batch_save_path.mkdir(parents=True, exist_ok=True)
        info_json_path = batch_save_path / "info.json"
        
        generated_records_summary: list[dict[str, Any]] = []
        self._tasks = [] # Reset tasks for this batch

        # Prepare prompts list with IDs
        prompts_to_process: list[dict[str, Any]] = []
        for idx, prompt_item in enumerate(inputs):
            if isinstance(prompt_item, str):
                prompts_to_process.append({"text": prompt_item, "id": str(idx)})
            elif isinstance(prompt_item, dict):
                if "text" not in prompt_item:
                    logger.warning(f"Prompt dictionary at index {idx} is missing 'text' key. Skipping.")
                    continue
                prompts_to_process.append({
                    "text": prompt_item["text"],
                    "id": str(prompt_item.get("id", idx)), # Ensure ID is string
                    "negative_prompt": prompt_item.get("negative_prompt", ""), # Default to empty
                })
            else:
                raise ValueError(f"Invalid prompt type at index {idx}: expected str or dict, got {type(prompt_item)}.")

        if not prompts_to_process:
            logger.warning("No valid prompts to process in abatch.")
            return

        random.shuffle(prompts_to_process) # Shuffle to distribute load potentially

        # Schedule all generation tasks
        for current_prompt_info in prompts_to_process:
            prompt_id = current_prompt_info["id"]
            for i_run in range(n): # Number of generations per prompt
                for client_name, client_instance in self._clients.items():
                    try:
                        # Construct a unique save path for each image
                        img_filename = f"{prompt_id}_{client_instance.prefix}{i_run}_{ShortUUID().uuid()[:6]}.png"
                        img_full_save_path = (batch_save_path / img_filename).as_posix() # Use as_posix for string path

                        task_params = {
                            "text": current_prompt_info["text"],
                            "negative_prompt": current_prompt_info.get("negative_prompt", ""),
                            "save_path": img_full_save_path,
                            # Add any other specific params from current_prompt_info if needed
                        }
                        self._tasks.append(client_instance.generate(**task_params))
                    except Exception as e:
                        logger.error(
                            f"Error adding task: generate image from {client_name}, run {i_run}, prompt ID {prompt_id}. Error: {e!s}",
                            exc_info=True
                        )
                        continue # Skip this specific task

        # Yield results as they complete
        for task in asyncio.as_completed(self._tasks):
            try:
                # await asyncio.sleep(0.01) # Small sleep to allow other tasks, might not be necessary
                image_result: ImageRecord = await task
                yield image_result # Yield the successful ImageRecord
                
                # Collect summary information for info.json
                if image_result and image_result.image and image_result.uri: # Successfully generated and saved
                    generated_records_summary.append({
                        "prompt_id": image_result.parameters.get("id", "unknown_id_in_params"), # Try to get ID from params if set
                        "original_prompt": image_result.prompt,
                        "negative_prompt": image_result.negative_prompt,
                        "model_used": image_result.model,
                        "image_uri": image_result.uri,
                        "generation_parameters": image_result.parameters,
                        "error": image_result.error, # Will be None if successful
                    })
                elif image_result and image_result.error: # Generation failed, record error
                     generated_records_summary.append({
                        "prompt_id": image_result.parameters.get("id", "unknown_id_in_params"),
                        "original_prompt": image_result.prompt,
                        "negative_prompt": image_result.negative_prompt,
                        "model_used": image_result.model,
                        "image_uri": None,
                        "generation_parameters": image_result.parameters,
                        "error": image_result.error,
                    })

            except Exception as e: # Catch errors from await task itself
                logger.error(f"Error collecting result from an image generation task: {e!s}", exc_info=True)
                # Optionally add an error entry to summary here if task details can be inferred
                continue

        # Save summary JSON
        try:
            with open(info_json_path, "w", encoding="utf-8") as f:
                json.dump(generated_records_summary, f, indent=2)
            logger.info(
                f"Batch image generation complete. Generated {len(generated_records_summary)} image results/errors "
                f"from {len(prompts_to_process)} prompts (x{n} runs each) across {len(self._clients)} models. "
                f"Summary saved to {info_json_path.as_posix()}"
            )
        except Exception as e:
            logger.error(f"Failed to save batch image generation summary to {info_json_path}: {e!s}")

        self._tasks.clear() # Clear tasks for next batch run
