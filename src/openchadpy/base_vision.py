"""
Base Vision Module
Abstract base class for Vision/Image Generation backends.
"""
from abc import abstractmethod
from typing import List, Optional, Union, Callable, Any, Dict
from PIL import Image
from .base_backend import BaseBackend

class BaseVision(BaseBackend):
    """
    Abstract base class for Vision backends.
    Provides image generation, video generation, and upscaling capabilities.
    """

    @abstractmethod
    def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = -1,
        batch_count: int = 1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images from text prompt.
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt for guidance
            width: Image width
            height: Image height
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed (-1 for random)
            batch_count: Number of images to generate
            progress_callback: Optional callback(step, total_steps, progress_pct)
        Returns:
            List of PIL Images
        """
        pass

    def generate_video(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 480,
        height: int = 832,
        video_frames: int = 33,
        num_inference_steps: int = 20,
        guidance_scale: float = 6.0,
        seed: int = -1,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video frames from text prompt.
        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt
            width: Frame width
            height: Frame height
            video_frames: Number of frames
            num_inference_steps: Denoising steps
            guidance_scale: CFG scale
            seed: Random seed
            progress_callback: Progress callback
        Returns:
            List of PIL Images (video frames)
        """
        raise NotImplementedError("Subclass may implement generate_video")
    
    def upscale(
        self,
        image: Union[str, Image.Image],
        upscale_factor: int = 2,
        **kwargs
    ) -> Image.Image:
        """
        Upscale an image.
        Args:
            image: Input image path or PIL Image
            upscale_factor: Upscaling factor
        Returns:
            Upscaled PIL Image
        """
        raise NotImplementedError("Subclass may implement upscale")
