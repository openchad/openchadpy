from typing import Optional
from fastapi.responses import Response
from PIL import Image
import io
import os

def process_image(
    file_path: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    quality: int = 85,
    format: Optional[str] = None,
):
    """
    Process and resize images
    Args:
        file_path: Absolute path to the image file
        width: Target width (maintains aspect ratio if height not provided)
        height: Target height (maintains aspect ratio if width not provided)
        quality: JPEG quality (1-100)
        format: Output format (jpeg, png, webp, etc.)
    """
    try:
        # Open the image
        img = Image.open(file_path)
        # Convert RGBA to RGB if saving as JPEG
        output_format = format.upper() if format else img.format or "PNG"
        if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            # Create white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = background
        # Resize if dimensions provided
        if width or height:
            original_width, original_height = img.size
            # Calculate target dimensions maintaining aspect ratio
            if width and height:
                # Both provided - use thumbnail to maintain aspect ratio
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
            elif width:
                # Only width provided
                aspect_ratio = original_height / original_width
                target_height = int(width * aspect_ratio)
                img = img.resize((width, target_height), Image.Resampling.LANCZOS)
            elif height:
                # Only height provided
                aspect_ratio = original_width / original_height
                target_width = int(height * aspect_ratio)
                img = img.resize((target_width, height), Image.Resampling.LANCZOS)
        # Save to bytes buffer
        img_byte_arr = io.BytesIO()
        # Save with appropriate parameters
        save_kwargs = {}
        if output_format in ("JPEG", "JPG"):
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif output_format == "PNG":
            save_kwargs["optimize"] = True
        elif output_format == "WEBP":
            save_kwargs["quality"] = quality
        img.save(img_byte_arr, format=output_format, **save_kwargs)
        img_byte_arr.seek(0)
        # Determine MIME type
        mime_types = {
            "JPEG": "image/jpeg",
            "JPG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
            "BMP": "image/bmp",
            "TIFF": "image/tiff",
        }
        media_type = mime_types.get(output_format, "image/png")
        return Response(
            content=img_byte_arr.getvalue(),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")
