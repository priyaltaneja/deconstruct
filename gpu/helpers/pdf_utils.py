"""
PDF Processing Utilities
Handles PDF to image conversion with caching to prevent duplicate conversions.
"""

from typing import List, Dict, Optional
from io import BytesIO
import base64


class PDFContext:
    """
    Context manager for PDF processing that caches image conversions.
    Prevents duplicate PDF-to-image conversions which are expensive.
    """

    def __init__(self, pdf_bytes: bytes):
        self._pdf_bytes = pdf_bytes
        self._images_cache: Dict[int, List] = {}
        self._page_count: Optional[int] = None

    @property
    def pdf_bytes(self) -> bytes:
        return self._pdf_bytes

    @property
    def page_count(self) -> int:
        """Get page count, converting only if needed."""
        if self._page_count is None:
            from pdf2image import convert_from_bytes
            images = self.get_images(dpi=72)  # Low DPI for counting
            self._page_count = len(images)
        return self._page_count

    def get_images(self, dpi: int = 200, first_page: int = 1, last_page: Optional[int] = None) -> List:
        """
        Get images from PDF, caching by DPI.

        Args:
            dpi: Resolution for conversion
            first_page: First page to convert (1-indexed)
            last_page: Last page to convert (inclusive), None for all pages

        Returns:
            List of PIL Images
        """
        from pdf2image import convert_from_bytes

        cache_key = (dpi, first_page, last_page)

        # Simple DPI-based caching (full page range only)
        if dpi in self._images_cache and first_page == 1 and last_page is None:
            return self._images_cache[dpi]

        kwargs = {"dpi": dpi}
        if last_page is not None:
            kwargs["first_page"] = first_page
            kwargs["last_page"] = last_page

        images = convert_from_bytes(self._pdf_bytes, **kwargs)

        # Cache full conversions
        if first_page == 1 and last_page is None:
            self._images_cache[dpi] = images

        return images

    def get_images_as_base64(self, dpi: int = 300, max_pages: int = 10, format: str = "PNG") -> List[Dict]:
        """
        Get images as base64-encoded content blocks for LLM APIs.

        Args:
            dpi: Resolution for conversion
            max_pages: Maximum number of pages to include
            format: Image format (PNG, JPEG)

        Returns:
            List of content blocks suitable for vision LLM APIs
        """
        images = self.get_images(dpi=dpi)

        image_content = []
        for img in images[:max_pages]:
            buffered = BytesIO()
            img.save(buffered, format=format)
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": f"image/{format.lower()}",
                    "data": img_b64,
                },
            })

        return image_content


def convert_pdf_to_images(pdf_bytes: bytes, dpi: int = 200, first_page: int = 1, last_page: Optional[int] = None) -> List:
    """
    Convert PDF bytes to list of PIL Images.

    For repeated access to the same PDF, use PDFContext instead.

    Args:
        pdf_bytes: PDF file content
        dpi: Resolution for conversion
        first_page: First page to convert (1-indexed)
        last_page: Last page to convert (inclusive), None for all pages

    Returns:
        List of PIL Images
    """
    from pdf2image import convert_from_bytes

    kwargs = {"dpi": dpi}
    if last_page is not None:
        kwargs["first_page"] = first_page
        kwargs["last_page"] = last_page

    return convert_from_bytes(pdf_bytes, **kwargs)


def get_page_count(pdf_bytes: bytes) -> int:
    """
    Get the number of pages in a PDF.

    Args:
        pdf_bytes: PDF file content

    Returns:
        Number of pages
    """
    from pypdf import PdfReader
    from io import BytesIO

    reader = PdfReader(BytesIO(pdf_bytes))
    return len(reader.pages)
