from io import BytesIO

from PIL import Image


def _try_rapidocr(pil_image: Image.Image) -> str | None:
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return None

    engine = RapidOCR()
    result, _ = engine(pil_image)
    if not result:
        return ""
    return "\n".join(item[1] for item in result if len(item) > 1 and item[1])


def _try_pytesseract(pil_image: Image.Image) -> str | None:
    try:
        import pytesseract
    except ImportError:
        return None

    return pytesseract.image_to_string(pil_image, lang="chi_sim+eng")


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    for extractor in (_try_rapidocr, _try_pytesseract):
        text = extractor(pil_image)
        if text is not None:
            return text.strip()

    raise RuntimeError(
        "No OCR backend is installed. Install `rapidocr_onnxruntime` or `pytesseract` "
        "to enable OCR-RAG."
    )
