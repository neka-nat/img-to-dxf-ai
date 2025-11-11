import base64
import json
import os
from io import BytesIO

import requests
from PIL import Image


def edit_image_with_reve(prompt: str, image: Image.Image) -> Image.Image:
    REVE_API_KEY = os.getenv("REVE_API_KEY")
    if not REVE_API_KEY:
        raise ValueError("REVE_API_KEY is not set")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    headers = {
        "Authorization": f"Bearer {REVE_API_KEY}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    payload = {
        "edit_instruction": prompt,
        "reference_image": image_base64,
        "version": "latest"
    }

    try:
        response = requests.post("https://api.reve.com/v1/image/edit", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        result_image_base64 = result['image']
        return Image.open(BytesIO(base64.b64decode(result_image_base64)))
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse response: {e}")
