import base64
from io import BytesIO
from typing import Literal

from PIL import Image
from google import genai
from google.genai import types
from openai import OpenAI

from img_to_dxf_ai.reve_client import edit_image_with_reve


def resize_img(image: Image.Image, target_w: int) -> Image.Image:
    w, h = image.size
    target_h = int(round(h * (target_w / float(w))))
    resized_image = image.resize((target_w, target_h), Image.LANCZOS)
    return resized_image


# _prompt = """画像は2Dの機械図面です。正面図や側面図といった方向に投影された面図が描かれています。
# **この機械図面の線の位置や形状は維持し、線の色のみを以下のように色分けしてください。**
# - 赤(#ff0000): 寸法線(寸法線の引き出し線、矢印、寸法の文字を含む)
# - 黒(#000000): 破線、点線、一点鎖線、二点鎖線といった補助線
# - 青(#0000ff): 寸法線、補助線を除いた外形線の中の直線
# - 緑(#00ff00): 寸法線、補助線を除いた外形線の中の円もしくは円弧
# """

_prompt = """The image is a 2D mechanical drawing. It contains projected views in various directions.
**Without changing any positions or shapes of the lines in this mechanical drawing, only recolor the lines as follows:**

* Red (#ff0000): Dimension lines (including extension lines, arrows, and dimension text)
* Black (#000000): Auxiliary lines such as dashed, dotted, chain, or double-chain lines
* Blue (#0000ff): Straight outer contour lines, excluding dimension and auxiliary lines
* Green (#00ff00): Circular or arc-shaped outer contour lines, excluding dimension and auxiliary lines
"""


def colorize_img(input_img: Image.Image, backend: Literal["openai", "google", "reve"] = "google") -> Image.Image:
    if backend == "openai":
        client = OpenAI()
        img_bytes = BytesIO()
        width, _ = input_img.size
        input_img.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        result = client.images.edit(
            model="gpt-image-1",
            image=("image.png", img_bytes, "image/png"),
            prompt=_prompt,
            input_fidelity="high",
            quality="high",
            output_format="png"
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_bytes))
        image = resize_img(image, width)
        return image
    elif backend == "google":
        buffer = BytesIO()
        width, _ = input_img.size
        input_img.save(buffer, format="PNG")
        buffer.seek(0)
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview",
            contents=[
                types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
                types.Part.from_text(text=_prompt),
            ],
        )
        out_part = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                out_part = part
                break
        if out_part is None:
            raise ValueError("no image in model response")
        image = Image.open(BytesIO(out_part.inline_data.data))
        image = resize_img(image, width)
        return image
    elif backend == "reve":
        return edit_image_with_reve(_prompt, input_img)
    else:
        raise ValueError(f"Invalid backend: {backend}")

