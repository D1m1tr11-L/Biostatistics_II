import base64
import json
from openai import OpenAI

client = OpenAI()


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def extract_cv_parameters_from_image(image_bytes: bytes):
    """
    Use LLM to read CV parameters from a KM plot image.

    Returns
    -------
    params : dict
        {
          "plot_box": {...},
          "x_axis": {...},
          "y_axis": {...},
          "groups": [
            {
              "name": "...",
              "color_label": "...",
              "lower_hsv": [H,S,V],
              "upper_hsv": [H,S,V]
            },
            ...
          ]
        }
    meta : dict
    """
    image_b64 = encode_image_to_base64(image_bytes)

    prompt = """
You are analyzing a Kaplan-Meier survival plot image for downstream computer vision extraction.

Return ONLY valid JSON.
Do not include markdown.
Do not include explanations.

Task:
1. Identify the main plotting box:
   - x_left
   - x_right
   - y_top
   - y_bottom

2. Identify x-axis range:
   - min
   - max
   - unit

3. Identify y-axis range:
   - min
   - max

4. Identify exactly up to 2 survival groups from the legend if visible.

5. For each group, estimate the dominant curve color and provide an approximate HSV threshold range for CV extraction.
   - lower_hsv should be [H,S,V]
   - upper_hsv should be [H,S,V]
   - H in [0,179], S in [0,255], V in [0,255]
   - Use broad but reasonable ranges for the visible curve color
   - Ignore confidence bands, legend lines, and text if possible

Return this JSON structure exactly:

{
  "plot_box": {
    "x_left": 0,
    "x_right": 0,
    "y_top": 0,
    "y_bottom": 0
  },
  "x_axis": {
    "min": 0,
    "max": 0,
    "unit": "months"
  },
  "y_axis": {
    "min": 0.0,
    "max": 1.0
  },
  "groups": [
    {
      "name": "Group A",
      "color_label": "blue",
      "lower_hsv": [90, 80, 50],
      "upper_hsv": [140, 255, 255]
    },
    {
      "name": "Group B",
      "color_label": "orange",
      "lower_hsv": [5, 100, 80],
      "upper_hsv": [25, 255, 255]
    }
  ]
}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        top_p=1,
        seed=12345,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": (
                    "Return valid JSON only. "
                    "Estimate plot box, axes, group names, and approximate HSV ranges for CV extraction."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                            "detail": "high"
                        }
                    }
                ]
            }
        ]
    )

    content = response.choices[0].message.content
    params = json.loads(content)

    meta = {
        "system_fingerprint": getattr(response, "system_fingerprint", None)
    }

    return params, meta