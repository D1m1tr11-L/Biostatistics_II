import json
from typing import List, Dict, Any
from openai import OpenAI

client = OpenAI()


def infer_relationships_from_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use LLM to infer A→C relationships from multiple literature summaries.
    """

    prompt = f"""
You are given multiple structured literature summaries.

Each summary may contain:
- intervention_or_exposure
- outcome

Goal:
1. Identify pairs:
   A → B
   B → C

2. If possible, infer:
   A → C

Rules:
- Use ONLY information provided
- Do NOT hallucinate
- If no valid chain, return empty list
- Be strict: B must match conceptually (not necessarily exact string)

Return JSON ONLY:

{{
  "chains": [
    {{
      "A": "...",
      "B": "...",
      "C": "...",
      "inferred_relationship": "...",
      "confidence": "high/medium/low",
      "explanation": "..."
    }}
  ]
}}

Input summaries:
{json.dumps(summaries, indent=2)}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "developer",
                "content": (
                    "You are an expert in causal inference and epidemiology. "
                    "Your job is to detect chains across studies and infer relationships."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return json.loads(response.choices[0].message.content)