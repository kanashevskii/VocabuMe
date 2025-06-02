from openai import OpenAI
from decouple import config

client = OpenAI(api_key=config("OPENAI_API_KEY"))

def generate_word_data(word: str) -> dict:
    prompt = f"""
You are an English language assistant.

For the English word or phrase "{word}", return the following in JSON format:
1. "translation": its most common translation into Russian.
2. "transcription": the IPA transcription (phonetic pronunciation) of the English word, not the translation.
3. "example": a short, simple sentence in English using the word.

Example format:
{{
  "translation": "перевод",
  "transcription": "həʊˈevə", 
  "example": "However, it was too late."
}}

Only return the JSON object. No extra text.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        if content.startswith("```json"):
            content = content.strip("```json").strip("` \n")

        return eval(content)  # безопасно при своём контроле
    except Exception as e:
        print("OpenAI error:", e)
        return None
