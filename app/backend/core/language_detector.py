from langdetect import detect

def detect_language(text):
    """Detects the language of the given text and returns the language code.
    If the language is not supported, it returns "en-us" as default."""
    try:
        detected_lang = detect(text)
        if detected_lang=="de":
            return "de-de"
        else:
            return "en-us"
    except Exception as e:
        print(f"Language detection error: {e}")
        return "en-us"
