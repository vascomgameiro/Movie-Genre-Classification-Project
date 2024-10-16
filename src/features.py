import nltk

REGION_MAP =     {
        "American": "Western",
        "British": "Western",
        "Canadian": "Western",
        "Australian": "Western",
        "Bollywood": "South Asian",
        "Telugu": "South Asian",
        "Tamil": "South Asian",
        "Malayalam": "South Asian",
        "Bengali": "South Asian",
        "Kannada": "South Asian",
        "Marathi": "South Asian",
        "Punjabi": "South Asian",
        "Assamese": "South Asian",
        "Chinese": "East Asian",
        "Japanese": "East Asian",
        "South_Korean": "East Asian",
        "Hong Kong": "East Asian",
        "Filipino": "Southeast Asian",
        "Bangladeshi": "South Asian",
        "Russian": "European",
        "Turkish": "Middle Eastern",
        "Egyptian": "Middle Eastern",
        "Malaysian": "Southeast Asian",
    }

def select_tokens(text, selected_tokens, tokenizer=nltk.word_tokenize):
    """
    Cleans a single document by keeping only the tokens present in the selected_tokens set.

    Parameters:
    text (str): The text document to clean.
    selected_tokens (set or list): The set or list of tokens to retain in the text.
    tokenizer (function): A function to tokenize the text (defaults to nltk.word_tokenize).

    Returns:
    str: The cleaned text with only the selected tokens.
    """

    if isinstance(text, str):
        tokens = tokenizer(text)
        filtered_tokens = [token for token in tokens if token in selected_tokens]
        return " ".join(filtered_tokens)
    return text