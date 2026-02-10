CRYPTO_KEYWORDS = [
    "биткоин", "bitcoin", "btc",
    "курс биткоина", "цена биткоина"
]


def is_crypto_query(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in CRYPTO_KEYWORDS)


def must_use_internet(text: str) -> bool:
    if is_crypto_query(text):
        return True
    return False
