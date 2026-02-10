import aiohttp

COINGECKO_URL = "https://api.coingecko.com/api/v3/simple/price"


async def get_bitcoin_price():
    params = {
        "ids": "bitcoin",
        "vs_currencies": "usd,eur"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                COINGECKO_URL,
                params=params,
                timeout=15
            ) as resp:
                data = await resp.json()

        btc = data.get("bitcoin", {})
        usd = btc.get("usd")
        eur = btc.get("eur")

        if usd:
            return f"₿ Bitcoin:\nUSD: ${usd}\nEUR: €{eur}"
        else:
            return "Не удалось получить цену биткоина."
    except Exception:
        return "Ошибка получения данных о биткоине."
