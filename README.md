# Football AI Predictor MVP

Aplikacja przewiduje wynik meczu piłkarskiego w układzie 1/X/2. Bierze pod uwagę:

- formę z ostatnich 5 i 10 meczów,
- bilans bramek,
- trudność ostatnich rywali przez własne Elo,
- miejsce meczu, neutralny teren i gospodarza turnieju,
- typ turnieju, np. World Cup / friendly,
- kursy bukmacherskie, jeśli podasz je ręcznie lub pobierzesz przez API,
- aktualizację stanu drużyn po dopisaniu kolejnych wyników.

To jest MVP do nauki i rozbudowy, nie gwarancja zysku z zakładów.

## 1. Uruchomienie lokalnie

```bash
cd football_predictor_ai
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Wejdź w przeglądarce:

```text
http://127.0.0.1:8000
```

## 2. Dane, które możesz pobrać

### Najlepsze pod reprezentacje / Mundial

Pobierz `results.csv` z datasetu `martj42/international_results` albo z Kaggle `International football results from 1872 to ...`.

Wymagane kolumny:

```text
date, home_team, away_team, home_score, away_score, tournament, city, country, neutral
```

### Dobre pod ligi + kursy historyczne

Pobierz CSV z football-data.co.uk. Aplikacja rozpoznaje format:

```text
Date, HomeTeam, AwayTeam, FTHG, FTAG
```

oraz, jeśli są dostępne:

```text
B365H, B365D, B365A
```

albo średnie kursy typu `AvgH`, `AvgD`, `AvgA`.

## 3. Trening

1. Wgraj CSV w panelu.
2. Kliknij `Trenuj model`.
3. Model zapisze się w `artifacts/model.joblib`, a stan drużyn w `artifacts/team_state.json`.

## 4. Predykcja

Wpisz np.:

```text
Home: Mexico
Away: South Africa
Country: Mexico
Tournament: FIFA World Cup
Neutral: false
```

Możesz dopisać kursy ręcznie, np. 1 / X / 2. Aplikacja policzy:

- prawdopodobieństwa modelu,
- “fair odds”, czyli kurs uczciwy według modelu,
- potencjalne value, jeśli kurs bukmachera jest wyraźnie wyższy niż modelowy fair odds.

## 5. Kursy live / pre-match z internetu

Najprostsza opcja: The Odds API.

1. Załóż konto i weź klucz API.
2. W `.env` ustaw:

```text
ODDS_API_KEY=twoj_klucz
ODDS_REGIONS=eu,uk
```

3. Uruchom aplikację ponownie.
4. W panelu użyj `Pokaż soccer sports`, żeby znaleźć poprawny `sport_key`.
5. Potem kliknij `Pobierz kursy dla meczu`.

Uwaga: live odds zwykle są limitowane albo płatne. Nie polecam scrapować stron bukmacherów, bo to często łamie regulaminy i szybko się psuje.

## 6. Aktualizacja po meczu

Po zakończeniu meczu wpisz wynik i kliknij `Zapisz wynik`. Aplikacja:

- doda mecz do `data/matches.csv`,
- przeliczy stan formy i Elo,
- przy kolejnej predykcji będzie już znała nowy mecz.

Model nie jest automatycznie trenowany po każdym jednym meczu, bo to zwykle nie ma sensu. Lepiej trenować np. po większej paczce nowych danych.

## 7. Co warto rozbudować jako następne

- baza SQLite/PostgreSQL zamiast CSV,
- harmonogram pobierania wyników i kursów,
- backtesting strategii zakładów,
- kalibracja prawdopodobieństw i tracking closing-line value,
- osobny model goli, np. Poisson / Dixon-Coles,
- dane o składach, kontuzjach, xG, kartkach, podróżach i pogodzie,
- logowanie predykcji przed meczem, żeby nie oszukiwać samego siebie po fakcie.

## 8. API

Dokumentacja FastAPI:

```text
http://127.0.0.1:8000/docs
```

Najważniejsze endpointy:

- `POST /upload` – wgranie CSV,
- `POST /train` – trening modelu,
- `POST /predict` – predykcja meczu,
- `POST /result` – dodanie wyniku po meczu,
- `GET /team/{team}/form` – ostatnie mecze drużyny,
- `GET /odds/sports` – lista sportów z The Odds API,
- `GET /odds/match` – kursy na konkretny mecz.
