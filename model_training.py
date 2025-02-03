import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import joblib

# ----------------------------------------------------------------------------------
# 1. Raccolta Dati Storici (Football-Data.co.uk)
# ----------------------------------------------------------------------------------
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

# Pulizia dati
df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'AvgH', 'AvgD', 'AvgA']]
df.columns = ['Date', 'Home', 'Away', 'HG', 'AG', 'Result', 'Quote_H', 'Quote_D', 'Quote_A']

# Calcola media gol ultime 5 partite
df['Home_Avg'] = df.groupby('Home')['HG'].transform(lambda x: x.rolling(5).mean())
df['Away_Avg'] = df.groupby('Away')['AG'].transform(lambda x: x.rolling(5).mean())

# Rimuovi righe con dati mancanti
df.dropna(inplace=True)

# ----------------------------------------------------------------------------------
# 2. Raccolta Quote in Tempo Reale (The Odds API)
# ----------------------------------------------------------------------------------
API_KEY = 'db67e843648ab6266f01df627ee10fe6'  # Registrati su https://the-odds-api.com
sport = 'soccer_epl'
url = f'https://api.the-odds-api.com/v4/sports/{sport}/odds/?apiKey={API_KEY}&regions=eu'

try:
    response = requests.get(url)
    data = response.json()
    live_odds = pd.DataFrame([{
        'Home': game['home_team'],
        'Away': game['away_team'],
        'Quote_H': min([bookmaker['markets'][0]['outcomes'][0]['price'] for bookmaker in game['bookmakers']]),
        'Quote_D': min([bookmaker['markets'][0]['outcomes'][1]['price'] for bookmaker in game['bookmakers']]),
        'Quote_A': min([bookmaker['markets'][0]['outcomes'][2]['price'] for bookmaker in game['bookmakers']])
    } for game in data])
except Exception as e:
    print(f"Errore API: {e}")
    live_odds = pd.DataFrame()

# ----------------------------------------------------------------------------------
# 3. Preparazione Dati per il Modello
# ----------------------------------------------------------------------------------
# Converti risultato in numeri (H=0, D=1, A=2)
df['Target'] = df['Result'].map({'H': 0, 'D': 1, 'A': 2})

# Features: Media gol + quote
X = df[['Home_Avg', 'Away_Avg', 'Quote_H', 'Quote_D', 'Quote_A']]
y = df['Target']

# Split dati
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------------------------------
# 4. Addestramento Modello
# ----------------------------------------------------------------------------------
model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train, y_train)

# Valutazione
y_pred = model.predict(X_test)
print(f"Accuratezza modello: {accuracy_score(y_test, y_pred)*100:.2f}%")

# Salva il modello
joblib.dump(model, 'modello_calcio.pkl')

# ----------------------------------------------------------------------------------
# 5. Ottimizzazione (Opzionale)
# ----------------------------------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X_train, y_train)
print(f"Migliori parametri: {grid_search.best_params_}")
