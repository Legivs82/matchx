import streamlit as st
import pandas as pd
import joblib
import requests

# Carica il modello
model = joblib.load('modello_calcio.pkl')

# ----------------------------------------------------------------------------------
# Interfaccia Streamlit
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Previsione Calcio", layout="wide")
st.title("📊 Predizioni Scommesse Calcio")
st.markdown("Inserisci i dati per ottenere una previsione:")

# Input manuale
with st.expander("📝 Inserimento Manuale"):
    col1, col2 = st.columns(2)
    with col1:
        home_avg = st.number_input("Media gol ultime 5 partite in casa", min_value=0.0, format="%.2f")
        quote_h = st.number_input("Quote vittoria casa", min_value=1.0, format="%.2f")
    with col2:
        away_avg = st.number_input("Media gol ultime 5 partite in trasferta", min_value=0.0, format="%.2f")
        quote_d = st.number_input("Quote pareggio", min_value=1.0, format="%.2f")
        quote_a = st.number_input("Quote vittoria trasferta", min_value=1.0, format="%.2f")

# Previsione
if st.button("🔮 Calcola Previsione"):
    input_data = [[home_avg, away_avg, quote_h, quote_d, quote_a]]
    prediction = model.predict(input_data)[0]
    probabilità = model.predict_proba(input_data)[0]
    
    risultati = ["Vittoria Casa", "Pareggio", "Vittoria Trasferta"]
    st.subheader(f"**Risultato previsto:** {risultati[prediction]}")
    
    # Mostra probabilità
    st.write("Probabilità:")
    probs_df = pd.DataFrame({
        'Esito': risultati,
        'Probabilità (%)': probabilità * 100
    })
    st.bar_chart(probs_df.set_index('Esito'))

# ----------------------------------------------------------------------------------
# Previsioni Live (Opzionale)
# ----------------------------------------------------------------------------------
if st.checkbox("🔴 Mostra Previsioni Live da API"):
    API_KEY = 'db67e843648ab6266f01df627ee10fe6'
    url = f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds/?apiKey={API_KEY}&regions=eu'
    
    try:
        data = requests.get(url).json()
        for game in data:
            home = game['home_team']
            away = game['away_team']
            quote_h = min([bookmaker['markets'][0]['outcomes'][0]['price'] for bookmaker in game['bookmakers']])
            quote_d = min([bookmaker['markets'][0]['outcomes'][1]['price'] for bookmaker in game['bookmakers']])
            quote_a = min([bookmaker['markets'][0]['outcomes'][2]['price'] for bookmaker in game['bookmakers']])
            
            # Usa le medie storiche (da implementare)
            input_live = [[1.5, 1.2, quote_h, quote_d, quote_a]]  # Sostituisci con dati reali
            prediction = model.predict(input_live)[0]
            
            st.write(f"**{home} vs {away}**")
            st.write(f"Previsione: {risultati[prediction]}")
            
    except Exception as e:
        st.error(f"Errore nel recupero dati live: {e}")

st.markdown("---")
st.info("⚠️ Nota: Le previsioni non sono garantite. Usa con responsabilità.")
