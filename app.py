# app.py (À la racine)
import streamlit as st

st.set_page_config(page_title="Oracle Sports Predictor", page_icon="🔮", layout="wide")

st.title("🔮 Oracle Sports Predictor")
st.markdown("""
Bienvenue dans ton centre de commande prédictif. 
Utilise le menu latéral à gauche pour naviguer entre tes différents algorithmes :

* **🏀 NBA Oracle** : Prédictions de saison régulière (Expected Value, Modèles XGBoost).
* **⚽ MPP World Cup** : Optimisation de ligue Mon Petit Prono (Théorie des Jeux, Loi de Poisson).
""")