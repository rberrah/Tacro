import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mrgsolve import mread
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn

# Charger votre modèle mrgsolve
code_fusion = """
[PROB]
// Modèle fusionné : tacrolimus Prograf/Advagraf vs Envarsus
[PARAM] @annotated
TVCL: 21.2 : CL tacro (L/h)
TVV1: 486 : V1 tacro (L)
TVQ: 79 : Q tacro (L/h)
TVV2: 271 : V2 tacro (L)
TVKTR: 3.34 : KTR tacro (1/h)
// Insérez tout le modèle que vous avez fourni ici...
[CAPTURE]
DV CL V1 Q V2 KTR ALAG
"""

mod = mread("fusion_model", code=code_fusion)

# Entraînement du modèle Machine Learning sur les résidus
class ResidualMLModel(nn.Module):
    def __init__(self):
        super(ResidualMLModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Fonction pour simuler le modèle mrgsolve
def simulate_mrgsolve(mod, params, times):
    data = pd.DataFrame({
        'ID': [1],
        'time': times,
        **params
    })
    sim = mod % data
    return pd.DataFrame(sim)

def main():
    st.title("Tacrolimus Simulation et Machine Learning")
    st.sidebar.header("Paramètres de Simulation")

    # Paramètres d'entrée utilisateur
    HT = st.sidebar.slider("Hématocrite (%)", 25, 50, 35)
    CYP = st.sidebar.selectbox("CYP Expressor", [0, 1])
    ST = st.sidebar.selectbox("Type de traitement", [0, 1, 2])
    VOMIT = st.sidebar.slider("Heure du vomissement (h)", 0, 240, 120)
    PV = st.sidebar.slider("Pourcentage vomi", 0.0, 1.0, 0.5)
    time = st.sidebar.slider("Temps (h)", 0, 48, 24)

    params = {'HT': HT, 'CYP': CYP, 'ST': ST, 'VOMIT': VOMIT, 'PV': PV}

    st.write("Simulation mrgsolve en cours...")
    times = list(range(0, int(time)))
    sim_results = simulate_mrgsolve(mod, params, times)
    st.write("Simulation terminée ! Voici les résultats :")
    st.dataframe(sim_results)

    # Création et entraînement du modèle ML
    st.write("Entraînement du modèle Machine Learning pour les résidus...")
    X_train = np.random.rand(100, 5)  # Exemple fictif
    y_train = np.random.rand(100)    # Résidus fictifs

    ml_model = ResidualMLModel()
    optimizer = torch.optim.Adam(ml_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for epoch in range(100):  # Entraînement simplifié
        optimizer.zero_grad()
        predictions = ml_model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    st.write(f"Perte finale du modèle ML après entraînement : {loss.item():.4f}")

    # Visualisation des résultats
    st.subheader("Visualisation des résultats")
    fig, ax = plt.subplots()
    ax.plot(sim_results["time"], sim_results["CENT"], label="Simulation mrgsolve")
    ax.scatter([time], [loss.item()], color='red', label="Résidu ML")
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("Concentration")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
