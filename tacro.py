import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import torch
import torch.nn as nn

# -- Charger mrgsolve --
from mrgsolve import mread
model_code = """
[PROB]
# Modèle fusionné : tacrolimus Prograf/Advagraf vs Envarsus

[PARAM] @annotated
TVCL        : 21.2  : CL tacro (L/h)
TVV1        : 486   : V1 tacro (L)
TVQ         : 79    : Q tacro (L/h)
TVV2        : 271   : V2 tacro (L)
TVKTR       : 3.34  : KTR tacro (1/h)
HTCL        : -1.14 : Effet HT sur CL
CYPCL_TAC   : 2.00  : Effet CYP sur CL tacro
STKTR       : 1.53  : Effet ST sur KTR
STV1        : 0.29  : Effet ST sur V1

TVCL_ENV    : 19.6  : CL Envarsus (L/h)
TVV1_ENV    : 123   : V1 Envarsus (L)
TVQ_ENV     : 74.9  : Q Envarsus (L/h)
TVV2_ENV    : 500   : V2 Envarsus (L)
TVKTR_ENV   : 0.752 : KTR Envarsus (1/h)
CYPCL_ENV   : 1.625 : Effet CYP sur CL Envarsus
LAG         : 2.29  : Lag time Envarsus (h)

PROP_TAC : 0.012 : Erreur proportionnelle tacro
ADD_TAC  : 0.50  : Erreur additive tacro
PROP_ENV : 0.208 : Erreur proportionnelle Envarsus
ADD_ENV  : 0.307 : Erreur additive Envarsus

ETA1 : 0 : ETA sur CL
ETA2 : 0 : ETA sur V1
ETA3 : 0 : ETA sur Q
ETA4 : 0 : ETA sur V2
ETA5 : 0 : ETA sur KTR

$PARAM @annotated @covariates
HT    : 35   : Hématocrite (%)
CYP   : 0    : CYP expressor (1) ou non (0)
ST    : 1    : 0 = Advagraf, 1 = Prograf, 2 = Envarsus
VOMIT : 120  : Heure du vomissement (h)
PV    : 1    : Pourcentage vomi (0 à 1)

[CMT] @annotated
DEPOT   : Dosing compartment (mg)
TRANS1  : Transit compartment 1 (mg)
TRANS2  : Transit compartment 2 (mg)
TRANS3  : Transit compartment 3 (mg)
CENT    : Central compartment (mg) [OBS]
PERI    : Peripheral compartment (mg)

[OMEGA] 0 0 0 0 0

[GLOBAL]
double CL, V1, Q, V2, KTR, ALAG;

[MAIN]
double current_time = self.time;

if (ST == 2) {
  // Envarsus
  CL  = TVCL_ENV * pow(CYPCL_ENV, CYP) * exp(ETA1 + ETA(1));
  V1  = TVV1_ENV * exp(ETA2 + ETA(2));
  Q   = TVQ_ENV * exp(ETA3 + ETA(3));
  V2  = TVV2_ENV * exp(ETA4 + ETA(4));
  KTR = TVKTR_ENV * exp(ETA5 + ETA(5));
  ALAG = LAG;
} else {
  // Tacro (Prograf ou Advagraf)
  CL  = TVCL * pow(HT / 35, HTCL) * pow(CYPCL_TAC, CYP) * exp(ETA1 + ETA(1));
  V1  = TVV1 * pow(STV1, ST) * exp(ETA2 + ETA(2));
  Q   = TVQ * exp(ETA3 + ETA(3));
  V2  = TVV2 * exp(ETA4 + ETA(4));
  KTR = TVKTR * pow(STKTR, ST) * exp(ETA5 + ETA(5));
  ALAG = 0;
}

ALAG_DEPOT = ALAG;

[SIGMA] @annotated
DUM1 : 0 : (placeholder)
DUM2 : 0 : (placeholder)

[ODE]
double activate_DEPOT = (current_time < VOMIT) ? 1.0 : (1 - PV);
dxdt_DEPOT   = -KTR * DEPOT;
dxdt_TRANS1  = KTR * DEPOT * activate_DEPOT - KTR * TRANS1;
dxdt_TRANS2  = KTR * TRANS1 - KTR * TRANS2;

if (ST == 2) {
  dxdt_TRANS3 = 0;
  dxdt_CENT   = KTR * TRANS2 - (CL + Q) * CENT / V1 + Q * PERI / V2;
} else {
  dxdt_TRANS3 = KTR * TRANS2 - KTR * TRANS3;
  dxdt_CENT   = KTR * TRANS3 - (CL + Q) * CENT / V1 + Q * PERI / V2;
}

dxdt_PERI = Q * CENT / V1 - Q * PERI / V2;

[TABLE]
double prop_error = (ST == 2) ? PROP_ENV : PROP_TAC;
double add_error  = (ST == 2) ? ADD_ENV  : ADD_TAC;

capture DV = (CENT / (V1 / 1000)) * (1 + prop_error) + add_error;
$CAPTURE DV CL V1 Q V2 KTR ALAG
"""
mod = mread("fusion_model", code=model_code)

# -- Machine Learning pour résidus --
class ResidualMLP(nn.Module):
    def __init__(self, input_dim):
        super(ResidualMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Charger des données fictives ou utilisateurs
def generate_synthetic_data():
    return pd.DataFrame({
        'ID': np.arange(1, 11),
        'HT': np.random.uniform(30, 45, 10),
        'CYP': np.random.choice([0, 1], 10),
        'ST': np.random.choice([0, 1, 2], 10),
        'time': np.linspace(0, 24, 10),
        'dose': np.random.uniform(1, 10, 10)
    })

# Simulation mrgsolve
def simulate_mrgsolve(data, time):
    sim_df = mod % data % f'@time={time}' % "mrgsim"
    return sim_df

# Application Streamlit
def main():
    st.title("Tacrolimus Simulation et Modèle Hybride")
    st.sidebar.header("Paramètres de simulation")

    # Entrées utilisateur
    HT = st.sidebar.slider("Hématocrite (%)", 25, 50, 35)
    CYP = st.sidebar.selectbox("CYP Expressor", [0, 1])
    ST = st.sidebar.selectbox("Type de traitement", [0, 1, 2])
    dose = st.sidebar.slider("Dose administrée", 1.0, 10.0, 5.0)
    time = st.sidebar.slider("Temps (h)", 0, 48, 24)

    data = generate_synthetic_data()
    data.loc[0, ['HT', 'CYP', 'ST', 'dose']] = [HT, CYP, ST, dose]

    # -- Simulation mrgsolve --
    st.write("## Simulation avec mrgsolve")
    sim_results = simulate_mrgsolve(data, time)
    st.dataframe(sim_results)

    # -- Machine Learning sur résidus --
    st.write("## Modèle Machine Learning pour Résidus")
    # Exemple de données d'apprentissage fictives
    X_train = data[['HT', 'CYP', 'ST', 'dose']].values
    y_train = data['dose'].values + np.random.normal(0, 0.1, len(data))

    model = ResidualMLP(input_dim=4)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    for epoch in range(100):  # Entraînement simple
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()

    st.write(f"Perte du modèle ML: {loss.item():.4f}")

    # Visualisation combinée
    st.write("## Visualisation des résultats")
    fig, ax = plt.subplots()
    ax.plot(sim_results['time'], sim_results['CENT'], label="Simulation mrgsolve")
    ax.scatter([time], [loss.item()], color="red", label="Résidus Machine Learning")
    ax.set_xlabel("Temps (h)")
    ax.set_ylabel("Concentration")
    ax.legend()
    st.pyplot(fig)


if __name__ == "__main__":
    main()
