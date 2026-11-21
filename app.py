# %% [markdown]
# 1. Cargamos los datos 
# 2. Cargamos los datos futuros 
# 3. perparar los datos futuros: normalizar , dummies 
# 4. Aplicamos el modelo para la predicción.

# %%
# Cargamos lo datos 
#Importamos librerías básicas
import pandas as pd # manipulacion dataframes
import numpy as np  # matrices y vectores
import matplotlib.pyplot as plt #gráfica

# %%
#Cargamos el modelo guardado (mismo orden en que se guardó: modelo, normalizador, dummies)
import pickle
filename = 'modelo-reg.pkl'
modelo, min_max_scaler, encoder_dummies = pickle.load(open(filename, 'rb'))
#modelo

# %%
#Cargamos los datos futuros
#quotechar="'" quita las comillas simples de los textos ('Mass Effect' -> Mass Effect)
#df = pd.read_csv("videojuegos-datosFuturos.csv", quotechar="'")
#df

# %% [markdown]
# ## INTERFAZ GRÁFICA

# %%
#Se crea interfaz gráfica con streamlit para captura de los datos
 
import streamlit as st # librería para crear aplicaciones web
 
st.title('Predicción de inversión en una tienda de videojuegos')
 
Edad = st.slider('Edad', min_value=14, max_value=52, value=20, step=1)
videojuego = st.selectbox('Videojuego', ["'Mass Effect'","'Battlefield'", "'Fifa'","'KOA: Reckoning'","'Crysis'","'Sim City'","'Dead Space'","'F1'"])
Plataforma = st.selectbox('Plataforma', ["'Play Station'", "'Xbox'","PC","Otros"])
Sexo = st.selectbox('Sexo', ['Hombre', 'Mujer'])
Consumidor_habitual = st.selectbox('Consumidor_habitual', ['True', 'False'])
 
 
#Dataframe
datos = [[Edad, videojuego,Plataforma,Sexo,Consumidor_habitual]]
df = pd.DataFrame(datos, columns=['Edad', 'videojuego','Plataforma','Sexo','Consumidor_habitual']) #Dataframe con los mismos nombres de variables

# %%
data = df.copy() #creamos una copia de seguridad


#Configuración de variables
variables_numericas = ["Edad"]
variables_categoricas = ["videojuego", "Sexo", "Plataforma", "Consumidor_habitual"]

# %%
#Creación de dummies con transform
dummies = encoder_dummies.transform(data[variables_categoricas])
data = pd.concat([data[variables_numericas], dummies], axis="columns")
data.head()

# %%
#Se normaliza la edad para predecir con Knn, Red, SVM, Reg.
# Si se selecciona un tree, se comentan las líneas
 
#data[variables_numericas]= min_max_scaler.transform(data[variables_numericas])
#data.head()

# %% [markdown]
# 
# ##  Predicción
# 
# 

# %%
#Hacemos la predicción con el Tree
Y_pred = modelo.predict(data)
print(Y_pred)

# %%
df['Prediccion']=Y_pred
df.head()

# %%
df

# %%
# Recordar medida de error del modelo
 
st.warning("El modelo tiene un error del 15% (mape: error porcentual)")
