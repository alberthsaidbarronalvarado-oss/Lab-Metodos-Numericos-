import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder


# DATOS DE ENTRENAMIENTO


datos = {
    "Edad":[
        20,22,25,28,30,35,40,45,50,55,
        23,27,32,38,42,48,60,29,34,41,
        26,31,36,44,52,58,24,33,39,47
    ],

    "Ingreso":[
        1000,1200,1500,1800,5000,
        4500,6000,5500,7000,8000,
        1300,2000,3500,4000,5000,
        6500,9000,2500,3200,4800,
        1700,2800,4200,5800,7200,
        8500,1400,3100,4600,6200
    ],

    "Empleo":[
        "No","No","Sí","Sí","No",
        "Sí","No","Sí","Sí","No",
        "No","Sí","Sí","Sí","No",
        "Sí","No","No","Sí","Sí",
        "No","Sí","Sí","Sí","Sí",
        "No","No","Sí","Sí","Sí"
    ],

    "Compra":[
        "No","No","No","No","Sí",
        "Sí","Sí","Sí","Sí","Sí",
        "No","No","Sí","Sí","Sí",
        "Sí","Sí","No","Sí","Sí",
        "No","Sí","Sí","Sí","Sí",
        "Sí","No","Sí","Sí","Sí"
    ]
}

df = pd.DataFrame(datos)


# CODIFICACIÓN


encoder_empleo = LabelEncoder()

df["Empleo"] = encoder_empleo.fit_transform(df["Empleo"])

X = df[["Edad", "Ingreso", "Empleo"]]
y = df["Compra"]


# MODELO


modelo = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

modelo.fit(X, y)


# FUNCIÓN PREDECIR


def predecir():

    try:

        edad = int(entry_edad.get())
        ingreso = float(entry_ingreso.get())

        empleo_texto = combo_empleo.get()

        empleo_num = encoder_empleo.transform(
            [empleo_texto]
        )[0]

        cliente = [[
            edad,
            ingreso,
            empleo_num
        ]]

        resultado = modelo.predict(cliente)[0]

        probabilidades = modelo.predict_proba(cliente)

        clases = modelo.classes_

        prob_no = 0
        prob_si = 0

        for i in range(len(clases)):

            if clases[i] == "No":
                prob_no = probabilidades[0][i] * 100

            if clases[i] == "Sí":
                prob_si = probabilidades[0][i] * 100

        texto = f"""
Resultado: {resultado}

Probabilidad de Compra:
{prob_si:.2f} %

Probabilidad de No Compra:
{prob_no:.2f} %
"""

        lbl_resultado.config(
            text=texto
        )

    except Exception as e:

        messagebox.showerror(
            "Error",
            "Ingrese datos válidos"
        )

# INTERFAZ


ventana = tk.Tk()

ventana.title(
    "Árbol de Decisión - Compra o No Compra"
)

ventana.geometry("500x500")

titulo = tk.Label(
    ventana,
    text="Predicción de Compra",
    font=("Arial",18,"bold")
)

titulo.pack(pady=10)

# Edad

tk.Label(
    ventana,
    text="Edad"
).pack()

entry_edad = tk.Entry(
    ventana
)

entry_edad.pack()

# Ingreso

tk.Label(
    ventana,
    text="Ingreso Mensual"
).pack()

entry_ingreso = tk.Entry(
    ventana
)

entry_ingreso.pack()

# Empleo

tk.Label(
    ventana,
    text="¿Tiene empleo?"
).pack()

combo_empleo = ttk.Combobox(
    ventana,
    values=["Sí","No"]
)

combo_empleo.current(0)

combo_empleo.pack()

# Botón

btn = tk.Button(
    ventana,
    text="Predecir",
    command=predecir,
    bg="lightgreen",
    font=("Arial",12,"bold")
)

btn.pack(pady=20)

# Resultado

lbl_resultado = tk.Label(
    ventana,
    text="",
    font=("Arial",12)
)

lbl_resultado.pack()

ventana.mainloop()
 ## compran o no compran

import matplotlib.pyplot as plt

compras = df["Compra"].value_counts()

plt.figure(figsize=(6,4))
plt.pie(
    compras,
    labels=compras.index,
    autopct="%1.1f%%"
)

plt.title("Porcentaje General de Compras")
plt.show()

##compra segun empleo
tabla_empleo = pd.crosstab(
    df["Empleo"],
    df["Compra"],
    normalize="index"
) * 100

tabla_empleo.plot(kind="bar")

plt.title("Compra según Empleo")
plt.ylabel("Porcentaje")
plt.show()

##segun edad 
df["GrupoEdad"] = pd.cut(
    df["Edad"],
    bins=[18,30,45,60,80],
    labels=[
        "18-30",
        "31-45",
        "46-60",
        "61+"
    ]
)
tabla_edad = pd.crosstab(
    df["GrupoEdad"],
    df["Compra"],
    normalize="index"
) * 100

tabla_edad.plot(kind="bar")

plt.title("Compra según Edad")
plt.ylabel("Porcentaje")
plt.show()

##segun sus ingresos
df["GrupoIngreso"] = pd.cut(
    df["Ingreso"],
    bins=[0,2000,4000,6000,10000],
    labels=[
        "0-2000",
        "2001-4000",
        "4001-6000",
        "6001+"
    ]
)

tabla_ingreso = pd.crosstab(
    df["GrupoIngreso"],
    df["Compra"],
    normalize="index"
) * 100

tabla_ingreso.plot(kind="bar")

plt.title("Compra según Ingreso")
plt.ylabel("Porcentaje")
plt.show()