import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Estilo para gráficos limpios (ideal para informes o documentos en LaTeX)
plt.style.use('seaborn-v0_8-whitegrid')

# --- EJERCICIO 1 ---
def plot_ejercicio1():
    df = pd.read_csv("ej1_equilibrio.csv")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(df['iteracion'], df['magnetizacion'], color='tab:blue', alpha=0.8)
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Magnetización por sitio', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title('Termalización (L=16, T=2.0)')

    plt.tight_layout()
    plt.savefig('ej1_magnetizacion.png', dpi=300)
    plt.show()

# --- EJERCICIO 3 ---
def func_exp(r, xi, A):
    # C(r) decae exponencialmente
    return A * np.exp(-r / xi)

def plot_ejercicio3():
    df = pd.read_csv("ej3_correlacion.csv")
    T_vals = df['T'].values

    xi_vals = []

    # Extraer r=1 hasta L/2
    r_cols = [col for col in df.columns if col.startswith('C_')]
    r_vals = np.array([int(col.split('_')[1]) for col in r_cols])

    for index, row in df.iterrows():
        C_data = row[r_cols].values.astype(float)

        # Ignorar valores negativos o muy ruidosos por el logaritmo
        valid = C_data > 0
        r_fit = r_vals[valid]
        C_fit = C_data[valid]

        if len(r_fit) > 2:
            try:
                # Ajuste para encontrar xi
                popt, _ = curve_fit(func_exp, r_fit, C_fit, p0=[1.0, 1.0], maxfev=1000)
                xi_vals.append(popt[0])
            except RuntimeError:
                xi_vals.append(np.nan)
        else:
            xi_vals.append(np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(T_vals, xi_vals, 'o-', color='tab:purple')
    ax.set_xlabel('Temperatura')
    ax.set_ylabel(r'Longitud de correlación $\xi$')
    ax.axvline(2.269, color='k', linestyle='--', alpha=0.5, label=r'$T_c$ Onsager')
    ax.set_title('Divergencia de la Longitud de Correlación')
    ax.legend()

    plt.tight_layout()
    plt.savefig('ej3_correlacion.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    print("Generando Gráfico Ejercicio 1...")
    plot_ejercicio1()

    print("Generando Gráfico Ejercicio 3...")
    plot_ejercicio3()