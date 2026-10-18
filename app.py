import streamlit as st
import numpy as np
import math
import sympy as sp
import matplotlib.pyplot as plt
from sympy.physics.quantum import Commutator, Dagger
from sympy.physics.quantum.boson import BosonOp

# Configuração da página
st.set_page_config(page_title="Oscilador Harmônico Quântico", layout="centered")

st.title("Oscilador Harmônico Quântico")
st.markdown("Comparação entre Mecânica Matricial Truncada, Álgebra Exata e Visualização Física.")

# ==========================================
# SEÇÃO 1: VISUALIZAÇÃO GRÁFICA 
# ==========================================
st.header("1. Visualização: Potencial, Níveis e Probabilidade")
st.markdown("Abaixo, vemos o poço de potencial parabólico. Sobre cada nível de energia, plotamos a **densidade de probabilidade** $|\psi_n(x)|^2$ da partícula ser encontrada naquela posição.")

N_plot = st.slider("Número de níveis para mostrar no gráfico:", min_value=3, max_value=12, value=6)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(-5, 5, 500)
V = 0.5 * x**2 

# Plotando o Potencial
ax.plot(x, V, 'k-', lw=2, label="Potencial $V(x) = \\frac{1}{2}x^2$")

# Função para calcular a função de onda do oscilador harmônico (adimensional)
def psi_n(n, x_vals):
    # Coeficientes do polinômio de Hermite H_n: [0, 0, ..., 1]
    coeffs = [0] * n + [1]
    H_n = np.polynomial.hermite.hermval(x_vals, coeffs)
    # Fator de normalização
    norm = 1.0 / np.sqrt((2**n) * math.factorial(n) * np.sqrt(np.pi))
    return norm * np.exp(-x_vals**2 / 2) * H_n

for n_val in range(N_plot):
    E_val = n_val + 0.5
    x_turn = np.sqrt(2 * E_val)
    
    # Linha base do nível de energia (mais clara agora)
    ax.hlines(E_val, -x_turn, x_turn, color='gray', lw=1, linestyle='--')
    ax.text(x_turn + 0.2, E_val, f"$E_{n_val}$", va='center', color='blue', fontweight='bold')
    
    # Calculando a densidade de probabilidade |psi|^2
    prob_density = np.abs(psi_n(n_val, x))**2
    
    # Escalando a curva de probabilidade para caber visualmente entre os níveis (altura máx ~0.8)
    scale = 0.8 / np.max(prob_density) 
    curva = E_val + prob_density * scale
    
    # Desenhando a curva de probabilidade
    ax.plot(x, curva, color='blue', lw=1.5)
    ax.fill_between(x, E_val, curva, color='blue', alpha=0.2)
    
    # Setas de transição na borda esquerda para não poluir o centro
    if n_val < N_plot - 1:
        E_next = n_val + 1.5
        x_arrow = - (0.5 + n_val * 0.3)
        ax.annotate("", xy=(x_arrow, E_next), xytext=(x_arrow, E_val),
                    arrowprops=dict(arrowstyle="<->", color='red', lw=1.5))

ax.set_xlabel("Posição (x)", fontsize=12)
ax.set_ylabel("Energia (E)", fontsize=12)
ax.set_ylim(0, N_plot + 1)
ax.set_xlim(-np.sqrt(2*(N_plot+2)), np.sqrt(2*(N_plot+2))) 
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15))

st.pyplot(fig)
st.caption("As áreas preenchidas em azul mostram a probabilidade quântica. Note como os estados mais altos começam a ter maior probabilidade nas bordas do potencial (limite clássico) e as setas vermelhas indicam as transições de dipolo $\Delta n = \pm 1$.")

st.divider()

# ==========================================
# SEÇÃO 2: NUMPY (MATRIZES FINITAS)
# ==========================================
st.header("2. Abordagem Numérica (NumPy)")

N = st.number_input("Escolha o tamanho da matriz truncada (N):", min_value=3, max_value=50, value=6)

# Cálculos Numéricos
n = np.arange(N)
X = np.diag(np.sqrt((n[:-1] + 1)/2), 1)
X = X + X.T 
E = n + 0.5
W = E[:, None] - E[None, :] 
P = 1j*W*X 
C = X@P - P@X

st.subheader("1. Matriz Comutador e o Paradoxo do Traço")

def gerar_latex_matriz_traco(matriz, max_show=8):
    dim = matriz.shape[0]
    linhas = []
    
    def formatar(v, is_diag):
        val = round(v.imag)
        if val == 0: return "0"
        if val == 1: texto = "i"
        elif val == -1: texto = "-i"
        else: texto = f"{val}i"
        return f"\\color{{red}}{{{texto}}}" if is_diag else texto

    if dim <= max_show:
        for i in range(dim):
            linha = [formatar(matriz[i, j], i == j) for j in range(dim)]
            linhas.append(" & ".join(linha))
    else:
        for i in range(3):
            linha = [formatar(matriz[i, j], i == j) for j in range(3)]
            linha.append("\\cdots")
            linha.append(formatar(matriz[i, dim-1], i == dim-1))
            linhas.append(" & ".join(linha))
        linhas.append("\\vdots & \\vdots & \\ddots & \\vdots & \\vdots")
        linha_fim = ["0", "0", "\\cdots", "0", formatar(matriz[dim-1, dim-1], True)]
        linhas.append(" & ".join(linha_fim))
        
    return "\\begin{pmatrix}\n" + " \\\\\n".join(linhas) + "\n\\end{pmatrix}"

st.latex(r"C = [X, P] = " + gerar_latex_matriz_traco(C))
st.latex(r"\text{Tr}(C) = \sum_{k=1}^{N} C_{kk} = \color{red}{i + i + \dots " + f" {-(N-1)}i" + r"} = 0")

st.info("O último nível sofre um colapso drástico para forçar a soma da diagonal a ser zero. O comutador so se realiza de fato em dimensão infinita")

# === TRK COM NOTAÇÃO INDICIAL ===
st.subheader("2. Regra de Soma TRK (Thomas-Reiche-Kuhn)")
st.markdown("A regra TRK dita que a soma das forças de oscilador deve ser igual a 1 para **todos** os níveis de energia $n$:")

trk = np.round(2*np.einsum('ln,ln->n', W, abs(X)**2), 6)

def gerar_latex_vetor_trk(vetor, max_show=8):
    dim = len(vetor)
    linhas = []
    if dim <= max_show:
        for i in range(dim):
            val = int(vetor[i]) if vetor[i].is_integer() else vetor[i]
            linhas.append(f"\\color{{red}}{{{val}}}" if i == dim-1 else str(val))
    else:
        for i in range(3):
            linhas.append(str(int(vetor[i])))
        linhas.append("\\vdots")
        val = int(vetor[-1]) if vetor[-1].is_integer() else vetor[-1]
        linhas.append(f"\\color{{red}}{{{val}}}")
    return "\\begin{pmatrix}\n" + " \\\\\n".join(linhas) + "\n\\end{pmatrix}"

st.latex(r"\sum_{l=0}^{N-1} 2(E_l - E_n) |X_{ln}|^2 = " + gerar_latex_vetor_trk(trk))

st.info("Assim como no comutador, a álgebra matricial quebra a regra da Física no último estado do vetor pela ausência dos estados superiores.")

st.divider()

# ==========================================
# SEÇÃO 3: SYMPY
# ==========================================
st.header("3. Abordagem Analítica (SymPy)")

a = BosonOp('a')
X_sym = (a + Dagger(a)) / sp.sqrt(2)
P_sym = sp.I * (Dagger(a) - a) / sp.sqrt(2)

comutador_XP = Commutator(X_sym, P_sym)
resultado = comutador_XP.doit().expand()

st.latex(r"[X, P] = " + sp.latex(resultado))
st.success("O SymPy avalia a álgebra quântica num espaço infinito. O resultado é a constante imaginária exata, livre de quebras na fronteira.")
