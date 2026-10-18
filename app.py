import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.physics.quantum import Commutator, Dagger
from sympy.physics.quantum.boson import BosonOp

# Configuração da página
st.set_page_config(page_title="Oscilador Harmônico Quântico", layout="centered")

st.title("Oscilador Harmônico Quântico")
st.markdown("Comparação entre Mecânica Matricial Truncada, Álgebra Exata e Visualização Física.")

# ==========================================
# SEÇÃO 1: VISUALIZAÇÃO GRÁFICA (NOVO)
# ==========================================
st.header("1. Visualização: Potencial e Transições")
st.markdown("O oscilador harmônico quântico possui níveis de energia igualmente espaçados. Como o operador posição $\hat{X}$ governa as transições de dipolo, as regras de seleção permitem apenas saltos entre níveis vizinhos ($\Delta n = \pm 1$).")

# Controle para o gráfico
N_plot = st.slider("Número de níveis para mostrar no gráfico:", min_value=3, max_value=15, value=6)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-5, 5, 400)
V = 0.5 * x**2 # Potencial harmônico em unidades naturais

# Plotando o poço de potencial
ax.plot(x, V, 'k-', lw=2, label="Potencial $V(x) = \\frac{1}{2}x^2$")

# Plotando os níveis de energia e transições
for n_val in range(N_plot):
    E_val = n_val + 0.5
    # Pontos de retorno clássico (onde a energia corta a parábola: E = 0.5 * x^2)
    x_turn = np.sqrt(2 * E_val)
    
    # Linha do nível de energia
    ax.hlines(E_val, -x_turn, x_turn, color='blue', lw=2, alpha=0.7)
    ax.text(x_turn + 0.2, E_val, f"$E_{n_val}$", va='center', color='blue', fontweight='bold')
    
    # Setas de transição permitida (Delta n = +/- 1)
    if n_val < N_plot - 1:
        E_next = n_val + 1.5
        # Desloca as setas um pouco para a direita para não embolar
        x_arrow = 0.5 + n_val * 0.3 
        ax.annotate("", xy=(x_arrow, E_next), xytext=(x_arrow, E_val),
                    arrowprops=dict(arrowstyle="<->", color='red', lw=1.5, shrinkA=0, shrinkB=0))

# Ajustes estéticos do gráfico
ax.set_xlabel("Posição (x)", fontsize=12)
ax.set_ylabel("Energia (E)", fontsize=12)
ax.set_title(f"Níveis de Energia e Regra de Seleção $\Delta n = \pm 1$", fontsize=14)
ax.set_ylim(0, N_plot + 1)
ax.set_xlim(-np.sqrt(2*(N_plot+2)), np.sqrt(2*(N_plot+2))) # Ajusta o zoom do eixo x dinamicamente
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc="upper left")

# Mostra o gráfico no Streamlit
st.pyplot(fig)
st.caption("Em vermelho, as transições permitidas. Isso explica por que a matriz $X$ abaixo só possui elementos fora da diagonal principal!")

st.divider()

# ==========================================
# SEÇÃO 2: NUMPY
# ==========================================
st.header("2. Abordagem Numérica (NumPy)")
st.markdown("Simulação usando matrizes finitas truncadas para $N$ estados.")

N = st.number_input("Escolha o número de níveis retidos na matriz (N):", min_value=3, max_value=50, value=10)

n = np.arange(N)
X = np.diag(np.sqrt((n[:-1] + 1)/2), 1)
X = X + X.T 
E = n + 0.5
W = E[:, None] - E[None, :] 
P = 1j*W*X 
C = X@P - P@X

st.subheader("Resultados do Comutador $[X, P]$")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Diagonal Principal (Parte Imaginária):**")
    st.write(np.round(np.diag(C).imag, 6))

with col2:
    st.markdown("**Traço da Matriz:**")
    st.write(np.round(np.trace(C), 6))
    
st.markdown("**Regra de Soma TRK:**")
trk = np.round(2*np.einsum('ln,ln->n', W, abs(X)**2), 6)
st.write(trk)
st.info(f"Note o artefato numérico no último nível ($n={N-1}$) causado pelo truncamento da matriz infinita.")

st.divider()

# ==========================================
# SEÇÃO 3: SYMPY
# ==========================================
st.header("3. Abordagem Analítica (SymPy)")
st.markdown("Cálculo exato usando álgebra simbólica de dimensão infinita.")

a = BosonOp('a')
X_sym = (a + Dagger(a)) / sp.sqrt(2)
P_sym = sp.I * (Dagger(a) - a) / sp.sqrt(2)

comutador_XP = Commutator(X_sym, P_sym)
resultado = comutador_XP.doit().expand()

st.latex(r"[X, P] = " + sp.latex(resultado))
st.success("O resultado analítico mostra a constante imaginária exata, provando que o erro acima é puramente um efeito de borda computacional da matriz finita.")
