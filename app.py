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
# SEÇÃO 1: VISUALIZAÇÃO GRÁFICA 
# ==========================================
st.header("1. Visualização: Potencial e Transições")
st.markdown("O oscilador harmônico quântico possui níveis de energia igualmente espaçados. Como o operador posição $\hat{X}$ governa as transições de dipolo, as regras de seleção permitem apenas saltos entre níveis vizinhos ($\Delta n = \pm 1$).")

N_plot = st.slider("Número de níveis para mostrar no gráfico:", min_value=3, max_value=15, value=6)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.linspace(-5, 5, 400)
V = 0.5 * x**2 

ax.plot(x, V, 'k-', lw=2, label="Potencial $V(x) = \\frac{1}{2}x^2$")

for n_val in range(N_plot):
    E_val = n_val + 0.5
    x_turn = np.sqrt(2 * E_val)
    
    ax.hlines(E_val, -x_turn, x_turn, color='blue', lw=2, alpha=0.7)
    ax.text(x_turn + 0.2, E_val, f"$E_{n_val}$", va='center', color='blue', fontweight='bold')
    
    if n_val < N_plot - 1:
        E_next = n_val + 1.5
        x_arrow = 0.5 + n_val * 0.3 
        ax.annotate("", xy=(x_arrow, E_next), xytext=(x_arrow, E_val),
                    arrowprops=dict(arrowstyle="<->", color='red', lw=1.5, shrinkA=0, shrinkB=0))

ax.set_xlabel("Posição (x)", fontsize=12)
ax.set_ylabel("Energia (E)", fontsize=12)
ax.set_title(f"Níveis de Energia e Regra de Seleção $\Delta n = \pm 1$", fontsize=14)
ax.set_ylim(0, N_plot + 1)
ax.set_xlim(-np.sqrt(2*(N_plot+2)), np.sqrt(2*(N_plot+2))) 
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc="upper left")

st.pyplot(fig)
st.caption("Em vermelho, as transições permitidas. Isso explica por que a matriz $X$ abaixo só possui elementos fora da diagonal principal!")

st.divider()

# ==========================================
# SEÇÃO 2: NUMPY (AGORA COM MATRIZ DINÂMICA)
# ==========================================
st.header("2. Abordagem Numérica (NumPy)")
st.markdown("Simulação usando matrizes finitas truncadas para $N$ estados.")

N = st.number_input("Escolha o número de níveis retidos na matriz (N):", min_value=3, max_value=50, value=6)

# Cálculos Numéricos
n = np.arange(N)
X = np.diag(np.sqrt((n[:-1] + 1)/2), 1)
X = X + X.T 
E = n + 0.5
W = E[:, None] - E[None, :] 
P = 1j*W*X 
C = X@P - P@X

st.subheader("Matriz Comutador $[X, P]$ e o Paradoxo do Traço")
st.markdown("Na álgebra matricial, a operação de **traço** é a soma dos elementos da diagonal principal. "
            "Abaixo, destacamos a diagonal em vermelho para mostrar exatamente onde o artefato numérico aparece para forçar a soma a ser zero:")

# Função para desenhar a matriz Numpy como LaTeX
def gerar_latex_matriz_traco(matriz, max_show=8):
    dim = matriz.shape[0]
    linhas = []
    
    def formatar(v, is_diag):
        val = round(v.imag)
        if val == 0: return "0"
        if val == 1: texto = "i"
        elif val == -1: texto = "-i"
        else: texto = f"{val}i"
        # Destaca a diagonal em vermelho
        return f"\\color{{red}}{{{texto}}}" if is_diag else texto

    if dim <= max_show:
        for i in range(dim):
            linha = [formatar(matriz[i, j], i == j) for j in range(dim)]
            linhas.append(" & ".join(linha))
    else:
        # Se for muito grande, mostra o topo esquerdo e o canto inferior direito
        for i in range(3):
            linha = [formatar(matriz[i, j], i == j) for j in range(3)]
            linha.append("\\cdots")
            linha.append(formatar(matriz[i, dim-1], i == dim-1))
            linhas.append(" & ".join(linha))
        
        linhas.append("\\vdots & \\vdots & \\ddots & \\vdots & \\vdots")
        
        linha_fim = ["0", "0", "\\cdots", "0", formatar(matriz[dim-1, dim-1], True)]
        linhas.append(" & ".join(linha_fim))
        
    return "\\begin{pmatrix}\n" + " \\\\\n".join(linhas) + "\n\\end{pmatrix}"

# Renderiza a Matriz Quadrada
st.latex(r"C = [X, P] = " + gerar_latex_matriz_traco(C))

# Renderiza a equação do traço (soma da diagonal)
st.latex(r"\text{Tr}(C) = \sum_{k=1}^{N} C_{kk} = \color{red}{i + i + \dots " + f" {-(N-1)}i" + r"} = 0")

st.info("Na teoria exata, a diagonal seria infinita e preenchida apenas com $i$. "
        "Aqui, o último nível sofre um colapso drástico (o $-5i$ na matriz 6x6, por exemplo) provando que o truncamento quebra a álgebra na borda.")

st.markdown("**Regra de Soma TRK:**")
trk = np.round(2*np.einsum('ln,ln->n', W, abs(X)**2), 6)
st.write(trk)

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
