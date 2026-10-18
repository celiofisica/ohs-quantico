import streamlit as st
import numpy as np
import sympy as sp
from sympy.physics.quantum import Commutator, Dagger
from sympy.physics.quantum.boson import BosonOp

# Configuração da página
st.set_page_config(page_title="Oscilador Harmônico Quântico", layout="centered")

st.title("Oscilador Harmônico Quântico")
st.markdown("Comparação entre Mecânica Matricial Truncada (Numérica) e Álgebra de Operadores (Analítica).")

# ==========================================
# SEÇÃO 1: NUMPY
# ==========================================
st.header("1. Abordagem Numérica (NumPy)")
st.markdown("Simulação usando matrizes finitas truncadas para $N$ estados.")

# Controle interativo para o aluno escolher o tamanho da matriz
N = st.slider("Escolha o número de níveis retidos (N):", min_value=3, max_value=20, value=10)

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
    # Mostra o erro no último elemento devido ao truncamento
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
# SEÇÃO 2: SYMPY
# ==========================================
st.header("2. Abordagem Analítica (SymPy)")
st.markdown("Cálculo exato usando álgebra simbólica de dimensão infinita.")

a = BosonOp('a')
X_sym = (a + Dagger(a)) / sp.sqrt(2)
P_sym = sp.I * (Dagger(a) - a) / sp.sqrt(2)

comutador_XP = Commutator(X_sym, P_sym)
resultado = comutador_XP.doit().expand()

st.markdown("Equação do Comutador calculada pelo servidor:")
# O Streamlit renderiza SymPy perfeitamente em LaTeX na web
st.latex(r"[X, P] = " + sp.latex(resultado))

st.success("O resultado analítico mostra a constante imaginária exata, sem o paradoxo do traço zero ou erros de borda da matriz finita.")