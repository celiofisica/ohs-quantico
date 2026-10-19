"""
OHS Quântico — página principal.

O oscilador harmônico simples (OHS) na mecânica matricial de Heisenberg:
matrizes truncadas (NumPy), verificação algébrica em dimensão infinita (SymPy)
e, só para comparação, a imagem de Schrödinger.

Material complementar ao artigo sobre o OHS via mecânica matricial de
Heisenberg (Revista Brasileira de Ensino de Física, 2026).
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from sympy.physics.quantum import Commutator, Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form

st.set_page_config(page_title="Oscilador Harmônico Quântico", layout="centered")

st.title("Oscilador Harmônico Quântico")
st.markdown(
    "Mecânica matricial truncada, álgebra exata em dimensão infinita e, "
    "para comparação, a imagem de Schrödinger."
)

st.info(
    r"**Nota sobre unidades:** por convenção e para simplificar a notação "
    r"matemática, em todo este código adotamos o sistema de *unidades naturais*: "
    r"a constante de Planck reduzida, a massa da partícula e a frequência do "
    r"oscilador são tomadas como iguais a 1 ($\hbar = m = \omega = 1$)."
)

st.divider()

# ==========================================
# SEÇÃO 1: MATRIZES TRUNCADAS (NUMPY)
# ==========================================
st.header("1. Abordagem matricial truncada (NumPy)")

st.markdown(
    r"""
Seguimos a ordem do artigo. As únicas entradas são:

1. as **amplitudes de transição** $|X_{n,n+1}|^2 = (n+1)/2$, fixadas pela regra de
   soma de Thomas–Reiche–Kuhn (TRK) aplicada nível a nível, com transições só
   entre vizinhos ($\Delta n = \pm 1$);
2. a **dinâmica**: como cada elemento $X_{rs}$ oscila com a frequência de Bohr
   $\omega_{rs} = E_r - E_s$, o momento é $P_{rs} = i\,(E_r - E_s)\,X_{rs}$.

O comutador $[X, P]$ é então **calculado** a partir dessas matrizes, não postulado.
O preço de usar um computador é truncar a matriz em $N \times N$ — e é justamente
aí que a física aparece.
"""
)

N = st.number_input(
    "Escolha o tamanho da matriz truncada (N):", min_value=3, max_value=50, value=6
)

# --- Cálculos numéricos ---
n = np.arange(N)
X = np.diag(np.sqrt((n[:-1] + 1) / 2), 1)
X = X + X.T                      # X real e simétrica: X_{n,n+1} = sqrt((n+1)/2)
E = n + 0.5                      # níveis E_n = n + 1/2
W = E[:, None] - E[None, :]      # W_{rs} = E_r - E_s  (frequências de Bohr)
P = 1j * W * X                   # P_{rs} = i (E_r - E_s) X_{rs}  (dinâmica)
C = X @ P - P @ X                # comutador

st.subheader(
    "Resultados do comutador [X, P] = XP − PX",
    anchor="resultados-do-comutador-x-p-x-p-x-p",
)


def gerar_latex_matriz_traco(matriz, max_show=8):
    dim = matriz.shape[0]
    linhas = []

    def formatar(v, is_diag):
        val = round(v.imag)
        if val == 0:
            return "0"
        if val == 1:
            texto = "i"
        elif val == -1:
            texto = "-i"
        else:
            texto = f"{val}i"
        return f"\\color{{red}}{{{texto}}}" if is_diag else texto

    if dim <= max_show:
        for i in range(dim):
            linha = [formatar(matriz[i, j], i == j) for j in range(dim)]
            linhas.append(" & ".join(linha))
    else:
        for i in range(3):
            linha = [formatar(matriz[i, j], i == j) for j in range(3)]
            linha.append("\\cdots")
            linha.append(formatar(matriz[i, dim - 1], i == dim - 1))
            linhas.append(" & ".join(linha))
        linhas.append("\\vdots & \\vdots & \\ddots & \\vdots & \\vdots")
        linha_fim = ["0", "0", "\\cdots", "0", formatar(matriz[dim - 1, dim - 1], True)]
        linhas.append(" & ".join(linha_fim))

    return "\\begin{pmatrix}\n" + " \\\\\n".join(linhas) + "\n\\end{pmatrix}"


st.latex(r"C = [X, P] = " + gerar_latex_matriz_traco(C))
st.latex(
    r"\operatorname{Tr}(C) = \sum_{k=1}^{N} C_{kk} = \color{red}{i + i + \dots "
    + f" {-(N - 1)}i"
    + r"} = 0"
)

st.info(
    "O paradoxo do traço. Para matrizes finitas, $Tr(XP - PX) = 0$ sempre, enquanto $Tr(i \mathbb{I}) = iN$. Logo, $[X, P] = i \mathbb{I}$ é impossível em dimensão finita. O que acontece aqui fisicamente: o último nível, $n = N-1$, perdeu o parceiro superior ($n=N$) que lhe daria a contribuição $+iN$ no comutador. Sem essa parcela, seu elemento diagonal despenca para $-(N-1)i$, que é exatamente o saldo negativo necessário para anular o traço de toda a matriz. Só uma matriz de dimensão infinita escapa disso."
)

# --- TRK com notação indicial ---
st.subheader("Regra de soma TRK (Thomas–Reiche–Kuhn)")
st.markdown(
    "A regra TRK dita que a soma das forças de oscilador deve ser igual a 1 para "
    "**todos** os níveis de energia $n$:"
)

trk = np.round(2 * np.einsum("ln,ln->n", W, abs(X) ** 2), 6)


def gerar_latex_vetor_trk(vetor, max_show=8):
    dim = len(vetor)
    linhas = []
    if dim <= max_show:
        for i in range(dim):
            val = int(vetor[i]) if vetor[i].is_integer() else vetor[i]
            linhas.append(f"\\color{{red}}{{{val}}}" if i == dim - 1 else str(val))
    else:
        for i in range(3):
            linhas.append(str(int(vetor[i])))
        linhas.append("\\vdots")
        val = int(vetor[-1]) if vetor[-1].is_integer() else vetor[-1]
        linhas.append(f"\\color{{red}}{{{val}}}")
    return "\\begin{pmatrix}\n" + " \\\\\n".join(linhas) + "\n\\end{pmatrix}"


st.latex(r"\sum_{l=0}^{N-1} 2(E_l - E_n) |X_{ln}|^2 = " + gerar_latex_vetor_trk(trk))

st.info(
    "Assim como no comutador, a álgebra matricial quebra a regra da física no último "
    "estado do vetor pela ausência dos estados superiores. Não é coincidência: a "
    "diagonal de $[X,P]$ é $i$ vezes a soma TRK de cada nível — são o mesmo objeto."
)

st.divider()

# ==========================================
# SEÇÃO 2: VERIFICAÇÃO ALGÉBRICA (SYMPY)
# ==========================================
st.header("2. Verificação algébrica em dimensão infinita (SymPy)")

st.markdown(
    r"""
**O que é o SymPy?** Para um computador tradicional, resolver matemática quase
sempre significa fazer contas com matrizes numéricas e aproximações. O **SymPy**
(*Symbolic Python*) manipula a álgebra simbolicamente — com letras como
$x$, $p$, $a$, $a^\dagger$ — exatamente como fazemos no papel, sem truncar nada.

**Atenção ao que esta seção faz e ao que ela não faz.** O SymPy trata $a$ e
$a^\dagger$ como operadores bosônicos, ou seja, parte da regra $[a, a^\dagger] = 1$.
Essa regra é *equivalente* a $[X, P] = i\hbar$. Portanto o cálculo abaixo **não
deduz** o comutador — ele **verifica** que a álgebra é consistente e que, sem
truncamento, a anomalia da fronteira desaparece. A dedução propriamente dita é a
da Seção 1 (e do artigo): TRK diagonal + dinâmica.
"""
)

a = BosonOp("a")
X_sym = (a + Dagger(a)) / sp.sqrt(2)
P_sym = sp.I * (Dagger(a) - a) / sp.sqrt(2)

bruto = Commutator(X_sym, P_sym).doit().expand()
resultado = normal_ordered_form(bruto)

st.latex(
    r"[X, P] = "
    + sp.latex(bruto)
    + r" = i\,[a, a^\dagger] = "
    + sp.latex(resultado)
    + r" \equiv i\hbar \quad (\text{pois } \hbar = 1)"
)

st.success(
    "Em dimensão infinita não existe “último estado”, e o argumento do traço da "
    "Seção 1 simplesmente não se aplica: $X$ e $P$ são matrizes infinitas "
    "(operadores ilimitados) e $\\operatorname{Tr}(XP)$ nem sequer está definido. "
    "O resultado exato $i\\hbar$ aparece sem anomalia alguma — que é precisamente o "
    "que as matrizes truncadas não conseguem reproduzir."
)

st.divider()

# ==========================================
# SEÇÃO 3: PARA COMPARAÇÃO — IMAGEM DE SCHRÖDINGER
# ==========================================
st.header("3. Para comparação: a imagem de Schrödinger")

st.markdown(
    r"""
Nada do que foi feito acima usou funções de onda — esse é o ponto da mecânica
matricial: só entram grandezas observáveis (frequências e intensidades de
transição). Para quem já conhece a imagem de Schrödinger, a figura abaixo mostra
o mesmo sistema do jeito mais familiar: o poço parabólico $V(x) = \tfrac{1}{2}x^2$,
os níveis $E_n = n + \tfrac{1}{2}$ e, sobre cada nível, a **densidade de
probabilidade** $|\psi_n(x)|^2$.
"""
)

N_plot = st.slider(
    "Número de níveis para mostrar no gráfico:", min_value=3, max_value=12, value=6
)

fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(-5, 5, 500)
V = 0.5 * x**2

ax.plot(x, V, "k-", lw=2, label=r"Potencial $V(x) = \frac{1}{2}x^2$")


def psi_n(n_, x_vals):
    """Autofunção normalizada do OHS em unidades naturais."""
    coeffs = [0] * n_ + [1]
    H_n = np.polynomial.hermite.hermval(x_vals, coeffs)
    norm = 1.0 / np.sqrt((2**n_) * math.factorial(n_) * np.sqrt(np.pi))
    return norm * np.exp(-(x_vals**2) / 2) * H_n


for n_val in range(N_plot):
    E_val = n_val + 0.5
    x_turn = np.sqrt(2 * E_val)

    ax.hlines(E_val, -x_turn, x_turn, color="gray", lw=1, linestyle="--")
    ax.text(x_turn + 0.2, E_val, f"$E_{n_val}$", va="center", color="blue", fontweight="bold")

    prob_density = np.abs(psi_n(n_val, x)) ** 2
    scale = 0.8 / np.max(prob_density)
    curva = E_val + prob_density * scale

    ax.plot(x, curva, color="blue", lw=1.5)
    ax.fill_between(x, E_val, curva, color="blue", alpha=0.2)

    if n_val < N_plot - 1:
        E_next = n_val + 1.5
        x_arrow = -(0.5 + n_val * 0.3)
        ax.annotate(
            "",
            xy=(x_arrow, E_next),
            xytext=(x_arrow, E_val),
            arrowprops=dict(arrowstyle="<->", color="red", lw=1.5),
        )

ax.set_xlabel("Posição (x)", fontsize=12)
ax.set_ylabel("Energia (E)", fontsize=12)
ax.set_ylim(0, N_plot + 1)
ax.set_xlim(-np.sqrt(2 * (N_plot + 2)), np.sqrt(2 * (N_plot + 2)))
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15))

st.pyplot(fig)
st.caption(
    r"As áreas preenchidas em azul mostram a probabilidade quântica. Note como os "
    r"estados mais altos começam a ter maior probabilidade nas bordas do potencial "
    r"(limite clássico) e as setas vermelhas indicam as transições de dipolo "
    r"$\Delta n = \pm 1$ — as únicas que entram nas matrizes da Seção 1."
)

st.divider()

# ==========================================
# SOBRE
# ==========================================
with st.expander("Sobre este aplicativo"):
    st.markdown(
        """
Material complementar ao artigo sobre o oscilador harmônico na mecânica
matricial de Heisenberg (*Revista Brasileira de Ensino de Física*, 2026).
A página **Mulliken e a energia de ponto zero** (menu lateral) acompanha o
exercício sobre o efeito isotópico no BO do caderno de exercícios.

- Código-fonte: [github.com/celiofisica/ohs-quantico](https://github.com/celiofisica/ohs-quantico)
- Notas de aula: [doi:10.5281/zenodo.22285340](https://doi.org/10.5281/zenodo.22285340)
- Licença: MIT
"""
    )
