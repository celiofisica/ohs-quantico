"""
OHS Quântico — página 2: o efeito isotópico de Mulliken (1924) no BO
e a descoberta da energia de ponto zero.

Acompanha o exercício sobre o efeito isotópico de Mulliken do caderno de
exercícios complementar ao artigo.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="O Efeito Isotópico de Mulliken (1924)", layout="wide")

st.title("A Descoberta da Energia de Ponto Zero (1924)")
st.markdown(
    r"""
Dez meses antes de Heisenberg formular a Mecânica Matricial, R. S. Mulliken analisou
a luz emitida pelo monóxido de boro (BO). Ele usou uma “alavanca” fornecida pela
própria natureza: os isótopos. A molécula de $^{10}$BO tem a mesma nuvem eletrônica
e a mesma “mola” da $^{11}$BO, mas é ligeiramente mais leve. Isso altera as
frequências de vibração. A forma como o espectro se desloca revelou o meio quantum
($\hbar\omega/2$) escondido no estado fundamental.
"""
)

st.divider()

# ==========================================
# PARÂMETROS FÍSICOS (BO)
# ==========================================
# Massas atômicas (u) e constantes espectroscópicas do 11BO (cm^-1), sistema alfa
mB10, mB11, mO = 10.0129, 11.0093, 15.9949
mu10 = mB10 * mO / (mB10 + mO)
mu11 = mB11 * mO / (mB11 + mO)
rho = np.sqrt(mu11 / mu10)     # omega(10BO) = rho * omega(11BO), rho > 1

wl, wxl = 1885.69, 11.81       # estado inferior X 2Sigma+: omega_e'', omega_e x_e''
wu, wxu = 1260.70, 11.16       # estado superior A 2Pi:     omega_e',  omega_e x_e'
Te = 23958.7                   # termo eletrônico

# ==========================================
# SEÇÃO 1: A ALAVANCA ISOTÓPICA E O ESPECTRO
# ==========================================
st.header("1. O experimento interativo")
st.markdown(
    r"Use os controles abaixo para testar as duas hipóteses da física em 1924. "
    r"A antiga teoria de Bohr assumia que a energia vibracional era um número inteiro "
    r"de quanta ($E = v\hbar\omega$). A nova teoria exigia o meio quantum "
    r"($E = (v + 1/2)\hbar\omega$)."
)

col_ctrl, col_plot = st.columns([1, 2.5])

with col_ctrl:
    st.subheader("Controles da teoria")
    half_quantum = st.toggle("Ativar energia de ponto zero (+ 1/2)", value=True)
    anarmonicidade = st.toggle("Incluir anarmonicidade (refinamento)", value=False)

    st.info(
        f"**Massa reduzida ($\\mu$):**\n\n"
        f"$^{{11}}$BO: {mu11:.3f} u\n\n"
        f"$^{{10}}$BO: {mu10:.3f} u\n\n"
        f"**Razão ($\\rho$):** {rho:.4f}"
    )


def G(v, w, wx, is_half, is_anarm):
    """Termo vibracional G(v) em cm^-1."""
    s = v + 0.5 if is_half else v
    term = w * s
    if is_anarm:
        term -= wx * (s**2)
    return term


def nu(vu, vl, r, is_half, is_anarm):
    """Número de onda da banda (v', v'') para o isótopo de razão r."""
    return Te + G(vu, wu * r, wxu * r * r, is_half, is_anarm) - G(vl, wl * r, wxl * r * r, is_half, is_anarm)


bandas = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]

with col_plot:
    fig, ax = plt.subplots(figsize=(10, 4))

    for (vu, vl) in bandas:
        pos_11 = nu(vu, vl, 1, half_quantum, anarmonicidade)
        pos_10 = nu(vu, vl, rho, half_quantum, anarmonicidade)
        shift = pos_10 - pos_11

        ax.axvline(pos_11, color="blue", lw=2, ymin=0, ymax=0.6)
        ax.axvline(pos_10, color="red", lw=2, ymin=0, ymax=0.4)
        ax.annotate(
            f"({vu},{vl})\n$\\Delta$ = {shift:.1f}",
            xy=(pos_11, 0.7),
            xytext=(pos_10, 0.7),
            arrowprops=dict(arrowstyle="<-", color="black"),
            ha="center",
            fontsize=9,
        )

    ax.set_xlabel("Número de onda (cm$^{-1}$)", fontsize=12)
    ax.set_yticks([])
    ax.set_title("Espectro de emissão: $^{11}$BO (azul) vs $^{10}$BO (vermelho)", fontsize=14)
    ax.set_xlim(21500, 26500)

    shift_00 = nu(0, 0, rho, half_quantum, anarmonicidade) - nu(0, 0, 1, half_quantum, anarmonicidade)
    if abs(shift_00) < 0.1:
        ax.text(
            Te, 0.85,
            "Atenção: a banda (0,0) NÃO sofreu deslocamento!\nIsso contradiz os dados experimentais.",
            color="red", ha="center", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="red"),
        )
    else:
        ax.text(
            Te, 0.85,
            f"Banda (0,0) deslocada em {shift_00:.1f} cm$^{{-1}}$\n(confirma a existência da energia de ponto zero!)",
            color="green", ha="center", fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="green"),
        )

    st.pyplot(fig)

st.divider()

# ==========================================
# SEÇÃO 2: A MATEMÁTICA DO INTERCEPTO
# ==========================================
st.header(r"2. O intercepto fixo de $-9{,}1\ \mathrm{cm^{-1}}$")
st.markdown(
    "A grande sacada foi subtrair a previsão teórica *sem* o meio quantum da previsão "
    "*com* o meio quantum. Ao fazer isso, a dependência nos níveis $v'$ e $v''$ "
    "desaparece, e sobra uma constante global (um **intercepto**) igual à diferença "
    "das energias de ponto zero dos dois estados eletrônicos."
)

st.latex(
    r"\Delta\tilde{\nu}_{\text{com}} - \Delta\tilde{\nu}_{\text{sem}} "
    r"= (\rho - 1)\,\frac{\tilde{\omega}' - \tilde{\omega}''}{2} = -9{,}1\ \mathrm{cm^{-1}}"
)

dados = []
for (vu, vl) in bandas:
    pos_11 = nu(vu, vl, 1, half_quantum, anarmonicidade)
    pos_10 = nu(vu, vl, rho, half_quantum, anarmonicidade)
    shift_atual = pos_10 - pos_11

    shift_oposto = (
        nu(vu, vl, rho, not half_quantum, anarmonicidade)
        - nu(vu, vl, 1, not half_quantum, anarmonicidade)
    )

    diferenca = shift_atual - shift_oposto if half_quantum else shift_oposto - shift_atual

    dados.append(
        {
            "Banda (v', v'')": f"({vu}, {vl})",
            "Posição 11BO (cm⁻¹)": round(pos_11, 1),
            "Deslocamento atual Δ (cm⁻¹)": round(shift_atual, 1),
            "Diferença entre modelos (cm⁻¹)": round(diferenca, 1),
        }
    )

df = pd.DataFrame(dados)
st.table(df)

st.info(
    "Brinque com os botões na Seção 1. Note que a anarmonicidade altera ligeiramente "
    "as bandas altas, mas o intercepto permanece cravado em $-9{,}1\\ \\mathrm{cm^{-1}}$. "
    "O teste empírico era à prova de balas."
)

st.divider()

# ==========================================
# SEÇÃO 3: CONEXÃO COM A MECÂNICA MATRICIAL
# ==========================================
st.header("3. Onde o artigo de Heisenberg encontra o experimento")
st.markdown(
    "Em que passo da nossa dedução matricial o meio quantum entra? "
    "**Na regra de soma TRK aplicada ao nível fundamental $v = 0$.**"
)
st.latex(r"E_{e,0} - T_e = \mu\omega_e^2|q_{01}|^2 = \frac{\hbar\omega_e}{2}")
st.markdown(
    "Mulliken não sabia disso porque a matriz quântica ainda não havia sido inventada "
    "em 1924. Ele pescou empiricamente o exato meio quantum que Heisenberg deduziria "
    "puramente pela álgebra dez meses depois!"
)

st.divider()

# ==========================================
# SEÇÃO 4: O PRINCÍPIO DE FRANCK-CONDON
# ==========================================
st.header("4. O princípio de Franck–Condon e as curvas de Born–Oppenheimer")
st.markdown(
    r"""
Para fechar o quadro, precisamos entender **por que** uma única transição eletrônica
produz aquele sistema inteiro de bandas $(v', v'')$ visto na Seção 1, em vez de
apenas a banda $(0,0)$.

Isso ocorre devido ao **princípio de Franck–Condon**. Como os elétrons são quase
$20.000$ vezes mais leves que os núcleos, a transição eletrônica ocorre quase
instantaneamente. O núcleo não tem tempo de se mover. No gráfico, isso significa
que a transição deve ser desenhada como uma seta estritamente **vertical**.

*A figura abaixo é esquemática (unidades arbitrárias), mas o deslocamento entre as
curvas foi escolhido da ordem do observado no BO: $r_e$ passa de $1{,}20$ Å no estado
$X$ para $1{,}35$ Å no estado $A$, o que corresponde a um deslocamento de cerca de
três larguras da função de onda do estado fundamental.*
"""
)

fig2, ax2 = plt.subplots(figsize=(8, 6))

# Coordenada R (distância internuclear, em unidades da largura do estado fundamental)
R = np.linspace(-3, 8, 600)

# Estado inferior (X): mínimo em R = 0, k = 1
R_e_X = 0.0
k_X = 1.0
V_X = 0.5 * k_X * (R - R_e_X) ** 2

# Estado superior (A): deslocado para a direita (r_e maior) e mola mais relaxada (k menor).
# Com R_e_A = 3.2 a transição vertical a partir de R = 0 atinge a curva superior
# entre v' = 3 e v' = 4, reproduzindo o que o texto descreve.
R_e_A = 3.2
k_A = 0.6
T_e_plot = 8.0
V_A = T_e_plot + 0.5 * k_A * (R - R_e_A) ** 2

ax2.plot(R, V_X, "b-", lw=2, label=r"Estado inferior ($X\,^2\Sigma^+$)")
ax2.plot(R, V_A, "r-", lw=2, label=r"Estado superior ($A\,^2\Pi$)")

# Nível fundamental (v'' = 0) e sua densidade de probabilidade (gaussiana)
E_X_0 = 0.5 * np.sqrt(k_X)
ax2.hlines(E_X_0, -1, 1, color="blue", alpha=0.5, linestyle="--")
psi_X_0 = np.exp(-((R - R_e_X) ** 2) / np.sqrt(1 / k_X))
ax2.fill_between(R, E_X_0, E_X_0 + psi_X_0 * 1.5, color="blue", alpha=0.2)
ax2.text(-1.1, E_X_0, "$v''=0$", color="blue", va="center", ha="right")

# Níveis do estado superior
w_A = np.sqrt(k_A)
for v_prime in range(5):
    E_A_v = T_e_plot + w_A * (v_prime + 0.5)
    lim = np.sqrt(2 * (E_A_v - T_e_plot) / k_A)
    ax2.hlines(E_A_v, R_e_A - lim, R_e_A + lim, color="red", alpha=0.5, linestyle="--")
    ax2.text(R_e_A + lim + 0.1, E_A_v, f"$v'={v_prime}$", color="red", va="center")

# Seta vertical de Franck–Condon, saindo do máximo da gaussiana (R = R_e_X)
y_impacto = T_e_plot + 0.5 * k_A * (R_e_X - R_e_A) ** 2
ax2.annotate(
    "",
    xy=(R_e_X, y_impacto),
    xytext=(R_e_X, E_X_0),
    arrowprops=dict(arrowstyle="->", color="black", lw=2),
)

ax2.set_xlabel("Distância internuclear ($R - R_e''$)", fontsize=12)
ax2.set_ylabel("Energia eletrônica + vibracional", fontsize=12)
ax2.set_title("Transição vertical e a sobreposição de estados", fontsize=14)
ax2.set_ylim(0, 14)
ax2.set_xlim(-2.5, 7.5)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="lower right")

st.pyplot(fig2)

st.info(
    "Observe a seta preta partindo do centro do nível $v''=0$ (onde a probabilidade é "
    "máxima). Por causa do deslocamento horizontal das curvas, a seta cruza a curva "
    "superior entre $v'=3$ e $v'=4$. É por isso que transições como $(3,0)$ ou $(4,0)$ "
    "são muito mais intensas do que a transição $(0,0)$ pura."
)

st.divider()

# ==========================================
# SEÇÃO 5: CONCLUSÃO (FRANCK-CONDON)
# ==========================================
st.header("5. Conclusão: a física por trás do salto vertical")
st.markdown(
    """
A observação de que transições como $(3,0)$ ou $(4,0)$ costumam ser muito mais
intensas do que a transição $(0,0)$ pura revela a essência do **princípio de
Franck–Condon**.

Isso ocorre fundamentalmente por uma questão de inércia e velocidade:

* **O salto é instantâneo:** como os elétrons são milhares de vezes mais leves que os
  núcleos atômicos, a mudança na nuvem eletrônica ocorre em uma fração de
  femtossegundo. Os núcleos, pesados e lentos, não têm tempo de se mover. No gráfico,
  isso exige que a transição seja uma seta estritamente **vertical** (a distância
  internuclear $R$ fica “congelada” durante o salto).
* **O efeito “mola tensionada”:** quando a molécula passa para o estado excitado, a
  ligação química geralmente fica mais fraca e comprida (por isso a curva vermelha é
  deslocada para a direita). Ao dar o salto vertical, a molécula “acorda” no novo
  estado com os núcleos espremidos fora do seu novo ponto de equilíbrio natural.
* **A consequência vibracional:** o efeito é análogo a soltar uma mola que estava
  comprimida. A molécula não cai suavemente no nível de repouso ($v'=0$). Em vez
  disso, ela é arremessada diretamente para níveis vibracionais muito agitados
  (como $v'=3$ ou $4$).
"""
)

st.subheader("O fator de Franck–Condon")
st.markdown(
    "Matematicamente, a intensidade da linha espectral é governada pelo **fator de "
    "Franck–Condon**, que é o quadrado da integral de sobreposição das funções de onda "
    "vibracionais de cada estado:"
)

st.latex(
    r"\text{Intensidade} \propto |\langle \psi_{v'} | \psi_{v''} \rangle|^2 "
    r"= \left( \int \psi_{v'}^*(R)\, \psi_{v''}(R)\, dR \right)^2"
)

st.info(
    "Visualmente, o “sino” central da função de onda de $v''=0$ se alinha, no eixo "
    "vertical, com os picos das bordas das funções de onda de níveis mais altos (como "
    "$v'=3$ ou $4$). O alinhamento com a função de onda de $v'=0$ (que está muito à "
    "direita) é pequeno, tornando a banda $(0,0)$ fraca. É assim que a geometria do "
    "potencial dita a intensidade da luz!"
)
