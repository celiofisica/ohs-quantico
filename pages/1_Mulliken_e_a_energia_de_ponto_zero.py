"""
OHS Quântico — página 2: o efeito isotópico de Mulliken (1924) no BO
e a descoberta da energia de ponto zero.

Acompanha o exercício sobre o efeito isotópico de Mulliken do caderno de
exercícios complementar ao artigo.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
# SEÇÃO 1: A MONTAGEM EXPERIMENTAL HISTÓRICA
# ==========================================
st.header("1. Esquemático: Como mede-se a vibração de uma molécula?")
st.markdown(
    r"""
Antes que a mecânica de matrizes ditasse as regras matemáticas do universo subatômico,
a natureza já havia dado seu depoimento num laboratório de química. As bandas haviam
sido fotografadas por W. Jevons em 1915, que as atribuiu ao nitreto de boro (BN).
Mulliken primeiro reanalisou os dados de Jevons, depois refez o experimento com um
**espectrógrafo de emissão**. O processo tinha três etapas:

1. **O "Fósforo" Quântico:** Uma forte descarga elétrica em um tubo com gás Nitrogênio rasgava as moléculas, criando um plasma brilhante chamado *Nitrogênio Ativo*.
2. **A "Lenha":** Esse gás fluía para uma câmara onde se injetava Tricloreto de Boro ($BCl_3$). Traços de oxigênio no sistema reagiam formando **Monóxido de Boro ($BO$)** altamente agitado. Ao se acalmarem, essas moléculas emitiam luz.
3. **A Revelação:** A luz atravessava um prisma separando suas frequências e atingia uma chapa fotográfica.
4. **Obs:** Importante observar no que segue que número de onda é a grandeza utilizada na espectroscopia em termo de cm^{-1}
> **O Golpe de Sorte:** O Boro injetado era uma mistura natural de isótopos: o $^{11}BO$ (comum) e o $^{10}BO$ (raro). Ambos emitiam luz simultaneamente. Na chapa, para cada linha forte do isótopo pesado, surgia uma *linha fantasma* do isótopo mais leve. Foi a magnitude desse deslocamento que mostrou que o emissor era o BO, e não o BN — e, em seguida, revelou o meio quantum de energia.
    """
)


def desenhar_esquema_mulliken():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # 1. Tubo de Nitrogênio
    ax.add_patch(patches.Rectangle((0.5, 2.5), 2, 1, fill=True, color="#add8e6", ec="black"))
    ax.text(1.5, 3.0, "Descarga Elétrica\n(N₂ $\\rightarrow$ N ativo)", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(3.0, 3.0), xytext=(2.5, 3.0), arrowprops=dict(arrowstyle="->", lw=2))

    # 2. Injeção de BCl3
    ax.add_patch(patches.Rectangle((3.2, 4.0), 2.6, 0.8, fill=True, color="#90ee90", ec="black"))
    ax.text(4.5, 4.4, "BCl₃\n(+ traços de O₂)", ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(4.0, 3.5), xytext=(4.0, 4.0), arrowprops=dict(arrowstyle="->", lw=2))

    # 3. Câmara de Reação
    ax.add_patch(patches.Circle((4.0, 2.5), 1.0, fill=True, color="#ffd700", ec="orange", alpha=0.6))
    ax.text(4.0, 2.5, "Câmara de Reação\nMoléculas de BO\nemitem luz", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.annotate("", xy=(6.0, 2.5), xytext=(5.0, 2.5), arrowprops=dict(arrowstyle="->", color="#ffd700", lw=3))

    # 4. Fenda e prisma
    ax.add_patch(patches.Rectangle((6.0, 2.0), 0.2, 1.0, fill=True, color="gray"))
    ax.add_patch(patches.Polygon([[6.5, 2.0], [7.5, 2.0], [7.0, 3.0]], fill=True, color="#e0e0e0", ec="black"))
    ax.text(7.0, 1.7, "Prisma", ha="center", va="center", fontsize=10)

    # Dispersão isotópica
    ax.annotate("", xy=(8.5, 3.2), xytext=(7.3, 2.5), arrowprops=dict(arrowstyle="->", color="blue", lw=2))
    ax.annotate("", xy=(8.5, 3.0), xytext=(7.3, 2.5), arrowprops=dict(arrowstyle="->", color="red", lw=2))

    # 5. Chapa Fotográfica
    ax.add_patch(patches.Rectangle((8.5, 1.0), 0.5, 3.0, fill=True, color="#f5f5f5", ec="black"))
    ax.text(8.75, 4.3, "Chapa Fotográfica\n(Linhas Isotópicas)", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.plot([8.5, 9.0], [3.2, 3.2], color="blue", lw=3)
    ax.text(9.2, 3.2, "¹¹BO (principal)", va="center", fontsize=9, color="blue")
    ax.plot([8.5, 9.0], [3.0, 3.0], color="red", lw=2)
    ax.text(9.2, 3.0, "¹⁰BO (fantasma)", va="center", fontsize=9, color="red")

    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")   # mantém a câmara circular
    return fig


fig_esquema = desenhar_esquema_mulliken()
st.pyplot(fig_esquema)

st.divider()

# ==========================================
# SEÇÃO 2: A ALAVANCA ISOTÓPICA E O ESPECTRO
# ==========================================
st.header("2. O experimento interativo")
st.markdown(
    r"Use os controles abaixo para simular a separação das linhas na chapa fotográfica. "
    r"A antiga teoria de Bohr assumia que a energia vibracional era um número inteiro "
    r"de quanta ($E = v\hbar\omega$). A nova teoria exigia o meio quantum "
    r"($E = (v + 1/2)\hbar\omega$)."
)

col_ctrl, col_plot = st.columns([1, 2.5])

with col_ctrl:
    st.subheader("Controles da teoria")
    half_quantum = st.toggle("Ativar energia de ponto zero (+ 1/2)", value=True)
    anarmonicidade = st.toggle("Incluir anarmonicidade (refinamento)", value=False)

    st.markdown("**Relação de massas:**")
    st.latex(r"\rho = \sqrt{\frac{\mu(^{11}\text{BO})}{\mu(^{10}\text{BO})}}")

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
# SEÇÃO 3: DO HAMILTONIANO AO INTERCEPTO
# ==========================================
st.header(r"3. O intercepto fixo de $-9{,}1\ \mathrm{cm^{-1}}$")

st.subheader("De onde vem a fórmula: o hamiltoniano da molécula")
st.markdown(
    r"""
Antes da fórmula do intercepto, vale ver de onde ela sai. Uma molécula diatômica $AB$
é um sistema de dois núcleos (massas $M_A$, $M_B$) e $n$ elétrons (massa $m_e$).
O hamiltoniano completo tem três partes:
"""
)
st.latex(
    r"H = \underbrace{-\frac{\hbar^2}{2M_A}\nabla_A^2 - \frac{\hbar^2}{2M_B}\nabla_B^2}_{T_N\ (\text{núcleos})}"
    r"\;\underbrace{-\sum_{i=1}^{n}\frac{\hbar^2}{2m_e}\nabla_i^2}_{T_e\ (\text{elétrons})}"
    r"\;+\;\underbrace{V_{ee} + V_{eN} + V_{NN}}_{\text{Coulomb}}"
)
st.markdown(
    r"""
**Passo 1 — Born–Oppenheimer.** Como $m_e/M \sim 10^{-4}$, os núcleos são lentos diante
dos elétrons. Na primeira etapa o termo $T_N$ é **retirado** do problema — os núcleos
ficam congelados a uma distância $R$ — e resolve-se só a parte eletrônica:
"""
)
st.latex(
    r"\big[\,T_e + V_{ee} + V_{eN} + V_{NN}\,\big]\,\psi_{\rm el}(\mathbf r;R) "
    r"= E_{\rm el}(R)\,\psi_{\rm el}(\mathbf r;R)"
)
st.markdown(
    r"""
O autovalor $E_{\rm el}(R)$ é a **curva de Born–Oppenheimer** (as curvas da Seção 5).
Repare no que *não* aparece nela: as massas nucleares. $M_A$ e $M_B$ só entravam em
$T_N$, que foi posto de lado. Logo a curva $E_{\rm el}(R)$, o seu mínimo $T_e$ e a sua
curvatura são **idênticos** para $^{10}$BO e $^{11}$BO. É por isso que $T_e$ vai sumir do
deslocamento isotópico.

**Passo 2 — os núcleos voltam.** $T_N$ não é desprezível — é ele que faz a molécula
vibrar; ele só foi tratado em separado. Separando o centro de massa, o movimento relativo
dos núcleos obedece a
"""
)
st.latex(
    r"H_N = -\frac{\hbar^2}{2\mu}\frac{d^2}{dR^2} + E_{\rm el}(R), "
    r"\qquad \mu = \frac{M_A M_B}{M_A + M_B}"
)
st.markdown(
    r"""
(a rotação foi omitida: ela produz a estrutura fina de cada banda, não os deslocamentos
entre bandas de que tratamos aqui). Toda a dependência isotópica está em um único lugar:
a massa reduzida $\mu$.

**Passo 3 — aproximação harmônica.** Perto do mínimo, expande-se a curva em série de
Taylor e fica-se com o termo quadrático:
"""
)
st.latex(r"E_{\rm el}(R) \simeq T_e + \tfrac12\,k\,(R-R_e)^2, \qquad k = E_{\rm el}''(R_e)")
st.markdown(
    r"""
e $H_N$ vira exatamente o oscilador harmônico do artigo, com $\omega_e = \sqrt{k/\mu}$.
A mecânica matricial dá então $E_v = T_e + \hbar\omega_e\,(v+\tfrac12)$ — e é aqui que o
meio quantum entra (Seção 4). Como $k$ é o mesmo para os dois isótopos, trocar $\mu$ por
$\mu^i$ só reescala a frequência:
"""
)
st.latex(r"\omega_e^{\,i} = \rho\,\omega_e, \qquad \rho = \sqrt{\mu/\mu^i}")
st.markdown(
    r"""
Os termos cúbico e quártico da expansão dão a correção anarmônica
$-\omega_e x_e\,(v+\tfrac12)^2$, com $\omega_e x_e \propto 1/\mu$ — por isso ela escala
com $\rho^2$ (é o que o botão "anarmonicidade" liga).

**Passo 4 — a banda e o deslocamento.** Em números de onda, a banda $(v',v'')$ de emissão
fica em $\tilde\nu = T_e + G'(v') - G''(v'')$, com $G(v) = \tilde\omega_e\,(v+\tfrac12)$.
Subtraindo a posição no $^{11}$BO da posição no $^{10}$BO, $T_e$ cancela e sobra
"""
)
st.latex(
    r"\Delta\tilde\nu = (\rho - 1)\Big[\tilde\omega'\,(v'+\tfrac12) - \tilde\omega''\,(v''+\tfrac12)\Big]"
)
st.markdown(
    r"""
Sem o meio quantum (teoria de Bohr) seria $(\rho-1)\,[\tilde\omega' v' - \tilde\omega'' v'']$,
e a banda $(0,0)$ não se deslocaria nada.
"""
)

st.subheader("O intercepto")
st.markdown(
    "A grande sacada de Mulliken foi comparar o deslocamento isotópico previsto "
    "**sem** o meio quantum (Teoria de Bohr) com o deslocamento **com** o meio quantum (o real). "
    "Note na tabela abaixo como a diferença entre os dois modelos anula a dependência "
    "dos níveis de energia e sobra apenas uma constante global (o **intercepto**)."
)

st.latex(
    r"\Delta\tilde{\nu}_{\text{com}} - \Delta\tilde{\nu}_{\text{sem}} "
    r"= (\rho - 1)\,\frac{\tilde{\omega}' - \tilde{\omega}''}{2} = -9{,}1\ \mathrm{cm^{-1}}"
)

dados = []
for (vu, vl) in bandas:
    pos_11_com = nu(vu, vl, 1, True, anarmonicidade)
    pos_10_com = nu(vu, vl, rho, True, anarmonicidade)
    shift_com = pos_10_com - pos_11_com

    pos_11_sem = nu(vu, vl, 1, False, anarmonicidade)
    pos_10_sem = nu(vu, vl, rho, False, anarmonicidade)
    shift_sem = pos_10_sem - pos_11_sem

    intercepto = shift_com - shift_sem

    dados.append(
        {
            "Banda (v', v'')": f"({vu}, {vl})",
            "Deslocamento SEM 1/2 quantum (cm⁻¹)": round(shift_sem, 1),
            "Deslocamento COM 1/2 quantum (cm⁻¹)": round(shift_com, 1),
            "A Diferença (Intercepto em cm⁻¹)": round(intercepto, 1),
        }
    )

df = pd.DataFrame(dados)
st.table(df)

st.info(
    "Brinque com os botões na Seção 2. No modelo harmônico o intercepto é exatamente o "
    "mesmo para todas as bandas; com a anarmonicidade ligada ele oscila entre cerca de "
    "$-8{,}4$ e $-10{,}4\\ \\mathrm{cm^{-1}}$, porque a energia de ponto zero ganha a parcela "
    "$-\\omega_e x_e/4$ e o termo cruzado $(\\rho^2-1)\\,[\\omega_e x_e'(v'+\\tfrac14) - "
    "\\omega_e x_e''(v''+\\tfrac14)]$ depende fracamente da banda. "
    "Como a teoria antiga preveria zero de deslocamento para a banda (0,0), esse "
    "intercepto fixo foi a prova física de que a molécula vibra mesmo no seu estado "
    "de menor energia."
)

st.divider()

# ==========================================
# SEÇÃO 4: CONEXÃO COM A MECÂNICA MATRICIAL
# ==========================================
st.header("4. Onde o artigo de Heisenberg encontra o experimento")
st.markdown(
    "Em que passo da nossa dedução matricial o meio quantum entra? "
    "**Na regra de soma TRK aplicada ao nível fundamental $v = 0$.**"
)
st.latex(r"E_{e,0} - T_e = \mu\omega_e^2|q_{01}|^2 = \frac{\hbar\omega_e}{2}")
st.markdown(
    "Mulliken não sabia disso porque a matriz quântica ainda não havia sido inventada "
    "em 1924. Ele pescou empiricamente o exato meio quantum que Heisenberg deduziria "
    "puramente pela álgebra não-comutativa dez meses depois!"
)

st.divider()

# ==========================================
# SEÇÃO 5: O PRINCÍPIO DE FRANCK-CONDON
# ==========================================
st.header("5. O princípio de Franck–Condon e as curvas de Born–Oppenheimer")
st.markdown(
    r"""
Para fechar o quadro, precisamos entender **por que** uma única transição eletrônica
produz aquele sistema inteiro de bandas $(v', v'')$ visto na Seção 2, em vez de
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

R = np.linspace(-3, 8, 600)

R_e_X = 0.0
k_X = 1.0
V_X = 0.5 * k_X * (R - R_e_X) ** 2

# Com R_e_A = 3.2 a transição vertical a partir de R = 0 atinge a curva superior
# entre v' = 3 e v' = 4, reproduzindo o que o texto descreve.
R_e_A = 3.2
k_A = 0.6
T_e_plot = 8.0
V_A = T_e_plot + 0.5 * k_A * (R - R_e_A) ** 2

ax2.plot(R, V_X, "b-", lw=2, label=r"Estado inferior ($X\,^2\Sigma^+$)")
ax2.plot(R, V_A, "r-", lw=2, label=r"Estado superior ($A\,^2\Pi$)")

E_X_0 = 0.5 * np.sqrt(k_X)
ax2.hlines(E_X_0, -1, 1, color="blue", alpha=0.5, linestyle="--")
psi_X_0 = np.exp(-((R - R_e_X) ** 2) / np.sqrt(1 / k_X))
ax2.fill_between(R, E_X_0, E_X_0 + psi_X_0 * 1.5, color="blue", alpha=0.2)
ax2.text(-1.1, E_X_0, "$v''=0$", color="blue", va="center", ha="right")

w_A = np.sqrt(k_A)
for v_prime in range(5):
    E_A_v = T_e_plot + w_A * (v_prime + 0.5)
    lim = np.sqrt(2 * (E_A_v - T_e_plot) / k_A)
    ax2.hlines(E_A_v, R_e_A - lim, R_e_A + lim, color="red", alpha=0.5, linestyle="--")
    ax2.text(R_e_A + lim + 0.1, E_A_v, f"$v'={v_prime}$", color="red", va="center")

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
# SEÇÃO 6: CONCLUSÃO (FRANCK-CONDON)
# ==========================================
st.header("6. Conclusão: a física por trás do salto vertical")
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

st.divider()

# ==========================================
# SEÇÃO 7: REFERÊNCIAS
# ==========================================
st.header("7. Referência")
st.markdown(
    """
* MULLIKEN R. S. **Isotope Effects in the Band Spectra of Boron Monoxide and Silicon Nitride.** *Nature*, v. 113, p. 423–424 (1924).    """
)
