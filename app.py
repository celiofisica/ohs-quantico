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

# Aviso didático sobre unidades
st.info("**Nota sobre unidades:** Por convenção e para simplificar a notação matemática, em todo este código estamos adotando o sistema de *unidades naturais*. Isso significa que a constante de Planck reduzida, a massa da partícula e a frequência do oscilador são tomadas como iguais a 1 ($\hbar = m = \omega = 1$).")

st.divider()

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
    coeffs = [0] * n + [1]
    H_n = np.polynomial.hermite.hermval(x_vals, coeffs)
    norm = 1.0 / np.sqrt((2**n) * math.factorial(n) * np.sqrt(np.pi))
    return norm * np.exp(-x_vals**2 / 2) * H_n

for n_val in range(N_plot):
    E_val = n_val + 0.5
    x_turn = np.sqrt(2 * E_val)
    
    ax.hlines(E_val, -x_turn, x_turn, color='gray', lw=1, linestyle='--')
    ax.text(x_turn + 0.2, E_val, f"$E_{n_val}$", va='center', color='blue', fontweight='bold')
    
    prob_density = np.abs(psi_n(n_val, x))**2
    scale = 0.8 / np.max(prob_density) 
    curva = E_val + prob_density * scale
    
    ax.plot(x, curva, color='blue', lw=1.5)
    ax.fill_between(x, E_val, curva, color='blue', alpha=0.2)
    
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

st.info("O último nível sofre um colapso drástico para forçar a soma da diagonal a ser zero.")

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
# SEÇÃO 3: SYMPY (ÁLGEBRA SIMBÓLICA)
# ==========================================
st.header("3. Abordagem Analítica (SymPy)")

st.markdown("""
**O que é o SymPy?**  
Para um computador tradicional, resolver matemática quase sempre significa fazer contas com matrizes numéricas e aproximações. O **SymPy** (de *Symbolic Python*) muda essa lógica: ele ensina o computador a manipular a álgebra simbólica pura (usando letras como $x, p, a, a^\dagger$) exatamente como fazemos no papel, assumindo um espaço de dimensão infinita.
""")

a = BosonOp('a')
X_sym = (a + Dagger(a)) / sp.sqrt(2)
P_sym = sp.I * (Dagger(a) - a) / sp.sqrt(2)

comutador_XP = Commutator(X_sym, P_sym)
resultado = comutador_XP.doit().expand()

# Exibindo o resultado final explícito conectando com a Física:
st.latex(r"[X, P] = " + sp.latex(resultado) + r" \equiv i\hbar \quad (\text{pois } \hbar = 1)")

st.success("Perceba a mágica: como o SymPy entende a regra algébrica no infinito, chegamos ao resultado exato da mecânica quântica teórica ($i\hbar$), perfeitamente livre das anomalias na fronteira vistas nas matrizes finitas!")
# Configuração da página
st.set_page_config(page_title="O Efeito Isotópico de Mulliken (1924)", layout="wide")

st.title("A Descoberta da Energia de Ponto Zero (1924)")
st.markdown("""
Dez meses antes de Heisenberg formular a Mecânica Matricial, R. S. Mulliken analisou a luz emitida pelo monóxido de boro (BO). 
Ele usou uma "alavanca" fornecida pela própria natureza: os isótopos. A molécula de $^{10}\text{BO}$ tem a mesma nuvem eletrônica e a mesma "mola" da $^{11}\text{BO}$, mas é ligeiramente mais leve. 
Isso altera as frequências de vibração. A forma como o espectro se desloca revelou o meio quantum ($\hbar\omega/2$) escondido no estado fundamental.
""")

st.divider()

# ==========================================
# PARÂMETROS FÍSICOS (BO)
# ==========================================
mB10, mB11, mO = 10.0129, 11.0093, 15.9949
mu10 = mB10 * mO / (mB10 + mO)
mu11 = mB11 * mO / (mB11 + mO)
rho = np.sqrt(mu11 / mu10) 

wl, wxl = 1885.69, 11.81    
wu, wxu = 1260.70, 11.16    
Te = 23958.7                

# ==========================================
# SEÇÃO 1: A ALAVANCA ISOTÓPICA E O ESPECTRO
# ==========================================
st.header("1. O Experimento Interativo")
st.markdown("Use os controles abaixo para testar as duas hipóteses da física em 1924. A antiga teoria de Bohr assumia que a energia vibracional era um número inteiro ($E = v\hbar\omega$). A nova teoria exigia o meio quantum ($E = (v + 1/2)\hbar\omega$).")

col_ctrl, col_plot = st.columns([1, 2.5])

with col_ctrl:
    st.subheader("Controles da Teoria")
    half_quantum = st.toggle("Ativar Energia de Ponto Zero (+ 1/2)", value=True)
    anarmonicidade = st.toggle("Incluir Anarmonicidade (Refinamento)", value=False)
    
    st.info(f"**Massa Reduzida ($\\mu$):**\n\n$^{{11}}$BO: {mu11:.3f} u\n\n$^{{10}}$BO: {mu10:.3f} u\n\n**Razão ($\\rho$):** {rho:.4f}")

def G(v, w, wx, is_half, is_anarm):
    s = v + 0.5 if is_half else v
    term = w * s
    if is_anarm:
        term -= wx * (s ** 2)
    return term

def nu(vu, vl, r, is_half, is_anarm):
    return Te + G(vu, wu*r, wxu*r*r, is_half, is_anarm) - G(vl, wl*r, wxl*r*r, is_half, is_anarm)

bandas = [(0,0), (1,0), (0,1), (1,1), (2,0)]

with col_plot:
    fig, ax = plt.subplots(figsize=(10, 4))
    
    for (vu, vl) in bandas:
        pos_11 = nu(vu, vl, 1, half_quantum, anarmonicidade)
        pos_10 = nu(vu, vl, rho, half_quantum, anarmonicidade)
        shift = pos_10 - pos_11
        
        ax.axvline(pos_11, color='blue', lw=2, ymin=0, ymax=0.6)
        ax.axvline(pos_10, color='red', lw=2, ymin=0, ymax=0.4)
        ax.annotate(f"({vu},{vl})\n$\Delta$ = {shift:.1f}", 
                    xy=(pos_11, 0.7), xytext=(pos_10, 0.7),
                    arrowprops=dict(arrowstyle="<-", color='black'),
                    ha='center', fontsize=9)

    ax.set_xlabel("Número de Onda (cm$^{-1}$)", fontsize=12)
    ax.set_yticks([]) 
    ax.set_title("Espectro de Emissão: $^{11}$BO (Azul) vs $^{10}$BO (Vermelho)", fontsize=14)
    ax.set_xlim(21500, 26500)
    
    shift_00 = nu(0, 0, rho, half_quantum, anarmonicidade) - nu(0, 0, 1, half_quantum, anarmonicidade)
    if abs(shift_00) < 0.1:
        ax.text(Te, 0.85, "Atenção: A banda (0,0) NÃO sofreu deslocamento!\nIsso contradiz os dados experimentais.", 
                color='red', ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    else:
        ax.text(Te, 0.85, f"Banda (0,0) deslocada em {shift_00:.1f} cm$^{{-1}}$\n(Confirma a existência da energia de ponto zero!)", 
                color='green', ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='green'))

    st.pyplot(fig)

st.divider()

# ==========================================
# SEÇÃO 2: A MATEMÁTICA DO INTERCEPTO
# ==========================================
st.header("2. O Intercepto Fixo de $-9.1 \text{ cm}^{-1}$")
st.markdown("A grande sacada foi subtrair a previsão teórica *sem* o meio quantum da previsão *com* o meio quantum. Ao fazer isso, a dependência nos níveis $v'$ e $v''$ desaparece, e sobra uma constante global (um **intercepto**) igual à diferença das energias de ponto zero dos dois estados eletrônicos.")

st.latex(r"\Delta\tilde{\nu}_{\text{com}} - \Delta\tilde{\nu}_{\text{sem}} = (\rho - 1)\frac{\tilde{\omega}' - \tilde{\omega}''}{2} = -9.1 \text{ cm}^{-1}")

dados = []
for (vu, vl) in bandas:
    pos_11 = nu(vu, vl, 1, half_quantum, anarmonicidade)
    pos_10 = nu(vu, vl, rho, half_quantum, anarmonicidade)
    shift_atual = pos_10 - pos_11
    
    shift_oposto = (nu(vu, vl, rho, not half_quantum, anarmonicidade) - 
                    nu(vu, vl, 1, not half_quantum, anarmonicidade))
    
    diferenca = shift_atual - shift_oposto if half_quantum else shift_oposto - shift_atual
    
    dados.append({
        "Banda (v', v'')": f"({vu}, {vl})",
        "Posição 11BO (cm⁻¹)": round(pos_11, 1),
        "Deslocamento Atual Δ (cm⁻¹)": round(shift_atual, 1),
        "Diferença entre Modelos (cm⁻¹)": round(diferenca, 1)
    })

df = pd.DataFrame(dados)
st.table(df)

st.info("Brinque com os botões na Seção 1. Note que a anarmonicidade altera ligeiramente as bandas altas, mas o **intercepto** permanece cravado em $\approx -9.1 \text{ cm}^{-1}$. O teste empírico era à prova de balas.")

st.divider()

# ==========================================
# SEÇÃO 3: CONEXÃO COM A MECÂNICA MATRICIAL
# ==========================================
st.header("3. Onde o Artigo de Heisenberg Encontra o Experimento")
st.markdown("Em que passo da nossa dedução matricial o meio quantum entra? **Na regra de soma TRK aplicada ao nível fundamental $v = 0$.**")
st.latex(r"E_{e,0} - T_e = \mu\omega_e^2|q_{01}|^2 = \frac{\hbar\omega_e}{2}")
st.markdown("Mulliken não sabia disso porque a matriz quântica ainda não havia sido inventada em 1924. Ele pescou empiricamente o exato meio quantum que Heisenberg deduziria puramente pela álgebra dez meses depois!")

st.divider()

# ==========================================
# SEÇÃO 4: O PRINCÍPIO DE FRANCK-CONDON (NOVO)
# ==========================================
st.header("4. O Princípio de Franck-Condon e as Curvas de Born-Oppenheimer")
st.markdown("""
Para fechar o quadro, precisamos entender **por que** uma única transição eletrônica produz aquele sistema inteiro de bandas $(v', v'')$ visto na Seção 1, em vez de apenas a banda $(0,0)$.

Isso ocorre devido ao **Princípio de Franck-Condon**. Como os elétrons são quase $20.000$ vezes mais leves que os núcleos, a transição eletrônica ocorre quase instantaneamente. O núcleo não tem tempo de se mover. No gráfico, isso significa que a transição deve ser desenhada como uma seta estritamente **vertical**.
""")

fig2, ax2 = plt.subplots(figsize=(8, 6))

# Coordenada R (distância internuclear genérica)
R = np.linspace(-2.5, 3.5, 400)

# Parâmetros simulados para o Estado Inferior (X)
R_e_X = 0
k_X = 1.0
V_X = 0.5 * k_X * (R - R_e_X)**2

# Parâmetros simulados para o Estado Superior (A)
# Deslocado para a direita (R_e maior) e mola mais relaxada (k menor)
R_e_A = 1.0
k_A = 0.6
T_e_plot = 8
V_A = T_e_plot + 0.5 * k_A * (R - R_e_A)**2

# Desenhando as curvas de potencial
ax2.plot(R, V_X, 'b-', lw=2, label="Estado Inferior ($X\,^2\Sigma^+$)")
ax2.plot(R, V_A, 'r-', lw=2, label="Estado Superior ($A\,^2\Pi$)")

# Nível fundamental (v''=0) e sua onda de probabilidade (Gaussiana)
E_X_0 = 0.5 * np.sqrt(k_X)
ax2.hlines(E_X_0, -1, 1, color='blue', alpha=0.5, linestyle='--')
psi_X_0 = np.exp(- (R - R_e_X)**2 / np.sqrt(1/k_X)) 
ax2.fill_between(R, E_X_0, E_X_0 + psi_X_0 * 1.5, color='blue', alpha=0.2)
ax2.text(-1.1, E_X_0, "$v''=0$", color='blue', va='center', ha='right')

# Níveis do estado superior
w_A = np.sqrt(k_A)
for v_prime in range(5):
    E_A_v = T_e_plot + w_A * (v_prime + 0.5)
    lim = np.sqrt(2 * (E_A_v - T_e_plot) / k_A)
    ax2.hlines(E_A_v, R_e_A - lim, R_e_A + lim, color='red', alpha=0.5, linestyle='--')
    ax2.text(R_e_A + lim + 0.1, E_A_v, f"$v'={v_prime}$", color='red', va='center')

# A Seta Vertical de Franck-Condon (saindo do máximo da gaussiana)
y_impacto = T_e_plot + 0.5 * k_A * (R_e_X - R_e_A)**2
ax2.annotate("", xy=(R_e_X, y_impacto), xytext=(R_e_X, E_X_0),
             arrowprops=dict(arrowstyle="->", color='black', lw=2))

ax2.set_xlabel("Distância Internuclear ($R - R_e''$)", fontsize=12)
ax2.set_ylabel("Energia Eletrônica + Vibracional", fontsize=12)
ax2.set_title("Transição Vertical e a Sobreposição de Estados", fontsize=14)
ax2.set_ylim(0, 14)
ax2.set_xlim(-2, 3)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='lower right')

st.pyplot(fig2)

st.info("Observe a seta preta partindo do centro do nível $v''=0$ (onde a probabilidade é máxima). Por causa do deslocamento horizontal das curvas, a seta cruza a curva superior atingindo os níveis altos (perto de $v'=3$ e $4$). É por isso que transições como $(3,0)$ ou $(4,0)$ são frequentemente muito mais intensas do que a transição $(0,0)$ pura!")
