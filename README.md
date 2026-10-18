# OHS Quântico

Aplicativo interativo que acompanha o artigo **sobre o oscilador harmônico simples
(OHS) na mecânica matricial de Heisenberg** (*Revista Brasileira de Ensino de Física*,
2026). Construído em [Streamlit](https://streamlit.io) com NumPy, SymPy e Matplotlib.

- **App ao vivo:** <https://ohs-quantico-bwvgessytkhp5ywweadv6x.streamlit.app/>
  (hospedagem gratuita; o app pode levar alguns segundos para “acordar”)
- **Versão arquivada (DOI):** *inserir o DOI do Zenodo após a release v1.0.0*

Todo o código usa unidades naturais ($\hbar = m = \omega = 1$).

## O que o app faz

### Página principal — *Oscilador Harmônico Quântico*

1. **Abordagem matricial truncada (NumPy).** Segue a lógica do artigo: as únicas
   entradas são as amplitudes $|X_{n,n+1}|^2 = (n+1)/2$, fixadas pela regra de soma
   de Thomas–Reiche–Kuhn (TRK), e a dinâmica $P_{rs} = i\,(E_r - E_s)\,X_{rs}$.
   O comutador $[X,P]$ é então *calculado* para uma matriz $N\times N$ (N ajustável),
   e o app exibe o **paradoxo do traço**: como $\mathrm{Tr}(XP - PX) = 0$ em dimensão
   finita, o último elemento da diagonal vale $-(N-1)\,i$ em vez de $i$. A mesma
   anomalia aparece na soma TRK do último nível.
2. **Verificação algébrica em dimensão infinita (SymPy).** Com operadores bosônicos
   simbólicos, $[X,P] = i$ sai exato. O app deixa explícito que esta seção *verifica*
   a consistência da álgebra a partir de $[a,a^\dagger]=1$ — não é uma dedução
   independente do comutador.
3. **Para comparação: a imagem de Schrödinger.** Potencial, níveis e $|\psi_n(x)|^2$,
   lembrando que nada nas seções anteriores usou funções de onda.

### Página 2 — *Mulliken e a energia de ponto zero*

Acompanha o exercício sobre o efeito isotópico de Mulliken (1924) no monóxido de
boro, do caderno de exercícios complementar ao artigo. Permite ligar e desligar o
meio quantum e a anarmonicidade e ver o deslocamento isotópico das bandas
$^{11}$BO/$^{10}$BO, o intercepto fixo de $-9{,}1\ \mathrm{cm^{-1}}$ e a conexão com
a regra TRK no nível fundamental. Termina com uma ilustração esquemática do princípio
de Franck–Condon.

Constantes espectroscópicas do $^{11}$BO (sistema $\alpha$, $A\,^2\Pi \to X\,^2\Sigma^+$):
$T_e = 23958{,}7$, $\omega_e'' = 1885{,}69$, $\omega_e x_e'' = 11{,}81$,
$\omega_e' = 1260{,}70$, $\omega_e x_e' = 11{,}16$ (todos em cm$^{-1}$).

## Como rodar localmente

```bash
git clone https://github.com/celiofisica/ohs-quantico.git
cd ohs-quantico
python -m venv .venv && source .venv/bin/activate   # opcional
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501`. A segunda página aparece no menu lateral.
Testado com Python 3.12 e as versões fixadas em `requirements.txt`.

## Estrutura

```
app.py                                   # página principal (OHS: NumPy, SymPy, Schrödinger)
pages/1_Mulliken_e_a_energia_de_ponto_zero.py
requirements.txt                         # dependências com versões fixadas
CITATION.cff                             # metadados de citação
LICENSE                                  # MIT
```

## Material relacionado

- Notas de aula: <https://doi.org/10.5281/zenodo.22285340>
- Caderno de exercícios com soluções: *inserir DOI do Zenodo*

## Como citar

Veja `CITATION.cff` (o GitHub mostra o botão *Cite this repository*). Em texto:

> MARQUES, C. *OHS Quântico: o oscilador harmônico na mecânica matricial de
> Heisenberg — aplicativo interativo*, versão 1.0.0. Zenodo, 2026.
> DOI: *inserir após a release*.

## Licença

[MIT](LICENSE).
