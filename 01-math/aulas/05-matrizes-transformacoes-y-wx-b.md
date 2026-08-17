# Aula 05 — Matrizes como transformações: entendendo de verdade `y = Wx + b`

**Trilha:** Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisitos:** vetores, norma, produto escalar e similaridade de cosseno  
**Objetivo:** compreender matrizes como transformações de vetores e conectar essa interpretação diretamente às camadas de uma rede neural.

---

## 1. A equação que já apareceu desde a primeira aula

Desde o início do estudo apareceu:

\[
\boxed{y = Wx + b}
\]

Uma primeira interpretação é:

- `x` é a entrada;
- `W` contém os pesos;
- `b` é o bias;
- `y` é a saída.

Agora podemos ir além:

> **Uma matriz pode ser interpretada como uma transformação que recebe um vetor e produz outro vetor.**

Assim:

```text
vetor de entrada x
        ↓
transformação W
        ↓
       Wx
        ↓
+ deslocamento b
        ↓
        y
```

Essa estrutura aparece continuamente em redes neurais.

---

## 2. Antes da matriz: multiplicando um vetor por um escalar

Considere:

\[
x = \begin{bmatrix}2\\3\end{bmatrix}
\]

Multiplicando por 2:

\[
2x = \begin{bmatrix}4\\6\end{bmatrix}
\]

O vetor ficou maior e manteve a mesma direção. Um escalar altera o tamanho, mas uma matriz pode realizar transformações muito mais gerais.

---

## 3. Uma matriz pode transformar o espaço

Considere:

\[
W = \begin{bmatrix}2&0\\0&1\end{bmatrix},\quad
x = \begin{bmatrix}1\\1\end{bmatrix}
\]

Então:

\[
Wx =
\begin{bmatrix}2&0\\0&1\end{bmatrix}
\begin{bmatrix}1\\1\end{bmatrix}
=
\begin{bmatrix}2\\1\end{bmatrix}
\]

O vetor `(1,1)` virou `(2,1)`: a matriz esticou o espaço horizontalmente.

A mesma regra vale para qualquer vetor. Se:

\[
x = \begin{bmatrix}3\\2\end{bmatrix}
\]

então:

\[
Wx = \begin{bmatrix}6\\2\end{bmatrix}
\]

A transformação duplica a componente horizontal e preserva a vertical.

---

## 4. Pense na matriz como uma máquina

Uma representação mental útil:

```text
        W
x ─────────────→ y
```

A matriz recebe uma representação `x` e produz outra representação.

---

## 5. O que significam as linhas da matriz?

Considere:

\[
W=
\begin{bmatrix}
2&1\\
-1&3
\end{bmatrix},\quad
x=\begin{bmatrix}4\\2\end{bmatrix}
\]

A primeira saída é:

\[
2(4)+1(2)=10
\]

A segunda:

\[
-1(4)+3(2)=2
\]

Logo:

\[
Wx=\begin{bmatrix}10\\2\end{bmatrix}
\]

Cada linha de `W` realiza um produto escalar com `x`:

\[
[2,1]\cdot[4,2]=10
\]

\[
[-1,3]\cdot[4,2]=2
\]

Portanto:

\[
\boxed{\text{multiplicação matriz-vetor} = \text{vários produtos escalares}}
\]

---

## 6. Conectando com o produto escalar

Se as linhas de `W` forem `w_1,w_2,w_3,...`, então:

\[
Wx=
\begin{bmatrix}
w_1\cdot x\\
w_2\cdot x\\
w_3\cdot x\\
\vdots
\end{bmatrix}
\]

Essa interpretação é central em redes neurais.

---

## 7. Dimensões de uma matriz

Considere:

\[
W=
\begin{bmatrix}
1&2&3\\
4&5&6
\end{bmatrix}
\]

Então:

\[
W\in\mathbb{R}^{2\times3}
\]

Se:

\[
x=\begin{bmatrix}7\\8\\9\end{bmatrix}\in\mathbb{R}^{3}
\]

então:

\[
(2\times3)(3\times1)\rightarrow(2\times1)
\]

Logo:

\[
Wx\in\mathbb{R}^{2}
\]

### Regra das dimensões

\[
(m\times n)(n\times p)\rightarrow(m\times p)
\]

As dimensões internas precisam coincidir.

---

## 8. Por que erros de shape são tão comuns?

Se:

\[
W\in\mathbb{R}^{4\times3},\quad x\in\mathbb{R}^{5}
\]

teríamos:

\[
(4\times3)(5\times1)
\]

Como `3 ≠ 5`, a multiplicação não é definida.

Sempre pergunte:

> **O que cada dimensão representa?**

Depois:

> **As dimensões internas são compatíveis?**

Shapes descrevem a arquitetura matemática, não apenas um detalhe de programação.

---

## 9. O papel de `W` em uma rede neural

Se uma entrada possui três características:

\[
x\in\mathbb{R}^{3}
\]

E queremos quatro saídas, podemos usar:

\[
W\in\mathbb{R}^{4\times3}
\]

Então:

\[
Wx\in\mathbb{R}^{4}
\]

Cada linha de `W` produz uma saída.

```text
3 entradas
    ↓
matriz 4 × 3
    ↓
4 saídas
```

---

## 10. Um neurônio individual

Um neurônio recebe:

\[
x=\begin{bmatrix}x_1\\x_2\\x_3\end{bmatrix}
\]

possui pesos:

\[
w=\begin{bmatrix}w_1\\w_2\\w_3\end{bmatrix}
\]

calcula:

\[
w\cdot x=w_1x_1+w_2x_2+w_3x_3
\]

Depois soma um bias:

\[
\boxed{z=w\cdot x+b}
\]

Isso é um neurônio linear.

---

## 11. Vários neurônios juntos

Se temos quatro neurônios com pesos `w_1,...,w_4`, podemos empilhar esses vetores nas linhas de uma matriz:

\[
W=
\begin{bmatrix}
w_1^T\\
w_2^T\\
w_3^T\\
w_4^T
\end{bmatrix}
\]

Em vez de calcular cada neurônio separadamente, fazemos:

\[
\boxed{Wx}
\]

Uma operação matricial processa muitos produtos escalares de uma vez.

---

## 12. E o bias `b`?

Agora:

\[
\boxed{y=Wx+b}
\]

Suponha:

\[
Wx=\begin{bmatrix}2\\5\end{bmatrix},\quad
b=\begin{bmatrix}1\\-2\end{bmatrix}
\]

Então:

\[
y=\begin{bmatrix}3\\3\end{bmatrix}
\]

O bias desloca o resultado da transformação.

Tecnicamente:

- `Wx` é uma transformação linear;
- `Wx+b` é uma transformação afim.

Uma forma simples de guardar:

```text
Wx      → transforma
Wx + b  → transforma + desloca
```

---

## 13. Por que precisamos do bias?

Em uma função:

\[
y=wx
\]

quando `x=0`, necessariamente `y=0`. A reta passa pela origem.

Já:

\[
y=wx+b
\]

permite:

\[
x=0\Rightarrow y=b
\]

O bias aumenta a flexibilidade da transformação.

---

## 14. Exemplo numérico completo

Considere:

\[
x=\begin{bmatrix}2\\1\end{bmatrix}
\]

\[
W=
\begin{bmatrix}
0.5&1.0\\
-1.0&2.0
\end{bmatrix}
\]

\[
b=\begin{bmatrix}0.2\\0.5\end{bmatrix}
\]

Calculando:

\[
Wx=
\begin{bmatrix}
0.5(2)+1(1)\\
-1(2)+2(1)
\end{bmatrix}
=
\begin{bmatrix}2\\0\end{bmatrix}
\]

Somando o bias:

\[
y=
\begin{bmatrix}2\\0\end{bmatrix}
+
\begin{bmatrix}0.2\\0.5\end{bmatrix}
=
\boxed{\begin{bmatrix}2.2\\0.5\end{bmatrix}}
\]

### NumPy

```python
import numpy as np

x = np.array([2, 1])

W = np.array([
    [0.5, 1.0],
    [-1.0, 2.0]
])

b = np.array([0.2, 0.5])

y = W @ x + b
print(y)
```

Resultado:

```text
[2.2 0.5]
```

---

## 15. Agora imagine milhares de dimensões

Podemos ter:

\[
x\in\mathbb{R}^{4096}
\]

\[
W\in\mathbb{R}^{11008\times4096}
\]

Então:

\[
Wx\in\mathbb{R}^{11008}
\]

A matemática é a mesma da matriz `2×2`; apenas a escala mudou.

---

## 16. A importância para LLMs

Transformers usam matrizes constantemente. Mais adiante veremos expressões como:

\[
XW_Q,\quad XW_K,\quad XW_V
\]

Essas matrizes aprendidas transformam representações em Queries, Keys e Values.

Também aparecem em:

- projeções de atenção;
- feed-forward networks;
- embeddings;
- projeções de saída.

> **LLMs são intensivamente construídos sobre multiplicações de matrizes.**

---

## 17. Uma matriz pode mudar a dimensão

Se:

\[
x\in\mathbb{R}^{3}
\]

E:

\[
W\in\mathbb{R}^{2\times3}
\]

então:

\[
Wx\in\mathbb{R}^{2}
\]

Ou seja:

```text
3 dimensões
     ↓
     W
     ↓
2 dimensões
```

Se `W∈R^{10×3}`, a mesma entrada de 3 dimensões vira uma representação de 10 dimensões.

---

## 18. Matrizes como mudança de representação

Em Machine Learning, uma matriz pode converter características originais em novas representações úteis:

```text
características originais
          ↓
          W
          ↓
novas representações
```

Em visão:

```text
pixels → bordas → formas → objetos
```

Em linguagem:

```text
tokens → embeddings → representações contextuais → predição
```

---

## 19. De onde vêm os valores de `W`?

No início do treinamento, `W` começa com valores inicializados numericamente. O modelo ainda não sabe quais são os melhores pesos.

Durante o treinamento:

```text
W inicial
   ↓
predição
   ↓
erro
   ↓
gradiente
   ↓
atualização de W
   ↓
nova predição
```

Isso nos levará posteriormente ao gradient descent.

> **Grande parte do que o modelo aprende fica codificada nos valores das matrizes de pesos.**

---

## 20. Quantos parâmetros existem?

Se:

\[
W\in\mathbb{R}^{4\times3}
\]

há `4×3=12` pesos.

Se:

\[
b\in\mathbb{R}^{4}
\]

há mais quatro parâmetros.

Total:

\[
12+4=16
\]

Uma matriz `4096×4096` sozinha contém:

\[
4096^2=16\,777\,216
\]

pesos.

É assim que modelos modernos chegam rapidamente a milhões ou bilhões de parâmetros.

---

## 21. O que significa “modelo de 7 bilhões de parâmetros”?

Significa aproximadamente:

\[
7\,000\,000\,000
\]

valores ajustáveis distribuídos por matrizes, vetores de bias e outros parâmetros do modelo.

Esses números aprendidos determinam como as representações são transformadas.

---

## 22. Só `Wx+b` não é suficiente

Uma sequência formada apenas por transformações lineares/afins continua limitada. Para redes profundas aprenderem relações complexas, introduzimos funções não lineares.

Assim, uma camada típica passa a ser:

\[
\boxed{y=f(Wx+b)}
\]

onde `f` pode ser, por exemplo:

- ReLU;
- sigmoid;
- tanh;
- GELU;
- SiLU.

### Exemplo com ReLU

\[
ReLU(x)=\max(0,x)
\]

Se:

\[
z=\begin{bmatrix}2.2\\-0.5\\3.1\end{bmatrix}
\]

então:

\[
ReLU(z)=\begin{bmatrix}2.2\\0\\3.1\end{bmatrix}
\]

---

## 23. A arquitetura começa a aparecer

```text
entrada x
   ↓
W₁x + b₁
   ↓
ativação
   ↓
W₂x + b₂
   ↓
ativação
   ↓
W₃x + b₃
   ↓
saída
```

Isso já é a estrutura básica de uma rede neural multicamadas.

---

## 24. Ligando as aulas

- Aula 01 — vetores, matrizes e tensores;
- Aula 02 — norma e geometria;
- Aula 03 — produto escalar;
- Aula 04 — similaridade de cosseno;
- Aula 05 — transformações `y=Wx+b`.

A progressão pode ser resumida como:

```text
vetor x
  ↓
matriz W
  ↓
vários produtos escalares
  ↓
Wx
  ↓
bias b
  ↓
Wx + b
  ↓
ativação
  ↓
nova representação
```

---

## 25. Uma pergunta para sempre fazer

Quando encontrar algo como:

```text
W.shape = (512, 768)
x.shape = (768,)
```

pergunte:

1. O que representam as 768 dimensões?
2. Por que estamos produzindo 512 valores?
3. O que essa transformação está tentando aprender?

Não trate shapes como detalhes da biblioteca. Eles expressam a arquitetura do modelo.

---

# Exercícios da Aula 05

### 1. Escala em dois eixos

Considere:

\[
W=\begin{bmatrix}2&0\\0&3\end{bmatrix},\quad
x=\begin{bmatrix}2\\1\end{bmatrix}
\]

Calcule `Wx` e explique geometricamente o que aconteceu.

### 2. Multiplicação matriz-vetor

\[
W=\begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
x=\begin{bmatrix}2\\1\end{bmatrix}
\]

Calcule `Wx`.

### 3. Shapes

Se:

\[
W\in\mathbb{R}^{5\times3},\quad x\in\mathbb{R}^{3}
\]

qual a dimensão de `Wx`?

### 4. Compatibilidade

É possível multiplicar:

\[
W\in\mathbb{R}^{4\times5}
\]

por:

\[
x\in\mathbb{R}^{3}
\]

? Explique.

### 5. Bias

Se:

\[
Wx=\begin{bmatrix}3\\-2\end{bmatrix},\quad
b=\begin{bmatrix}1\\4\end{bmatrix}
\]

calcule `Wx+b`.

### 6. Dimensão de uma camada

Uma camada recebe 768 valores e produz 512. Qual deve ser a dimensão de `W`, usando a convenção:

\[
y=Wx
\]

?

### 7. Questão principal

Explique com suas palavras o que significa dizer:

> **“Uma matriz de pesos representa uma transformação aprendida.”**

Relacione sua resposta com:

\[
x\rightarrow Wx\rightarrow y
\]

---

## Mini desafio em NumPy

```python
import numpy as np

x = np.array([2, 1])

W = np.array([
    [0.5, 1.0],
    [-1.0, 2.0]
])

b = np.array([0.2, 0.5])

y = W @ x + b
print(y)
```

Depois altere manualmente os valores de `W` e observe como a saída muda.

Pergunta:

> **O que acontece com a transformação quando mudamos os pesos?**

---

## O que guardar

Não memorize matriz apenas como uma tabela de números.

Guarde:

\[
\boxed{\text{matriz}=\text{transformação}}
\]

E:

\[
\boxed{W=\text{parâmetros aprendíveis dessa transformação}}
\]

Assim:

\[
\boxed{y=Wx+b}
\]

significa:

> Pegue uma representação, transforme-a usando pesos aprendidos e desloque o resultado usando um bias.

---

# Próxima aula

## Aula 06 — Transformações geométricas: escala, rotação, reflexão e projeção

Vamos visualizar como matrizes conseguem:

- esticar;
- comprimir;
- rotacionar;
- refletir;
- projetar;
- deformar um espaço.

O objetivo será transformar multiplicação matricial de uma operação abstrata em algo visual e intuitivo.