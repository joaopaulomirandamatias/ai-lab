# 01 · Matemática para IA

**Meses M1–M2** · Roadmap: Nível 1 · **🔶 Gate I** no fim do M2

> Você não precisa virar matemático. Precisa conseguir entender as equações dos
> principais algoritmos e dos artigos científicos.

---

## O que dominar

**Álgebra linear** — escalares, vetores, matrizes, tensores, produto escalar, produto
matricial, transposição, inversa, determinante, norma, distância, similaridade de
cosseno, autovalores, autovetores, decomposição matricial, PCA, SVD.

Em particular, entender profundamente:

$$y = Wx + b$$

porque praticamente toda rede neural é construída sobre transformações desse tipo.

**Cálculo** — funções, limites, derivadas, derivadas parciais, regra da cadeia,
gradiente, Jacobiano, Hessiana, otimização, gradient descent.

Em particular:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L$$

A regra da cadeia é o coração do backpropagation. Se ela não estiver sólida, o M5 trava.

---

## Fonte principal

⭐ **Mathematics for Machine Learning** — Deisenroth, Faisal, Ong (grátis, `mml-book.github.io`),
capítulos 2–5. Escrito para quem vai fazer ML, não para matemáticos.

**Antes de abrir o livro:** assista 3Blue1Brown — *Essence of Linear Algebra* inteira.
Ela dá a intuição geométrica que transforma o livro de símbolos em significado.

Consulta: MIT 18.06 (Strang) · The Matrix Cookbook (identidades para derivar backprop).
Lista completa em [`recursos/livros.md`](../recursos/livros.md).

---

## Projetos

**P1 — Álgebra linear em NumPy (M1)**
Implementar do zero: produto escalar, produto matricial, norma, similaridade de cosseno,
autovalores, PCA, SVD. *Pronto quando* cada função bate com NumPy/SciPy em teste
automatizado e você explica geometricamente o que PCA faz.

**P2 — Gradient descent do zero (M2)** 🔶 **Gate I**
Regressão linear e logística com gradient descent escrito à mão. *Pronto quando* a
derivação está no papel, o código roda, e você tem um gráfico com três learning rates
diferentes — e sabe explicar cada curva.

---

## Armadilha comum

Estudar matemática "até se sentir pronto" e nunca chegar lá. Não existe esse ponto.

O critério não é sentir-se confortável — é **passar o Gate I**. Assim que você deriva e
implementa gradient descent, você tem matemática suficiente para o M3. O resto se aprende
sob demanda, quando um paper específico exigir.

Também: similaridade de cosseno parece um detalhe agora, mas é o que sustenta embeddings,
busca semântica e RAG lá no M11. Vale entender de verdade, não decorar a fórmula.
