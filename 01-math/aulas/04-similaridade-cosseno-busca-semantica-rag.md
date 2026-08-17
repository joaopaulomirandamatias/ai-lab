# Aula 04 — Similaridade de cosseno na prática: embeddings, busca semântica e RAG

**Trilha:** 01 · Matemática para IA  
**Etapa:** Álgebra Linear  
**Pré-requisito:** Aula 03 — Produto escalar  
**Data de estudo:** 2026-08-17  
**Objetivo:** entender como comparar a direção de vetores e como essa ideia aparece em embeddings, busca vetorial e sistemas RAG.

---

## 1. De onde estamos partindo?

Na aula anterior vimos que o produto escalar entre dois vetores pode ser escrito de duas maneiras:

$$
a\cdot b = \sum_{i=1}^{n}a_i b_i
$$

e geometricamente:

$$
a\cdot b = \|a\|\|b\|\cos(\theta)
$$

Essa segunda forma é especialmente importante. Ela nos diz que o produto escalar depende de três coisas:

- magnitude de $a$;
- magnitude de $b$;
- ângulo entre eles.

Mas surge um problema: se queremos comparar **significado**, talvez o tamanho dos vetores não seja o mais importante. Queremos responder algo como:

> **Os dois vetores apontam para direções semelhantes?**

É exatamente isso que a similaridade de cosseno nos ajuda a medir.

---

## 2. Eliminando a magnitude

Partimos de:

$$
a\cdot b = \|a\|\|b\|\cos(\theta)
$$

Dividimos os dois lados por $\|a\|\|b\|$:

$$
\boxed{\cos(\theta)=\frac{a\cdot b}{\|a\|\|b\|}}
$$

Essa expressão é chamada de **similaridade de cosseno** (*cosine similarity*).

---

## 3. O que estamos medindo?

Considere:

$$
a=\begin{bmatrix}1\\2\end{bmatrix}
$$

$$
b=\begin{bmatrix}10\\20\end{bmatrix}
$$

O vetor $b$ é dez vezes maior:

$$
b=10a
$$

As magnitudes são muito diferentes, mas os dois apontam exatamente para a mesma direção. Assim:

$$
\theta=0^\circ
$$

$$
\cos(0^\circ)=1
$$

Logo:

$$
\boxed{similaridade(a,b)=1}
$$

Apesar das magnitudes diferentes.

---

## 4. Interpretando o resultado

A similaridade de cosseno está relacionada ao intervalo:

$$
-1\leq \cos(\theta)\leq1
$$

| Similaridade | Interpretação geométrica |
|---:|---|
| $1$ | mesma direção |
| próxima de $1$ | direções muito semelhantes |
| $0$ | vetores perpendiculares |
| negativa | direções opostas |
| $-1$ | direções exatamente opostas |

Para embeddings, normalmente estamos interessados em encontrar os vetores com **maior similaridade**.

---

## 5. Primeiro exemplo completo

Considere:

$$
a=\begin{bmatrix}1\\2\end{bmatrix}
$$

$$
b=\begin{bmatrix}2\\4\end{bmatrix}
$$

Produto escalar:

$$
a\cdot b=1(2)+2(4)=10
$$

Normas:

$$
\|a\|=\sqrt{1^2+2^2}=\sqrt5
$$

$$
\|b\|=\sqrt{2^2+4^2}=\sqrt{20}
$$

Portanto:

$$
sim(a,b)=\frac{10}{\sqrt5\sqrt{20}}
$$

Como:

$$
\sqrt5\sqrt{20}=\sqrt{100}=10
$$

Temos:

$$
\boxed{sim(a,b)=1}
$$

Os vetores têm exatamente a mesma direção.

---

## 6. Vetores perpendiculares

Considere:

$$
a=\begin{bmatrix}1\\0\end{bmatrix}
$$

$$
b=\begin{bmatrix}0\\1\end{bmatrix}
$$

Produto escalar:

$$
a\cdot b=0
$$

Logo:

$$
sim(a,b)=0
$$

O ângulo é $90^\circ$ e $\cos(90^\circ)=0$.

---

## 7. Vetores opostos

Considere:

$$
a=\begin{bmatrix}1\\0\end{bmatrix}
$$

$$
b=\begin{bmatrix}-1\\0\end{bmatrix}
$$

O ângulo é $180^\circ$, então:

$$
\cos(180^\circ)=-1
$$

Portanto:

$$
\boxed{sim(a,b)=-1}
$$

---

## 8. Implementando em NumPy

```python
import numpy as np

a = np.array([1, 2])
b = np.array([2, 4])

dot = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)

similaridade = dot / (norm_a * norm_b)

print(similaridade)
```

Resultado:

```text
1.0
```

Podemos transformar isso em função:

```python
def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )
```

---

## 9. O que acontece se normalizarmos antes?

Na Aula 02 aprendemos:

$$
\hat a=\frac{a}{\|a\|}
$$

$$
\hat b=\frac{b}{\|b\|}
$$

Depois da normalização:

$$
\|\hat a\|=1
$$

$$
\|\hat b\|=1
$$

Então:

$$
sim(a,b)=\hat a\cdot\hat b
$$

> **Quando os vetores já estão normalizados, o produto escalar é igual à similaridade de cosseno.**

Isso ajuda a explicar por que muitos sistemas vetoriais normalizam embeddings antes da busca.

---

## 10. Entrando nos embeddings

Suponha, apenas para fins didáticos:

```text
café      = [0.72, 0.41, 0.88]
chá       = [0.69, 0.39, 0.84]
automóvel = [-0.42, 0.91, -0.35]
```

Queremos responder:

> Qual palavra está semanticamente mais próxima de “café”? 

Calculamos:

$$
sim(café, chá)
$$

$$
sim(café, automóvel)
$$

Exemplo ilustrativo:

```text
sim(café, chá)       = 0.998
sim(café, automóvel) = -0.12
```

Os valores acima são apenas um exemplo construído para explicar a ideia. O princípio importante é:

$$
\boxed{significado\rightarrow vetor\rightarrow geometria\rightarrow similaridade}
$$

---

## 11. Significado virou geometria

Uma palavra, frase, documento ou imagem pode ser convertida em:

$$
x\in\mathbb{R}^{n}
$$

Por exemplo:

$$
x\in\mathbb{R}^{768}
$$

ou:

$$
x\in\mathbb{R}^{1536}
$$

Não conseguimos visualizar 1536 dimensões, mas conseguimos calcular:

- $x\cdot y$;
- $\|x\|$;
- $d(x,y)$;
- $\cos(x,y)$.

Portanto conseguimos fazer geometria nesse espaço.

---

## 12. Busca por palavra-chave versus busca semântica

Imagine que temos o documento:

> O consumo moderado de café contém compostos bioativos e cafeína.

E o usuário pergunta:

> Quais são os efeitos de beber café?

Uma busca puramente baseada em palavras pode depender muito de termos idênticos. Uma busca semântica transforma pergunta e documento em embeddings e compara suas representações vetoriais.

---

## 13. Busca semântica passo a passo

Coleção:

```text
D1 — Cultivo de café no Brasil
D2 — Benefícios e composição do café
D3 — Manutenção de motores elétricos
D4 — Chá verde e cafeína
D5 — Legislação de trânsito
```

Pergunta:

> Qual a relação entre café e cafeína?

Transformamos a pergunta em embedding $q$ e cada documento em $d_1, d_2, ..., d_5$.

Calculamos:

$$
sim(q,d_i)
$$

Imagine:

```text
D1 = 0.78
D2 = 0.96
D3 = 0.10
D4 = 0.83
D5 = 0.06
```

Ranking:

```text
1º D2 → 0.96
2º D4 → 0.83
3º D1 → 0.78
4º D3 → 0.10
5º D5 → 0.06
```

Isso é um **ranking vetorial**.

---

## 14. Top-K

Em sistemas reais geralmente não queremos recuperar todos os documentos.

Podemos definir:

```text
Top K = 3
```

No exemplo seriam recuperados:

```text
D2
D4
D1
```

Esse conceito aparece continuamente em RAG.

---

## 15. RAG

RAG significa **Retrieval-Augmented Generation**.

Fluxo conceitual:

```text
Usuário
   ↓
Pergunta
   ↓
Embedding da pergunta
   ↓
Busca vetorial
   ↓
Similaridade de cosseno
   ↓
Top-K documentos
   ↓
Contexto
   ↓
LLM
   ↓
Resposta
```

O LLM não precisa encontrar a informação sozinho. Primeiro existe uma etapa de **recuperação**.

---

## 16. Exemplo de RAG

Usuário:

> Quais normas regulam determinado produto?

O sistema transforma a pergunta em $q$ e compara esse vetor contra milhares de embeddings:

$$
d_1,d_2,\ldots,d_{100000}
$$

Depois seleciona, por exemplo, o Top-5 e entrega esses documentos como contexto para o LLM responder.

---

## 17. Vector database

Com poucos documentos, comparar todos os vetores é fácil. Com milhões de embeddings, uma busca ingênua pode ficar cara.

É aí que entram bancos vetoriais e mecanismos de busca vetorial, que armazenam embeddings e possuem estruturas especializadas para encontrar vizinhos próximos.

```text
documento
    ↓
embedding
    ↓
vector database
```

Depois:

```text
consulta
   ↓
embedding
   ↓
vector database
   ↓
vetores próximos
```

---

## 18. Similaridade versus distância

Uma convenção comum é:

$$
\boxed{cosine\ distance=1-cosine\ similarity}
$$

Então:

```text
similaridade alta → distância baixa
```

Sempre pergunte:

> **A biblioteca está retornando similaridade ou distância?**

---

## 19. Não existe um limiar universal

É tentador pensar:

```text
similaridade > 0.8 → igual
similaridade < 0.8 → diferente
```

Mas isso não funciona universalmente. O valor depende de fatores como:

- modelo de embeddings;
- domínio dos dados;
- tamanho dos textos;
- forma de segmentação;
- idioma;
- distribuição dos embeddings;
- objetivo da aplicação.

O limiar precisa ser avaliado experimentalmente.

---

## 20. Ranking costuma ser mais importante que um valor isolado

Em busca semântica, muitas vezes não perguntamos se um score isolado é “alto o suficiente”, mas quais documentos possuem os maiores scores.

Exemplo:

```text
Documento A → 0.91
Documento B → 0.88
Documento C → 0.84
Documento D → 0.51
```

O ranking $A>B>C>D$ pode ser mais importante que interpretar isoladamente cada número.

---

## 21. Limitação importante

A similaridade de cosseno ignora magnitude. Isso é justamente sua vantagem em muitos casos, mas também pode ser uma limitação.

Considere:

$$
a=\begin{bmatrix}1\\2\end{bmatrix}
$$

$$
b=\begin{bmatrix}1000\\2000\end{bmatrix}
$$

A similaridade será $1$ porque possuem a mesma direção.

A métrica está dizendo:

> “as direções são idênticas.”

Ela não está dizendo:

> “os vetores são idênticos em todos os sentidos.”

---

## 22. A métrica depende do problema

**Distância Euclidiana**

$$
d(a,b)=\|a-b\|
$$

Pergunta: *Quão próximos os pontos estão?*

**Similaridade de cosseno**

$$
sim(a,b)=\frac{a\cdot b}{\|a\|\|b\|}
$$

Pergunta: *Quão semelhantes são suas direções?*

Nenhuma é universalmente melhor. São perguntas geométricas diferentes.

---

## 23. Ligação com KNN

No K-Nearest Neighbors, “próximo” precisa ser definido por uma métrica.

Podemos usar distância Euclidiana ou, dependendo da representação, cosine distance.

```text
dados
  ↓
representação vetorial
  ↓
métrica
  ↓
vizinhos
```

---

## 24. Espaço semântico e clusters

Itens semanticamente relacionados podem ocupar regiões próximas do espaço vetorial, formando agrupamentos (*clusters*).

A similaridade de cosseno ajuda a navegar esse espaço e encontrar representações alinhadas semanticamente.

---

## 25. O que um embedding realmente representa?

Não devemos imaginar que uma dimensão específica necessariamente significa uma característica humana explícita.

Em modelos modernos, as representações são normalmente distribuídas. O significado emerge do padrão formado por muitas dimensões simultaneamente.

---

## 26. Conexão entre as aulas

Aula 01:

$$
y=Wx+b
$$

Aula 02:

$$
\|x\|
$$

Aula 03:

$$
a\cdot b
$$

Aula 04:

$$
\frac{a\cdot b}{\|a\|\|b\|}
$$

Construção gradual:

```text
vetor
  ↓
magnitude
  ↓
produto escalar
  ↓
ângulo
  ↓
similaridade
  ↓
busca semântica
  ↓
RAG
```

---

## 27. Mini busca semântica em NumPy

```python
import numpy as np

query = np.array([0.8, 0.4, 0.2])

docs = {
    "café": np.array([0.79, 0.42, 0.18]),
    "chá": np.array([0.72, 0.46, 0.25]),
    "carro": np.array([-0.3, 0.8, -0.4])
}


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


resultados = []

for nome, vetor in docs.items():
    score = cosine_similarity(query, vetor)
    resultados.append((nome, score))

resultados.sort(key=lambda x: x[1], reverse=True)

for nome, score in resultados:
    print(nome, score)
```

Essa é uma versão extremamente simplificada de uma **busca vetorial**.

---

## 28. O salto conceitual

Não perguntamos apenas:

> A palavra “café” existe no documento?

Perguntamos:

> **Qual vetor está semanticamente mais próximo da consulta?**

Isso muda completamente a maneira como recuperamos informação.

---

## 29. Princípio para guardar

Quando olhar para:

$$
\frac{a\cdot b}{\|a\|\|b\|}
$$

pense:

> **Estou retirando a influência do tamanho para medir o alinhamento das direções.**

E, trabalhando com embeddings:

> **Estou usando geometria para comparar representações semânticas.**

---

## 30. Mapa mental da Aula 04

```text
Texto
  ↓
Embedding
  ↓
Vetor
  ↓
Normalização
  ↓
Produto escalar
  ↓
Similaridade de cosseno
  ↓
Ranking
  ↓
Top-K
  ↓
Busca semântica
  ↓
RAG
```

---

## Exercícios de fixação

### Questão 1

Considere:

$$
a=\begin{bmatrix}1\\2\end{bmatrix}
$$

$$
b=\begin{bmatrix}2\\4\end{bmatrix}
$$

Sem fazer todas as contas, qual deve ser aproximadamente a similaridade de cosseno? Explique por quê.

### Questão 2

Considere:

$$
a=\begin{bmatrix}1\\0\end{bmatrix}
$$

$$
b=\begin{bmatrix}0\\7\end{bmatrix}
$$

Qual é a similaridade de cosseno? O que isso significa geometricamente?

### Questão 3

Dois pares de vetores possuem scores de similaridade $0.97$ e $0.42$. Qual par está mais alinhado geometricamente?

### Questão 4

Por que não devemos assumir que `similaridade > 0.80` sempre significa que dois textos são semanticamente equivalentes?

### Questão 5

Explique com suas palavras o fluxo:

```text
pergunta
   ↓
embedding
   ↓
busca vetorial
   ↓
Top-K
   ↓
LLM
```

### Questão 6 — a mais importante

Imagine que você possui **1 milhão de documentos**. Explique por que transformar todos eles em embeddings previamente pode tornar a busca semântica mais eficiente do que pedir ao LLM para ler todos os documentos toda vez que o usuário fizer uma pergunta.

---

## Próxima aula

**Aula 05 — Matrizes como transformações: entendendo de verdade $y=Wx+b$**

Vamos estudar:

- o que uma matriz faz com um vetor;
- rotação, escala e transformação;
- dimensões de matrizes;
- multiplicação matriz-vetor;
- pesos de uma rede neural;
- por que uma camada neural é essencialmente uma transformação aprendida.
