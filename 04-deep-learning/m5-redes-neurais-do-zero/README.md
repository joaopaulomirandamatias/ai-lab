# M5 — Redes Neurais do Zero

**Trilha:** Especialista em IA  
**Pasta-mãe:** `04-deep-learning/`  
**Pré-requisito:** M1–M4 e Gate II  
**Entregável P5:** MLP em NumPy, com forward e backpropagation manuais, sem PyTorch, TensorFlow, JAX ou autograd.

> O objetivo do M5 não é “usar uma rede”. É conseguir explicar, derivar, implementar,
> verificar e depurar cada transformação que leva dos dados à loss e da loss aos gradientes.

## Regra de implementação

Até o capstone, são permitidos Python e NumPy para arrays, álgebra linear, amostragem e
visualização. Não são permitidos motores de diferenciação automática. scikit-learn pode
ser usado apenas para carregar/separar datasets ou calcular uma métrica de conferência,
nunca para implementar a rede ou o otimizador.

## Sequência didática completa — 24 aulas

| Aula | Tema | Evidência principal |
|---:|---|---|
| 01 | [Do modelo linear ao neurônio artificial](./aulas/01-neuronio-artificial.md) | Forward e derivadas locais de $z=Wx+b$ |
| 02 | Perceptron e regra de aprendizagem | Separação linear e atualização por erro |
| 03 | MLP, camadas densas e convenções de shape | Grafo e dimensões de uma rede de duas camadas |
| 04 | Funções de ativação | Derivar sigmoid, tanh, ReLU e variantes |
| 05 | Forward pass vetorizado | Cache de cada intermediário em NumPy |
| 06 | Losses de regressão e classificação binária | MSE e binary cross-entropy estável |
| 07 | Softmax e cross-entropy multiclasse | Log-sum-exp e gradiente $p-y$ |
| 08 | Derivadas locais e grafo computacional | VJP e gradiente upstream |
| 09 | Backward da camada afim | $\partial L/\partial W$, $\partial L/\partial b$ e $\partial L/\partial X$ |
| 10 | Backward das ativações | Máscaras, saturação e derivadas elemento a elemento |
| 11 | Backward das losses | Da loss ao logit sem saltos algébricos |
| 12 | Regra da cadeia aplicada à rede | Backprop em um exemplo escalar completo |
| 13 | Backprop vetorizado em uma MLP | Rede de duas camadas sem autograd |
| 14 | Gradient checking | Diferenças centrais e erro relativo |
| 15 | Inicialização de pesos | Simetria, Xavier/Glorot e He/Kaiming |
| 16 | Vanishing e exploding gradients | Propagação de variância e normas por camada |
| 17 | Mini-batch, epoch e embaralhamento | Estimador de gradiente e laço de dados |
| 18 | SGD e learning rate | Curvas de loss, estabilidade e convergência |
| 19 | Momentum e Nesterov | Velocidade, amortecimento e ravinas |
| 20 | Regularização | L2, L1, early stopping e capacidade |
| 21 | Dropout do zero | Máscara de Bernoulli e inverted dropout |
| 22 | Normalização quando apropriado | Padronização, LayerNorm e BatchNorm conceitual/manual |
| 23 | Treino e depuração de uma MLP | Overfit de lote, sanidade, métricas e análise de erros |
| 24 | Capstone P5 | MLP do zero em MNIST/Fashion-MNIST com relatório auditável |

## Progressão de domínio

### Bloco A — unidade de computação e forward · 01–07

O aluno constrói o caminho de entrada até a loss e domina shapes, ativações e estabilidade
numérica antes de calcular qualquer gradiente profundo.

### Bloco B — derivadas e backpropagation · 08–14

Cada operação expõe `forward` e `backward`. O aluno deriva no papel, implementa em NumPy
e compara o gradiente analítico com diferenças centrais.

### Bloco C — dinâmica de treinamento · 15–22

Inicialização, fluxo de gradientes, mini-batch, otimizadores, regularização, dropout e
normalização são tratados como mecanismos mensuráveis, não como listas de opções.

### Bloco D — integração e P5 · 23–24

O aluno integra a rede, treina, depura, mede e documenta o capstone. A mesma arquitetura
será reimplementada no M6 com PyTorch para comparar resultados e entender exatamente o
que `autograd` automatiza.

## Critério de passagem do P5

- [ ] forward de todas as camadas escrito manualmente;
- [ ] backward de camada afim, ativação e loss escrito manualmente;
- [ ] cada equação de gradiente acompanhada por shape;
- [ ] gradient checking por parâmetro com erro relativo declarado;
- [ ] mini-batch SGD e momentum implementados sem framework;
- [ ] inicialização coerente com a ativação;
- [ ] curvas de treino/validação e normas de gradientes por camada;
- [ ] baseline e conjunto de teste externo preservados;
- [ ] ablação de pelo menos inicialização, ativação e regularização;
- [ ] análise de erros e seção “o que o experimento não prova”.

## Referências-base verificadas

1. Goodfellow, Bengio e Courville — [Deep Learning](https://www.deeplearningbook.org/), caps. 6–8.
2. Prince — [Understanding Deep Learning](https://udlbook.github.io/udlbook/), caps. 3–7.
3. Stanford CS231n — [Neural Networks and Backpropagation](https://cs231n.stanford.edu/slides/2020/lecture_4.pdf).
4. Dive into Deep Learning — [Multilayer Perceptrons](https://d2l.ai/chapter_multilayer-perceptrons/index.html) e [Backpropagation](https://d2l.ai/chapter_multilayer-perceptrons/backprop.html).
5. Rumelhart, Hinton e Williams (1986) — [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0).
6. Glorot e Bengio (2010) — [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a.html).
7. He et al. (2015) — [Delving Deep into Rectifiers](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html).
8. Srivastava et al. (2014) — [Dropout](https://www.jmlr.org/papers/v15/srivastava14a.html).

## Próximo passo

Comece pela [Aula 01 — Do modelo linear ao neurônio artificial](./aulas/01-neuronio-artificial.md).
