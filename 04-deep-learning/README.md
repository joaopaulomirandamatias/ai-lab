# 04 · Deep Learning

**Meses M5–M7** · Roadmap: Níveis 4, 5 e 33 (RL)

> Aqui você começa a entender os modelos que deram origem à IA generativa moderna.

---

## O que dominar

**Redes neurais** — perceptron, multilayer perceptron, funções de ativação (ReLU,
Sigmoid, Softmax), forward propagation, loss functions, **backpropagation**, gradient
descent, SGD, Adam, learning rate, batch, epoch, dropout, batch normalization,
weight initialization.

**PyTorch** — tensors, autograd, `Dataset`, `DataLoader`, `nn.Module`, treinamento,
avaliação, GPU, CUDA, mixed precision, checkpoints. Conhecer também TensorFlow/Keras.

**Arquiteturas** (Nível 5)
- CNN — convolução, pooling, kernels, feature maps, ResNet, transfer learning
- Sequenciais — RNN, LSTM, GRU, encoder-decoder, seq2seq

Mesmo que Transformers dominem hoje, entender RNN/LSTM explica *por que* o Transformer
existe: o problema que ele resolve é a dependência sequencial que impede paralelização.

**Reinforcement Learning** (Nível 33, profundidade intermediária) — o suficiente para
entender RLHF e DPO no M10: MDP, política, recompensa, value function, Q-learning,
policy gradient, PPO.

---

## Fonte principal

⭐ **Understanding Deep Learning** — Simon Prince (grátis, `udlbook.github.io`).
Moderno, visual, com notebooks. Provavelmente a melhor recomendação isolada do repositório.

Implementação: ⭐ **Karpathy — Neural Networks: Zero to Hero** (`micrograd` → `makemore`).
Assistir **implementando junto**, não assistindo.

Código ao lado da teoria: D2L (`d2l.ai`) · Deep Learning with PyTorch (grátis).
RL: Sutton & Barto (grátis).

---

## Projetos

**P5 — MLP com backprop à mão (M5)** — NumPy puro, backpropagation derivado e
implementado por você. *Pronto quando* treina em MNIST e você desenha o grafo
computacional de memória.

**P6 — O mesmo MLP em PyTorch (M6)** — `nn.Module`, autograd, GPU, AMP, checkpoints.
*Pronto quando* os resultados batem com P5.

**P7 — CNN + transfer learning (M7)** — do zero, depois com ResNet pré-treinada.
*Pronto quando* você mede e explica a diferença de acurácia **e** de tempo.

---

## A sequência P5 → P6 não é redundante

Escrever backprop à mão e depois deixar o autograd fazer é o que transforma PyTorch de
mágica em ferramenta. Quem pula o P5 passa os próximos 14 meses tratando `.backward()`
como caixa-preta — e trava no M8, porque o Transformer exige entender o que flui por
onde.

É o mesmo princípio que o M12 aplica com agentes: construir do zero antes de usar
framework. O roadmap repete essa lição de propósito.
