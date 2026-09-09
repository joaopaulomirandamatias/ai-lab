# M6 — PyTorch: do cálculo manual ao treinamento reproduzível

**Trilha:** Especialista em IA · **Módulo:** [04 — Deep Learning](../README.md)  
**Pré-requisito:** [M5 e Capstone P5](../m5-redes-neurais-do-zero/aulas/24-capstone-p5.md).  
**Entregável P6:** reproduzir a MLP do P5 em PyTorch, explicando o que autograd automatiza e conferindo operações, gradientes, atualização e avaliação.

## Contrato da trilha

Esta grade de **24 aulas** foi registrada em 9 de setembro de 2026, antes da redação da Aula 01. A numeração, os temas e os nomes de arquivos abaixo são estáveis. Os nomes ainda sem link são contratos curriculares, não arquivos já publicados. Cada ciclo cria ou aprimora somente a próxima aula pendente.

Começamos com tensores e equivalência com NumPy; depois introduzimos autograd e abstrações de modelo/dados; por fim, reprodutibilidade, aceleração e avaliação. O estudante deve conseguir relacionar cada chamada às equações do M5. O uso de uma API não substitui o entendimento de seu contrato.

## Currículo canônico — 24 aulas

| Aula | Tema e arquivo estável em `aulas/` | Evidência de aprendizagem |
|---:|---|---|
| 01 | [Tensores: a ponte entre NumPy e PyTorch](aulas/01-tensores-numpy-pytorch.md) — `01-tensores-numpy-pytorch.md` | Shape, dtype, device, cópia/compartilhamento e forward equivalente |
| 02 | Eixos, indexação, broadcasting e layout — `02-eixos-broadcasting-layout.md` | Reduções, reshape, transpose, strides e contiguidade sem trocar a semântica |
| 03 | Autograd: de escalares a VJPs — `03-autograd-vjp.md` | Regra da cadeia e derivadas automáticas comparadas ao cálculo manual |
| 04 | Ciclo de vida do grafo e acúmulo de gradientes — `04-grafo-acumulo-gradientes.md` | Leaf tensors, zeroing, detach, operações in-place e ausência de grafo |
| 05 | Gradient checking e paridade com NumPy — `05-gradcheck-paridade-numpy.md` | Gradientes por parâmetro, precisão dupla, quinas e tolerâncias |
| 06 | `nn.Module`, parâmetros e buffers — `06-module-parametros-buffers.md` | Registro de estado, composição e inspeção dos parâmetros treináveis |
| 07 | Camadas lineares, ativações e inicialização — `07-linear-ativacoes-inicializacao.md` | Transposição de pesos do P5, inicializações controladas e comparação do forward |
| 08 | Losses, logits e reduções — `08-losses-logits-reducoes.md` | CE/BCE/MSE estáveis, targets corretos e equivalência do objetivo |
| 09 | `Dataset` e transformações sem vazamento — `09-dataset-transformacoes.md` | Unidade de análise, splits, ajuste no treino e contrato dos exemplos |
| 10 | `DataLoader`, amostragem e lotes — `10-dataloader-amostragem.md` | Cobertura, último lote, shuffle, workers e seeds |
| 11 | Laço de treinamento com SGD explícito — `11-laco-treino-sgd.md` | Forward, backward, atualização e métricas sem esconder etapas |
| 12 | `torch.optim`, momentum e schedules — `12-optim-momentum-schedules.md` | Convenções do P5, ordem dos steps e estado da taxa de aprendizagem |
| 13 | Adam, AdamW e regularização — `13-adam-adamw-regularizacao.md` | Momentos, correção de viés e L2 versus weight decay desacoplado |
| 14 | Avaliação, modos e seleção de checkpoint — `14-avaliacao-modos-selecao.md` | `train`/`eval`, `no_grad`, `inference_mode`, early stopping e teste reservado |
| 15 | Dropout e normalizações em PyTorch — `15-dropout-normalizacoes.md` | Paridade com implementações manuais, eixos e estatísticas de treino/inferência |
| 16 | Seeds, determinismo e fixtures — `16-seeds-determinismo-fixtures.md` | Repetição controlada e limites entre bibliotecas, versões e dispositivos |
| 17 | Persistência para inferência — `17-state-dict-inferencia.md` | `state_dict`, configuração, pré-processamento e teste de ida e volta |
| 18 | Checkpoints e retomada de treinamento — `18-checkpoints-retomada.md` | Modelo, otimizador, scheduler, RNG e posição dos dados restaurados |
| 19 | CPU, GPU e CUDA — `19-cpu-gpu-cuda.md` | Transferências, colocação consistente, memória e medição sincronizada |
| 20 | Precisão mista automática — `20-precisao-mista-amp.md` | `autocast`, scaling quando aplicável, estabilidade e comparação com FP32 |
| 21 | Profiling e eficiência do treinamento — `21-profiling-eficiencia.md` | Custo, memória, throughput, tamanho do lote e gargalos medidos |
| 22 | Panorama comparativo TensorFlow/Keras — `22-panorama-tensorflow-keras.md` | Mapear tensores, camadas, autodiff, modos e pesos numa pequena fixture |
| 23 | Integração P6: reproduzindo o P5 — `23-integracao-p6-paridade-p5.md` | Mesmo dataset, split, arquitetura, pesos, loss, atualização e orçamento |
| 24 | Capstone P6 e avaliação de domínio — `24-capstone-p6.md` | Relatório de paridade, curvas, teste reservado, desempenho e defesa das diferenças |

## Cobertura dos objetivos do módulo

| Objetivo herdado do módulo 04 | Tratamento no M6 |
|---|---|
| Tensors e autograd | Aulas 01–05; fórmulas e fixtures do M5 são a referência |
| `nn.Module`, ativações, losses e inicialização | Aulas 06–08 |
| `Dataset`, `DataLoader`, batch e epoch | Aulas 09–11 |
| SGD, momentum, Adam, learning rate e regularização | Aulas 11–15 |
| Avaliação e checkpoints | Aulas 14, 16–18 e 23–24 |
| GPU, CUDA e mixed precision | Aulas 19–21, com caminho CPU executável e limites de hardware declarados |
| Conhecer TensorFlow/Keras | Aula 22, comparação conceitual e operacional delimitada |
| P6: o mesmo MLP do P5 | Integração nas aulas 23–24, apoiada por todas as anteriores |

CNN, pooling, ResNet, transfer learning, RNN/LSTM/GRU, encoder-decoder/seq2seq e os fundamentos de RL (MDP, value function, Q-learning, policy gradient e PPO) pertencem à trilha **M7 — Arquiteturas**. O objetivo de RL é preparar o estudo posterior de alinhamento; RLHF e DPO ficam no módulo 06. NLP clássico, tokenização e attention pertencem ao módulo 05. Assim preservamos todos os objetivos do módulo-mãe sem repetir suas trilhas em M6.

## Critério de passagem P6

- [ ] Comparar forward, loss e gradientes com NumPy em fixture fixa antes de treinar.
- [ ] Explicar a orientação de `nn.Linear.weight` e transportar corretamente os pesos do P5.
- [ ] Igualar redução da loss, penalidade, convenção do otimizador, seeds e ordem dos lotes.
- [ ] Reproduzir os splits e o pré-processamento do P5 sem selecionar pelo teste.
- [ ] Registrar versões, dispositivo, dtype, tolerâncias e diferenças numéricas por etapa.
- [ ] Apresentar curvas de treino/validação e métricas no teste reservado, com análise de erros.
- [ ] Demonstrar inferência após recarga e retomada de treino com o estado necessário.
- [ ] Medir GPU/AMP quando houver hardware; registrar explicitamente trechos não executados.
- [ ] Comparar acurácia, tempo e memória sob orçamento declarado; não exigir igualdade bit a bit entre plataformas.
- [ ] Defender o que o experimento demonstra e o que permanece sem evidência.

O P5 de referência é o capstone MNIST do M5: 10.000 imagens de treino, 2.000 de validação e 10.000 de teste oficial. O P6 deve separar **paridade de implementação**, com pesos e entradas idênticos, de **variação experimental**, avaliada por execuções controladas. Uma seed de mesmo número em NumPy e PyTorch não produz automaticamente os mesmos pesos.

## Fontes e papel de cada material

Fontes verificadas em 9 de setembro de 2026. A Aula 01 registra PyTorch 2.6.0 como ambiente executado e identifica as versões de cada página consultada; cada aula posterior deve declarar suas versões e verificar as APIs que utilizar.

- **Fonte conceitual principal herdada:** Simon Prince, [Understanding Deep Learning](https://udlbook.github.io/udlbook/). Relacione os capítulos de redes, treinamento e regularização às aulas 03–15.
- **Referência normativa das APIs:** [documentação PyTorch](https://docs.pytorch.org/docs/2.14/index.html) e [instalação por versão/plataforma](https://pytorch.org/get-started/previous-versions/).
- **Código ao lado da teoria:** [Dive into Deep Learning](https://d2l.ai/), versão 1.0.3 consultada, e [Deep Learning with PyTorch](https://www.manning.com/books/deep-learning-with-pytorch), Stevens, Antiga e Viehmann. Consulte condições de acesso da editora; não presumimos acesso gratuito à edição integral.
- **Material complementar de implementação:** Karpathy, [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html). Releia o caminho cálculo manual → tensores; não é fundamento normativo das APIs.
- **Referência da aula comparativa 22:** [TensorFlow — Keras](https://www.tensorflow.org/guide/keras). Verifique versões e convenções ao realizar o laboratório dessa aula.
- **Sutton e Barto**, fonte de RL já declarada no módulo-mãe, permanece vinculada ao M7; não é antecipada na trilha de ferramentas PyTorch.

Comece pela [Aula 01](aulas/01-tensores-numpy-pytorch.md). A próxima é **Aula 02 — Eixos, indexação, broadcasting e layout**, conforme a grade acima.
