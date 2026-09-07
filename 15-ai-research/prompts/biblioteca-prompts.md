# Biblioteca de Prompts Científicos
## Prompts para apoiar, não substituir, o pesquisador

### 1. Expansão de termos de busca
> Minha pergunta científica é: [pergunta]. Gere sinônimos, termos técnicos, grafias alternativas e descritores relacionados. Organize em blocos conceituais. Não invente artigos nem referências.

### 2. Crítica de string booleana
> Analise esta string de busca para [base]. Identifique risco de baixa sensibilidade e baixa especificidade. Não execute a busca. Sugira alterações e explique o impacto esperado.

### 3. Extração com evidência
> Usando somente o texto fornecido, extraia: população, amostra, desenho, intervenção/exposição, desfechos, resultados e limitações. Para cada campo, indique a seção/trecho de origem. Se não estiver presente, responda "não localizado".

### 4. Síntese controlada
> Use apenas os estudos fornecidos. Separe achados convergentes, divergentes e incertos. Não crie novas referências. Para cada conclusão, liste quais estudos a sustentam.

### 5. Revisor adversarial
> Tente refutar minha interpretação. Procure confundidores, hipóteses alternativas, generalizações indevidas, causalidade não demonstrada e limitações. Não acrescente fatos externos.

### 6. Auditor de referências
> Analise a lista fornecida apenas quanto a sinais internos de inconsistência bibliográfica. Não confirme existência sem acesso a uma fonte bibliográfica. Marque o que precisa ser verificado externamente.

### 7. Código com critérios
> Antes de gerar código, descreva o método, pressupostos, entradas, saídas e testes que serão usados. Depois gere uma implementação mínima. Inclua verificações de erros e indique decisões que exigem validação humana.

### 8. Revisão estatística
> Não escolha um teste imediatamente. Primeiro faça perguntas sobre desenho, tipo das variáveis, independência, distribuição, tamanho amostral e hipótese. Depois apresente opções com pressupostos e limitações.

### 9. Redação sem inventar conteúdo
> Melhore clareza e coesão do texto abaixo sem adicionar fatos, resultados, referências ou interpretações novas. Preserve o significado científico.

### 10. Declaração de uso
> Com base neste registro de uso de IA, estruture uma declaração transparente. Não omita usos metodologicamente relevantes. Não afirme conformidade com uma política que não foi fornecida.

## Anti-padrões

Evite:
> Escreva minha revisão completa e coloque referências.

Prefira:
> Ajude a estruturar a revisão usando exclusivamente a matriz de evidências verificada que fornecerei.

Evite:
> Qual teste devo usar?

Prefira:
> Faça uma árvore de decisão a partir do desenho e pressupostos; depois discutirei a escolha.
