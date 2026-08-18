# Matemática para IA — Índice das 30 aulas

Este diretório implementa a trilha de Matemática do plano M1–M2. O objetivo não é formar um matemático puro, mas construir domínio suficiente para ler equações de ML/DL, derivar algoritmos centrais e implementar gradient descent do zero.

## Bloco A — Álgebra Linear fundamental

1. Escalares, vetores, matrizes e tensores
2. Vetores como objetos geométricos: norma e distância
3. Produto escalar
4. Similaridade de cosseno, embeddings e RAG
5. Matrizes como transformações: `y = Wx + b`
6. Transformações geométricas
7. Determinante
8. Matriz inversa
9. Sistemas lineares, Gauss e rank
10. Independência linear, span, base e dimensão
11. Os quatro subespaços fundamentais
12. Ortogonalidade, Gram-Schmidt e QR

## Bloco B — Álgebra Linear avançada para IA

13. Autovalores e autovetores — intuição
14. Autovalores, diagonalização e matrizes simétricas
15. Decomposições matriciais: LU, QR e espectral
16. SVD
17. PCA
18. Mínimos quadrados, pseudoinversa e condicionamento

## Bloco C — Cálculo para IA

19. Funções, composição, exponencial, log, sigmoid e softmax
20. Limites e continuidade
21. Derivadas e taxa de variação
22. Regras de derivação e regra da cadeia
23. Derivadas parciais e gradiente
24. Jacobiano, cadeia multivariada e autodiff
25. Hessiana, curvatura e Taylor

## Bloco D — Otimização e Gate I

26. Otimização, funções de perda e convexidade
27. Gradient Descent do zero
28. SGD, mini-batch e momentum
29. Regressão linear com Gradient Descent do zero
30. Regressão logística + Cross-Entropy + Gate I

## Gate I

O Gate I exige evidência prática, não apenas leitura. Ao final, você deve conseguir:

- derivar gradient descent;
- implementar regressão linear e logística sem autograd;
- comparar três learning rates;
- validar gradientes numericamente;
- explicar geometricamente as principais operações de Álgebra Linear;
- conectar `y=Wx+b`, regra da cadeia e otimização ao funcionamento de redes neurais.

## Referências-base

1. DEISENROTH, M. P.; FAISAL, A. A.; ONG, C. S. *Mathematics for Machine Learning*. Cambridge University Press, 2020. https://mml-book.github.io/
2. STRANG, G. *MIT 18.06 Linear Algebra*. MIT OpenCourseWare. https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/
3. GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. *Deep Learning*. MIT Press, 2016. https://www.deeplearningbook.org/
4. BOYD, S.; VANDENBERGHE, L. *Convex Optimization*. Cambridge University Press, 2004. https://web.stanford.edu/~boyd/cvxbook/
5. BAYDIN, A. G. et al. *Automatic Differentiation in Machine Learning: a Survey*. JMLR, 2018. arXiv:1502.05767.
6. JOLLIFFE, I. T.; CADIMA, J. *Principal component analysis: a review and recent developments*. Phil. Trans. R. Soc. A, 2016. DOI: 10.1098/rsta.2015.0202.
7. RUMELHART, D. E.; HINTON, G. E.; WILLIAMS, R. J. *Learning representations by back-propagating errors*. Nature, 1986. DOI: 10.1038/323533a0.
