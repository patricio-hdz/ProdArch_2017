---
title: 'Respuestas de control: métodos multivariados'
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Álgebra lineal

### a. Matriz de covarianzas, descomposición espectral y la forma cuadrática

Partiendo de la matriz de covarianzas, para datos centrados, podemos construirla como $\Sigma = X' X$, suponiendo que nuestros datos son centrados. La descomposición espectral de esta matriz $\Sigma$, nos dará información relevante acerca de cómo se comporta esta nube de puntos dentro del plano cartesiano (dado que estamos trabajando con el caso bivariado).

De la descomposición espectral, $\Sigma = P \Lambda P'$, podemos obtener información acerca de la dispersión de los datos $X$ y las direcciones en las cuales se da la mayor concentración. De la descomposición espectral, la matriz $\Lambda$ diagonal está compuesta por los eigenvalores de $\Sigma$. Estos eigenvalores nos dan las direcciones en las cuales los datos presentan mayor concentración. 

Finalmente, de la forma cuadrática $\{x : x'\Sigma x = 1\}$ sabemos que, puesto que $\Sigma$ es una matriz positiva definida, la forma cuadrática es siempre positiva para cualquier vector $x$ no nulo. Para el caso bivariado, la ecuación de la forma cuadrática define una elipse cuyos parámetros dependen de la matriz de covarianzas $\Sigma$. La dispersión y la dirección de los ejes de la elipse estarán dados por lo valores de esta matriz de covarianzas. Al igualar la ecuación a uno, obtenemos la elipse unitaria que contiene la dispersión de los datos $X$.

### b. Descomposición SVD y su diagonalización

