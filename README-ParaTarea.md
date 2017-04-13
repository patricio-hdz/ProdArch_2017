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

De la descomposición espectral, $\Sigma = P \Lambda P'$, podemos obtener información acerca de la dispersión de los datos $X$ y las direcciones en las cuales se da la mayor concentración. De la descomposición espectral, la matriz $\Lambda$ diagonal está compuesta por los eigenvalores de $\Sigma$. Los eigenvectores asociados a estos eigenvalores nos dan las direcciones en las cuales los datos presentan mayor concentración. 

Finalmente, de la forma cuadrática $\{x : x'\Sigma x = 1\}$ sabemos que, puesto que $\Sigma$ es una matriz positiva definida, la forma cuadrática es siempre positiva para cualquier vector $x$ no nulo. Para el caso bivariado, la ecuación de la forma cuadrática define una elipse cuyos parámetros dependen de la matriz de covarianzas $\Sigma$. La dispersión y la dirección de los ejes de la elipse estarán dados por lo valores de esta matriz de covarianzas. Al igualar la ecuación a uno, obtenemos la elipse unitaria que contiene la dispersión de los datos $X$.

### b. Descomposición SVD y su diagonalización
 
Dado que la descomposición SVD es un proceso de diagonalización general (para matrices de $m x n$ y no solo para matrices cuadradas) podemos interpretar ambas de la misma forma. Dad una matriz $A$ su descomposición SVD es $A=U\Sigma V'$ en donde la transformacion ortogonal $V$ produce una rotación, $\Sigma$ diagonal produce un deformamiento y $U$ produce una rotación. Para el caso de un conjunto de datos tendremos que su matriz de covarianzas es simétrica y por lo tanto su descomposición SVD será de la forma $\Sigma=P'\Lambda P$, con $P$ matriz ortogonal con los vectores propios de $\Sigma$ y $\Lambda$ matriz diagonal con los valores propios al cuadrado de $\Sigma$.

### c. Relación entre SVD y diagonalización de la matriz de covarianzas de A.

Sabemos que para datos centrados la matriz de covarianzas de $A$ está dada por $A'A$. Los valores singulares de la matriz $A$ son las raices cuadradas de los eigenvectores de $A'A$. Si la descomposición SVD de $A$ es $A=U\Sigma V'$ entonces las columnas de $V$ son los eigenvectores de la matriz $A'A$ (matriz de covarianzas de A).

## 2. Algebra lineal

### a. PCA a través de SVD

Cuando realizamos PCA, seguimos 4 pasos escencialmente durante este proceso:
  1. Centramos los datos para poder trabajar sin problemas de diferencias de escalas.
  2. Calculamos la matriz de covarianzas.
  3. Encontramos las componentes principales.
  4. Reducimos los datos en base a las componentes principales seleccionadas.
  
Cuando llevamos a cabo un PCA, buscamos encontrar una representación fiel pero reducida de nuestros datos. Debido a lo anterior, primero centramos los datos para trabajar en el origen sin problemas de escala; posteriormente calculamos la matriz de  covarianzas ya que éstas nos brindan una medida de las variaciones en los datos.

Para obtener las componentes principales, llevamos a cabo una descomposicón SVD de la matriz de covarianzas, dicha descomposición  como explicamos en los incisos anteriores nos permite a través de los eigenvectores más grandes y sus eigenvalores asociados, descubrir las direcciones y las magnitudes que mejor capturan la naturaleza de los datos. Finalmente utilizamos las componentes que hagan más sentido para reducir la dimensionalidad de la base con la cual trabajamos.

### b. Relación entre loadings del PCA y correlaciones entre variables originales y componentes.

Los loadings representan las correlaciones entre las variables originales y las componentes.
  
