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

### b. Relación entre loadings del PCA, correlaciones entre variables originales y componentes.

Los loadings representan las correlaciones entre las variables originales y las componentes.

### c. Interpretación de las componentes principales del PCA / Rotación varimax.

Materializamos el concepto del significado de cada componente en función de las variables originales dependiendo del sentido de las correlaciones que dicha componente tenga con ellas.

Por otro lado la rotación varimax es un método de rotación ortogonal que minimiza el número de varibales qque tienen saturaciones altas en cada factor. Simplifica la interpretación de los factores optimizando la solución por columna. Además, sabemos que las componentes representan nuevos ejes y la proyección de la nube de puntos sobre estos nos sirve para interpretar la relación entre las variables. La rotación varimax consigue que cada componente rotado presente correlaciones más claras con algunas variables, lo cual facilita la interpretabilidad.

## 3. Análisis de correspondencias

### b. Distancias entre perfiles

Para poder medir distancias adecuadamente entre cada perfil de fila, necesitamos medir la diferencia en distribución pesada inversamente proporcional al peso o masa de cada columna. Esto se conoce como distancia chi-cuadrada.

### d. El porqué se usa la distancia chi-cuadrada (ejemplo)

Utilizamos la distancia chi-cuadradada porque nos permite **comparar** situaciones de acuerdo a un criterio establecido por nosotros sin resultados engañosos derivados de cuestiones externas a nosotros. Por ejemplo, supongamos que nos entregan la siguiente tabla de frecuencias para 100 jugadores de basketball:

|Altura\Peso|50Kg.-70Kg.|70Kg.-90Kg.|90Kg.-110Kg.|     Total     |
|-----------|:---------:|:---------:|:----------:| -------------:|
|1.60m-1.70m|     7     |     15    |     8      |       30      |
|1.71m-1.72m|     6     |      9    |     3      |       18      |
|  >1.72m   |     2     |     10    |    40      |       52      |

De lo anterior supongamos que queremos comparar a aquellos jugadores cuyo peso está entre 70kg. y 90kg., para ello tomamos a aquellos jugadores cuya altura máxima sea de 1.72m. De primera instancia podríamos decir que hay 66% más jugadores que miden hasta 1.7 m y pesan entre 70kg. y 90kg. que los que pesan lo mismo y miden entre 1.71m y 1.72m.

Notemos que el comparativo anterior es sumamente injusto ya que el rango de altura de 1.71m a 1.72m es tan sólo de 2 cm mientras que el rango de 1.6m a 1.7m es de 10 cm, lo cual lo vuelve mucho más probable de tener casos. La distancia chi-cuadrada nos permite arreglar cuestiones como la anterior ya que la idea es medir "distancias" pero relativas a los universos respectivos de las variables, es decir, en el ejemplo anterior **50%** de los jugadores entre 1.6m y 1.7m de altura pesan entre 70kg. y 90kg., comparado con el mismo **50%** de los jugadores entre 1.71m y 1.72m de altura que pesan entre 70kg. y 90kg.

## 4. Escalamiento multidimensional

### a. Objetivo del Classical Multidimensional Scaling

Si tenemos una matriz de distancias entre puntos (matriz de disimilitudes), el escalamiento multidimensional clásico (también conocido como análisis de coordenadas principales), nos permité reconstruir la matriz de distancias euclidianas originales. Un gráfico de dispersión de esta matriz nos mostraría una visualización de las distancias originales de los puntos en un mapa.

### b. Demostración

### c. Mapa de ciudades

```{r}
require(graphics)

loc <- cmdscale(eurodist)
x <- loc[, 1]
# el signo menos pone al Norte arriba, de lo contrario el mapa aparece al revés
y <- loc[, 2]
plot(x, y, type = "n", xlab = "", ylab = "", asp = 1, axes = FALSE,
     main = "Distancias eurodist")
text(x, y, rownames(loc), cex = 0.6)
```

Como podemos observar en el mapa, el norte, que debería aparecer en la parte superior del mapa, se encuentra en la parte inferior. El mapa está rotado. Lo anterior, se debe a que el signo de la segunda columna es el opuesto. Multiplicar la segunda columna de la matriz obtenida del MDS clásico nos arrojaría el mapa correcto.

La causa de esto viene de las matrices que obtenemos de la descomposición en valores singulares. De la descomposición de una matriz $A$ como $A = U \Sigma V'$, la matriz $U$ está formada por los eigenvectores ortonormales de $AA'$, mientras que la matriz $V$ está formada por los eigenvectores ortonormales de la matriz $A'A$. Lo que representa esta descomposición es una serie de transformaciones lineales de una matriz; donde las matrices ortogonales $U$ y $V$ preservan el producto de matrices, tales como las rotaciones. Como transformaciones lineales, una matriz ortogonal funciones como una rotación. 

## 5. Temas adicionales

### 1. Correlaciones policóricas y poliseriales

Es una técnica que permite estimar la correlación entre pares variables latentes (normalmente distribuidas) que correspondan a dos variables ordinales.

### 2. Análisis factorial y structural equation modeling

El análisis factorial se usa para determinar qué variables están más relacionadas con los factores más comunes y cuántos factores son necesarios para "explicar" los datos a un nivel aceptable. 

El propósito del SEM es permitir el examen de un conjunto de relaciones entre una o más variables independientes, sean estas continuas o discretas, desarrollando una forma de romper las correlaciones observadas en un sistema de ecuaciones matematicas que describan sus hipótesis respecto a relaciones causales. 

### 3. PCA y regresión lineal

El objetivo de este análisis es interpretar los coeficientes de la regresión $\beta$ como combinaciones lineales de las entradas de la matriz $X$ ponderados por los valores de la matriz $\Sigma$, para poder explicar la variable de respuesta $y$. 

### 4. Item response theory

Es un paradigma psicométrico utilizado para diseñar, desarrollar, ensamblar, calificar y analizar evaluaciones. Su principal función es describir la probabilidad de una respuesta como función de un rasgo latente unidimensional de una persona, por ejemplo su habilidad cognitiva, física, conocimiento, actitud.

### 5. Multiple correspondence analysis

Es una extension del análisis de correspondencia simple, para un conjunto grande de variables categóricas; el objetivo es construir tablas de contingencia para las características de nuestras variables, para poder después aplicar el análisis de correspondencias simple a cada una de estas tablas de contingencia. 

### 6. Canonical correlation analysis

El propósito del análisis de correlaciones canónicas es describir la relación estadística que existe entre dos conjuntos de variables $X$ y $Y$. 

### 7. Mixed factor analysis

El análisis factorial mixto se utiliza cuando un conjunto de observaciones se pueden distinguir de acuerdo a variables tanto cualitativas como cuantitativas. Su objetivo es transformar las variables cualitativas para poder aplicar un PCA o transformar las variables cuantitativas para aplica un MCA (generalmente se prefiere la segunda).

### 8. Canonical correlations
