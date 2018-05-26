## Introducción

Este trabajo busca demostrar de manera evidente, mediante una implementación básica cómo es que se pueden explotar las más recientes características de los equipos de cómputo para lograr un procesamiento más veloz, para esto hemos decidido implementar una multiplicación de matrices de manera paralela y secuencial para poder medir los rendimientos y compararlos.

La idea es utilizar distintas herramientas como lo son: Docker-machine, maquinas en AWS y programación en paralelo para ejemplificar de manera práctica las mejoras en los tiempos de ejecución que se pueden tener hoy en día.

## Motivación
En los últimos años la velocidad de procesamiento ha crecido debido al uso de procesadores multi-core. Derivado de lo anterior se ha ido teniendo la creciente necesidad de modificar el código de los algoritmos que se usaban comúnmente para adaptarlos de la mejor manera a un paradigma distinto el cual representa una mayor eficiencia, dicho paradigma se le conoce como “procesamiento en paralelo”.

Existen distintas maneras de atacar el problema de paralelizar códigos secuenciales, sin embargo dada la formación de los autores de este trabajo, la manera más natural nos resultó aquella en la cual un grupo de ejecuciones que forman parte de uno o más ciclos “for” se ejecutan cada una en un procesador a manera de que posteriormente podamos reagrupar y obtener el resultado final esperado.

## Pseudo-algoritmo
El resultado final esperado es reducir el tiempo necesario que toma ejecutar un algoritmo mediante una asignación de tareas a los procesadores más eficiente.

Supongamos por ejemplo que deseamos realizar una suma de dos vectores <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{w&space;=&space;x&space;&plus;&space;y}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;\boldsymbol{w&space;=&space;x&space;&plus;&space;y}" title="\small \boldsymbol{w = x + y}" /></a>; con <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{x,y\epsilon&space;\mathbb{R}^n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;\boldsymbol{x,y\epsilon&space;\mathbb{R}^n}" title="\small \boldsymbol{x,y\epsilon \mathbb{R}^n}" /></a>:

![Imagen1](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im1.JPG)

En este caso podemos ver como las sumas se podrían realizar de manera simultánea, si asignamos cada pedazo de la suma global a distintos procesadores.

Un punto sumamente relevante en este caso, es que no existe una dependencia entre los procesos que se realizan en cada procesador, por ejemplo esto NO funcionaría si estuviéramos utilizando en cada operación el resultado de la operación anterior, ya que sería necesario forzosamente esperar a tener dicho resultado para poder continuar con las operaciones hasta llegar al resultado final.

Por otro lado una operación no tan directa es el producto punto de dos vectores <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;a&space;\boldsymbol{=&space;<x,y>}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;a&space;\boldsymbol{=&space;<x,y>}" title="\small a \boldsymbol{= <x,y>}" /></a>; con <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;\boldsymbol{x,y\epsilon&space;\mathbb{R}^n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;\boldsymbol{x,y\epsilon&space;\mathbb{R}^n}" title="\small \boldsymbol{x,y\epsilon \mathbb{R}^n}" /></a> y <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;a" title="a" /></a> un escalar:

![Imagen2](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im2.JPG)

En este caso es menos evidente la manera en que tendríamos que repartir las operaciones a realizarse en los procesadores, sin embargo a continuación presentamos la metodología que nos parece más "natural" para atacar este problema:

![Imagen3](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im3.JPG)

De esta manera se logra el objetivo que se tenía de eficientar la operación y funciona como una buena explicación base para mencionar que de esta misma manera decidimos atacar el problema de paralelizar la multiplicación de matrices ya que, en ese caso realizamos esta operación de producto punto entre vectores una y otra vez, pero al no depender una de otra podemos de igual manera "romper" las ejecuciones y paralelizar los procesos.

## Medidas control

Ahora presentaremos algunas métricas de control que utilizaremos para evaluar si nuestra implementación está teniendo mejores resultados que una ejecución secuencial tradicional o no; dichas métricas son las siguientes:

- *Tiempo real de procesamiento*
- *Tiempo ideal de procesamiento*
- *Speed-up*
- *Eficiencia*

#### Tiempo real de procesamiento

El *tiempo real de procesamiento* <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{t_1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{t_1}" title="\small {t_1}" /></a> es el tiempo que le lleva al equipo de cómputo ejecutar todas las instrucciones necesarias para completar un programa utilizando **un solo** thread.

#### Tiempo ideal de procesamiento

Si <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{t_1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{t_1}" title="\small {t_1}" /></a> es el *tiempo real* bajo la definición anterior y <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{n}" title="\small {n}" /></a> es el número de procesadores utilizados, entonces el *tiempo ideal* de procesamiento lo podemos definir como:

![Imagen4](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im4.JPG)

#### Speed-up

La ganancia en tiempo que se tiene con la ejecución en paralelo de un algoritmo en comparación con su ejecución secuencial se conoce como *speed-up*, y lo podemos visualizar de la siguiente manera:

![Imagen5](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im5.JPG)

donde <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{t_n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{t_n}" title="\small {t_n}" /></a> es el tiempo de procesamiento en paralelo para <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{n}" title="\small {n}" /></a> threads.

El *speed-up ideal* es aquel que crece a una razón 1:1, es decir, por cada thread que agregamos, el *speed-up* crece 1 unidad.

#### Eficiencia

La *eficiencia* se define como el *speed-up* por procesador, entonces tomando <a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{100}&space;\small&space;{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\dpi{100}&space;\small&space;{n}" title="\small {n}" /></a> como el número de procesadores, tenemos la siguiente expresión:

![Imagen6](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im6.JPG)


## Análisis y Resultados

Para generar las siguientes gráficas utilizamos una maquina de AWS con las siguientes características:

- Tipo de maquina: m4.2xlarge
- Número de cpu's virtuales: 8
- Memoria RAM: 32gb
- Vpc_id: El creado para la clase de MNO
- Subnet_id: El creado para la clase de MNO

Pensamos en dos propósitos al utilizar esta maquina levantada con docker-machine, el primero es que deseamos tener un trabajo lo más reproducible posible para que otras personas puedan jugar con el código del proyecto. El segundo propósito es que nos parece que al correr todo en una maquina "nueva" los resultados que obtenemos se vuelven más *justos* y comparables.

Nos parece relevante mencionar en esta parte, que el código está listo para recibir y hacer pruebas con matrices rectangulares, sin embargo, dado que la idea global del trabajo es comparar las ventajas en performance que tiene una implementación en paralelo contra una secuencial, sólo utilizamos matrices cuadradas ya que son más sencillas de manejar.

#### Resultados gráficos

###### Tiempos reales por thread
En la siguiente gráfica observamos los tiempos de ejecución cuando variamos tanto las dimensiones de las matrices a multiplicar, como cuando paralelizamos con distinto número de threads:

<p align="center">
  <img src="https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im7.JPG">
</p>

Podemos notar varias cosas de la gráfica anterior:

1. Los tiempos no varían por cantidad de threads para dimensiones "pequeñas".
2. Notamos un cambio en los tiempos a partir de dimesiones aproximadamente de 360x360.
3. No es claro qué cantidad de threads es la que tiene mejor performance, pero profundizaremos mas adelante.
4. Notamos algunos saltos atípicos sin embargo lo atribuimos a la matriz que utilizó en esa iteración ya que se generaron de manera aleatoria y los saltos se dan para TODAS las cantidades de threads.

###### Speed-up

Para poder medir las otras métricas de control que planteamos en la sección pasada, tomaremos las ejecuciones que se tienen para el caso de 1000x1000.

En la siguiente gráfica observamos el *speed-up* y el *speed-up ideal*:

<p align="center">
  <img src="https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im8.JPG">
</p>

Notamos que el *speed-up* es cercano al *speed-up ideal* hasta 4 threads, sin embargo posterior a esto baja y se estabiliza sin mostrar una mejora extra al aumentar la cantidad de threads.

###### Tiempo ideal de procesamiento

Observamos ahora una gráfica donde se compara el *tiempo real de procesamiento* contra el *tiempo ideal de procesamiento* bajo su definición de la sección anterior:

<p align="center">
  <img src="https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im9.JPG">
</p>

En este caso, notamos que a partir de 8 threads el *tiempo real* se estabiliza y por lo tanto la diferencia contra el *tiempo ideal* aumenta de manera más evidente, esto nos hace sentido ya que la maquina de donde se obtuvieron los resultados es una maquina con 8 procesadores.

###### Eficiencia

Finalmente presentamos una gráfica donde observamos el cambio en la eficiencia conforme modificamos el número de threads:

<p align="center">
  <img src="https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im10.JPG">
</p>

Notamos de la gráfica anterior que aunque con las demás métricas hemos demostrado un mejor performance en cuestión de tiempo, en este caso tenemos un área de oportunidad para mejorar la manera en que estamos paralelizando ya que la eficiencia cae rápidamente conforme aumentamos el número de threads.

## Conclusiones



