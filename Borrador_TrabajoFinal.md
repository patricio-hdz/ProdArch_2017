## Introducción

Este trabajo busca demostrar de manera evidente, mediante una implementación de principiantes cómo es que se pueden explotar las más recientes características de los equipos de cómputo para lograr una procesamiento más veloz.

La idea es utilizar distintas herramientas como lo son: contenderos de Docker, maquinas en AWS y programación en paralelo para ejemplificar de manera práctica las mejoras en los tiempos de ejecución que se pueden tener hoy en día.

## Motivación
En los últimos años la velocidad de procesamiento ha crecido debido al uso de procesadores multi-core. Derivado de lo anterior se ha ido teniendo la creciente necesidad de modificar el código de los algoritmos que se usaban comúnmente para adaptarlos de la mejor manera a un paradigma distinto el cual representa una mayor eficiencia, dicho paradigma se le conoce como “procesamiento en paralelo”.

Existen distintas maneras de atacar el problema de paralelizar códigos secuenciales, sin embargo dada la formación de los autores de este trabajo, la manera más natural nos resultó aquella en la cual un grupo de ejecuciones que forman parte de uno o más ciclos “for” se ejecutan cada una en un procesador a manera de que posteriormente podamos reagrupar y obtener el resultado final esperado.

## Pseudo-algoritmo
El resultado final esperado es reducir el tiempo necesario que toma ejecutar un algoritmo mediante una asignación de tareas a los procesadores más eficiente.

Supongamos por ejemplo que deseamos realizar una suma de dos vectores 

![Imagen1](https://github.com/patricio-hdz/ProdArch_2017/blob/master/tarea1/Im1.JPG)
