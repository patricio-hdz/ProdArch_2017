
## Objetivo

Nos centraremos en la implementación de un algoritmo para la multiplicación de matrices en paralelo.

### Patricio

Consulté la siguiente referencia que explica 3 algoritmos distintos para atacar el problema del producto de matrices en paralelo:

http://www.hpcc.unn.ru/mskurs/ENG/PPT/pp08.pdf

El primero de los algoritmos tiene por nombre *Block-Striped Decomposition*, el cual para realizar el producto AxB = C, 
toma un grupo de subtasks, cada uno compuesto por el producto punto entre 1 renglón de una matriz A con una columna de una matriz B para obtener 1 entrada de 
la matriz resultante C, y lo asigna a un procesador, de esta manera se paraleliza el producto.

otro de los algoritmos explicados es el algoritmo "Fox's method", el cual rompe en bloques a las matrices 
