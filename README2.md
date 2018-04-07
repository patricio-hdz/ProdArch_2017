
## Objetivo

Nos centraremos en la implementación de un algoritmo para la multiplicación de matrices en paralelo.

### Patricio

Consulté la siguiente referencia que explica 3 algoritmos distintos para atacar el problema del producto de matrices en paralelo:

http://www.hpcc.unn.ru/mskurs/ENG/PPT/pp08.pdf

El primero de los algoritmos tiene por nombre *Block-Striped Decomposition*, el cual para realizar el producto de matrices:

<p align="justify">
  Lasdas
</p>

<center>###### *AxB = C*,</center>

toma un grupo de subtasks, cada uno compuesto por el producto punto entre 1 renglón de *A* con una columna de *B* para obtener 1 entrada de *C*, y lo asigna a un procesador, de esta manera se paraleliza el producto.

Otro de los algoritmos explicados es el algoritmo *Fox's method*, el cual rompe en bloques a las matrices *A* y *B* y realiza el producto, al romper en bloques se paraleliza la operación ya que cada multiplicación de bloques se va agregando a un bloque de *C* y estos procesos se distribuyen en los procesadores.


