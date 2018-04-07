
## Objetivo

Nos centraremos en la implementación de un algoritmo para **la multiplicación de matrices en paralelo**.

### Patricio Hernández

Consulté la siguiente referencia que explica 3 algoritmos distintos para atacar el problema del producto de matrices en paralelo:

http://www.hpcc.unn.ru/mskurs/ENG/PPT/pp08.pdf

En general noté que la idea es definir el subtask que se desea atacar y posteriormente esos subtasks se pueden distribuir en los procesadores.

El primero de los algoritmos tiene por nombre *Block-Striped Decomposition*, el cual para realizar el producto de matrices: **AxB = C**,
toma un grupo de subtasks, cada uno compuesto por el producto punto entre 1 renglón de *A* con una columna de *B* para obtener 1 entrada de *C*, y lo asigna a un procesador, de esta manera se paraleliza el producto.

Otro de los algoritmos explicados es el algoritmo *Fox's method*, el cual rompe en bloques a las matrices *A* y *B* y realiza el producto, al romper en bloques se paraleliza la operación ya que cada multiplicación de bloques se va agregando a un bloque de *C* y estos procesos se distribuyen en los procesadores.

El último de los métodos explicados en la referencia es el de *Cannon's method*, el cual es un método que también rompe en bloques las operaciones entre las matrices A y B, sin embargo en este caso se tiene una descomposición en bloques inicial y el algoritmo va realizando un reacomodo de los bloques para que el producto se pueda realizar sin llevar a cabo operaciones de transmisión de datos adicionales. Las operaciones se distribuyen entre los procesadores para paralelizar el algoritmo.

Entiendo que aunque el método de Fox y el método de Cannon se parecen, en el primero se propone recorrer los bloques que se forman, mientras que en el segundo caso se propone reacomodar los bloques para que las operaciones se realicen sin recorrer los bloques.

### Alejandro Pérez

### Equipo

Dada nuestra formación, entendemos de mejor manera la metodología del algoritmo *Block-Striped Decomposition*, por lo que comenzaremos a desarrollar las funciones correspondientes para implementar este algoritmo en paralelo.


