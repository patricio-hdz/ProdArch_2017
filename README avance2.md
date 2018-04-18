## Alejandro Pérez Palacios

Encontré en las ligas:

  * http://www.cs.nthu.edu.tw/~cherung/teaching/2011anm/note02.pdf 
  * http://solomonik.cs.illinois.edu/talks/vecpar-2012.pdf
  
instrucciones para poder realizar la multiplicación de matrices en paralelo, además revisando las notas del curso sobre multiplicación de matrices en paralelo noté que se encuentra descrito el algoritmo de cannon de una manera gráfica y más clara. 

Viendo las descripciones y más fuentes noto que este tipo de código requiere de especificar la comunicación entre los nodos o procesadores a través de mensajes send y receive por lo que es necesario que utilicemos message passing (MPI) para implementarlo, entre las ventajas que mas se mencionan para el método de Cannon es que su arquitectura minimiza la comunicación entre los nodos, lo cual hace más facil la programación de los mensajes a través de estos.

De igual manera vemos que en las notas y referencias menciona que la arquitectura que necesita este algoritmo es llamada Toro por la forma en que los mensajes entre los nodos son enviados. 

## Patricio Hernández 

Como nota a lo que menciona Alejandro en las páginas 7 y 8 de las notas del curso podemos ver muy claramente el pseudocódigo que habría que seguir para la implementación en MPI del algoritmo.

Por otro lado y dando continuidad al avance anterior NO nos quedamos sólo con la información al respecto en MPI, también revisamos la siguiente referencia para una implementación en openMP:

  * http://personal.cimat.mx:8181/~miguelvargas/Course%20notes/Computo%20en%20paralelo%20con%20OpenMP%201.pdf
  
 Esta alternativa va acorde a la metodología *Block-Striped Decomposition* y la presentación es bastante completa en términos de permitirnos desarrollar el pseudocódigo ya que tiene bastante explicación práctica de cómo es que funciona openMP.
 
 Revisamos algunas otras referencias como por ejemplo:
 
   * https://prezi.com/amgiu_nlad9u/multiplicacion-de-matrices/

y notamos que en la mayoría de los trabajos en donde se trata este tema, se presenta un comparativo con una implementación secuencial para mostrar las mejoras que se tiene al atacar el problema de manera paralela.
 
  - Vamos a tratar de desarrollar las funciones para ambas opciones y derivado de esto vamos a 


## Equipo

Habiendo investigado esto podemos definir que el tipo de cluster que necesitamos es de MPI y en forma de Toro, por lo que utilizando la guía de la clase y nuestros créditos de 
AWS levantaremos un cluster de MPI para poder implementar el algoritmo. Cómo proximos pasos y avances deberemos tener el cluster de MPI y realizar pruebas para el envío de mensajes 
a través de este de tal manera que refleje la arquitectura de Toro.

