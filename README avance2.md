## Alejandro Pérez Palacios

Encontré en las ligas:

  * http://www.cs.nthu.edu.tw/~cherung/teaching/2011anm/note02.pdf 
  * http://solomonik.cs.illinois.edu/talks/vecpar-2012.pdf
  
instrucciones para poder realizar la multiplicación de matrices en paralelo, adems revisando las notas del curso sobre multiplicación de matrices en paralelo
noté que se encuentra descrito el algoritmo de cannon de una manera gráfica y más clara. 

Viendo las descripciones y ms fuentes noto que este tipo de código requiere de especificar la comunicación entre los nodos o procesadores
a través de mensajes send y receive por lo que es necesario que utilicemos message passing (MPI) para implementarlo, entre las ventajas
que mas se mencionan para el método de Cannon es que su arquitectura minimiza la comunicación entre los nodos, lo cual hace más facil la programación
de los mensajes a través de estos.

De igual manera vemos que en las notas y referencias menciona que la arquitectura que necesita este algoritmo es llamada Toro por la forma en que los mensajes entre los nodos
son enviados. 

## Patricio Hernández 

Continuando con la investigación veo el algoritmo de cannon de las notas y su descripción, entendiendo....

Chanc aquí puedes hacer un resumen del algoritmo y pseudo cdigo jaja.


## Equipo

Habiendo investigado esto podemos definir que el tipo de cluster que necesitamos es de MPI y en forma de Toro, por lo que utilizando la guía de la clase y nuestros créditos de 
AWS levantaremos un cluster de MPI para poder implementar el algoritmo. Cómo proximos pasos y avances deberemos tener el cluster de MPI y realizar pruebas para el envío de mensajes 
a través de este de tal manera que refleje la arquitectura de Toro.

