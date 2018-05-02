## Alex 
Como definimos y comentamos en el avance anterior, compararemos los resultados de la implementación secuencial con la
implementación en paralelo para distintos tamaños de matrices con la finalidad de identificar en qué puntos es más eficiente
una implementación con respecto a la otra. Por ejemplo identificar que el código secuencial es más eficiente para matrices
de tamaño menor a 1000 y después de 1000 ya gana el código paralelo. Es por esto que realicé un código en bash, que en cada
corrida genera las matrices A y B (utilizando el código que generó Patricio en el avance pasado) y realiza la multiplicación
de matrices secuencial y en paralelo, registrando los resultados:

  + Tiempo que toma realizar la operación.
  + Dimensiones de las matrices.
  + Threads utilizados (para el caso paralelo).

Con esto ya podemos comenzar a realizar las distintas pruebas y comparaciones que deseamos hacer para el trabajo.

La forma en que correríamos el código sería "sh loop.sh >> resultados.csv" y de esta forma genera una tabla con los resultados.
