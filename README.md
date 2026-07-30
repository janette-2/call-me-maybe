<i>This project has been created as part of the 42 curriculum by \<janrodri\>.</i>

<br>

### Description:

**call me maybe** convierte peticiones en lenguaje natural en llamadas a funciones
estructuradas (JSON), usando un LLM pequeño (Qwen3-0.6B) combinado con
decodificación restringida. El modelo **no responde la pregunta**, sino que
**identifica qué función llamar y con qué argumentos**.

### Instrucciones del LLM:

El LLM recibe el prompt del usuario y genera una secuencia de tokens que forma
un JSON con la estructura: `{"fn_name": "...", "args": {...}}`. No se usa como
chatbot o autocompletado — se usa como un **motor de procesamiento semántico**:
dado lenguaje natural, decide qué nombre de función corresponde a la consulta que se la ha pasado y qué argumentos son
los correspondientes a los pasados a esa función.

La **decodificación restringida** fuerza al modelo a que cada token generado
forme parte del JSON esperado. En cada paso, los tokens que no cumplen la
gramática reciben `-inf` en sus logits, haciendo imposible que el modelo los
elija. Así se garantiza JSON 100% válido incluso con un modelo de 500M
parámetros.

### Instrucciones:



### Resources:

#### LLM_SDK:
The package of the LLM (Large Language Model) used [Qwen 0.6B] contains the following key features:

**1.  encode(texto)** → Convierte el texto input a tokens (números de IDs que el modelo entiende). Se usa una vez al principio para convertir el prompt en IDs. Devuelve un listado de la conversión de cada palabra a su ID del token correspondiente.

**2.  get_logits_from_input_ids(lista de tokens)** → El modelo procesa los IDs y devuelve un vector de ~150,000 logits o elementos float. Son puntuaciones crudas de cuán probable es que la siguiente palabra se adecúe al contexto de lo anterior. Cada elemento del vector representa cada token del vocabulario en sus probabilidades de ser el siguiente elemento. Se usa en cada análisis durante el loop de generación.

**3. decode(lista de tokens)** → Convierte los tokens que se han filtrado y recopilado en la última respuesta para pasarlo de vuelta a texto legible. Se usa una vez al final, cuando termina la generación.

El resto (get_path_to_*) no se necesita para el proyecto.

### Algorithm:


### Design:


### Performance Analysis:


### Challenges found:

**1. Entender qué pasa dentro del LLM cuando "pasa por el modelo"**
Inicialmente no lograba visualizar qué ocurría internamente entre que entran
los tokens de entrada y salen los logits. Exploramos las tres capas internas:
Embedding (cada token → vector denso), Transformer (28 bloques de Self-Attention
+ FFN donde los tokens se contextualizan entre sí), y LM Head (el vector del
último token se proyecta al vocabulario completo generando ~150,000 logits).
Conclusión: el LLM es una máquina de predecir el siguiente token, y los logits
son puntuaciones crudas de esa predicción.

**2. Cuándo se usa decode en el flujo de generación**
Pensaba que decode se usaba paso a paso durante el loop. Descubrimos que no:
el loop entero trabaja con IDs numéricos, y decode se aplica una sola vez al
final para convertir la secuencia completa de IDs a texto legible.

**3. Propósito del LLM en el proyecto (¿por qué no regex?)**
Dudaba de por qué necesitábamos un LLM si podríamos parsear con regex.
Conclusión: el LLM aporta flexibilidad semántica para entender lenguaje natural
variado ("What is the sum of 2 and 3?" vs "Add 2 and 3" vs "Calculate 2+3")
sin hardcodear patrones. La decodificación restringida garantiza que la salida
sea JSON válido, y el LLM solo decide qué función y argumentos corresponden.

**4. Cómo hacer que el LLM conozca las funciones disponibles**
No veía claro cómo unir las definiciones de funciones con el prompt del usuario.
Llegamos a la solución del super-prompt: construir un mensaje que incluya
instrucciones del sistema + listado de funciones disponibles con sus parámetros
+ el prompt original del usuario. Esto orienta al modelo sin necesidad de
fine-tuning.

**5. Diferencia entre super-prompt y logit masking**
Confundía ambos conceptos como si fueran redundantes. Aclaramos que son
complementarios: el super-prompt orienta semánticamente al modelo (señal
débil, el modelo "intuye" lo que debe hacer), mientras que el logit masking
fuerza mecánicamente la estructura (garantía fuerte, el modelo no puede
desviarse aunque quiera).

**6. Formato de salida del LLM: semi-estructurado vs plantilla fija**
Dilema sobre cómo debería generar el LLM la respuesta: ¿en un formato
semi-estructurado tipo `fn_add_numbers(a=2, b=3)` para luego parsearlo?
¿O mejor separar la decisión del LLM de la construcción del JSON?
Conclusión: optamos por una plantilla JSON fija donde el programa fuerza
las partes invariables (`{`, `"fn_name"`, `"args"`, etc.) y el LLM solo
decide los valores variables (nombre de función y argumentos).

**7. Cómo averiguar los IDs de cada token (Token discovery)**
Para forzar tokens específicos en el loop, necesitábamos saber sus IDs
numéricos dentro del vocabulario de Qwen3 (~150,000 tokens). La solución
fue usar `encode("símbolo")` una vez al inicio para mapear cada pieza fija
a sus IDs, construyendo así un diccionario de tokens reutilizable.

**8. Detección de confianza baja del modelo**
Preocupación sobre qué hacer cuando el prompt no se corresponde con ninguna
función disponible. Exploramos la idea de analizar la distribución de
probabilidad: si el mejor nombre de función tiene una probabilidad baja
y las demás están repartidas, el modelo no tiene clara la respuesta.
Queda pendiente definir el umbral exacto y la acción a tomar.

**9. Reglas de formato JSON**
Dudas sobre si hay reglas de ordenación de claves, indentación o restricciones
adicionales. Conclusión: el orden de las claves en un objeto JSON no importa
(aunque nuestra plantilla fuerza uno concreto), y las únicas reglas estrictas
son: comillas dobles en claves y strings, sin comas finales, sin comentarios,
100% parseable por `json.loads()`.

**10. Separar decisión del LLM vs partes forzadas del programa**
Costó distinguir qué genera el LLM (solo nombre de función y valores de
argumentos) y qué fuerza el programa directamente (llaves, comillas, comas,
nombres de clave como `fn_name` y `args`). Conclusión: el programa recorre
el autómata de estados y en los estados "fijos" fuerza el token directamente;
en los estados "de decisión" deja que el LLM elija entre opciones limitadas.

**11. Manejo de prompts sin función correspondiente**
Si el modelo recibe un prompt que no encaja con ninguna función, siempre va
a generar algo (no puede "callarse"). La solución planteada es: si la
probabilidad máxima entre los nombres de función es baja, detectarlo y
manejarlo como caso especial. Aún por definir la implementación concreta.


### Tests Strategy:


### Usage examples:


## NOTAS:

Chars a sacar los IDs para guardarlos en diccionario:

"{"
"}"
":"
","
" "
"\n"
"fn_name"
"args"
"fn_add_numbers"
"fn_greet"
"fn_reverse_string"
"fn_get_square_root"
"fn_substitute_string_with_regex"
"a"
"b"
"name"
"s"
"source_string"
"regex"
"replacement"