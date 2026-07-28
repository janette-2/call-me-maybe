<i>This project has been created as part of the 42 curriculum by \<janrodri\>.</i>

<br>

### Description:



### Instructions:



### Resources:

#### LLM_SDK:
The package of the LLM (Large Language Model) used [Qwen 0.6B] contains the following key features:

1.  encode(texto) → Convierte el texto input a tokens (números de IDs que el modelo entiende). Se usa una vez al principio para convertir el prompt en IDs. Devuelve un listado de la conversión de cada palabra a su ID del token correspondiente.
2.  get_logits_from_input_ids(lista de tokens) → El modelo procesa los IDs y devuelve un vector de ~150,000 logits o elementos float. Son puntuaciones crudas de cuán probable es que la siguiente palabra se adecúe al contexto de lo anterior. Cada elemento del vector representa cada token del vocabulario en sus probabilidades de ser el siguiente elemento. Se usa en cada análisis durante el loop de generación.
3. decode(lista de tokens) → Convierte los tokens que se han filtrado y recopilado en la última respuesta para pasarlo de vuelta a texto legible. Se usa una vez al final, cuando termina la generación.

El resto (get_path_to_*) no se necesita para el proyecto.

### Algorithm:


### Design:


### Performance Analysis:


### Challenges found:


### Tests Strategy:


### Usage examples:

