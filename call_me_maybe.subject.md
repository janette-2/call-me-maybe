

<!-- Start of picture text -->
y<br><!-- End of picture text -->

Introducción a las llamadas a función en LLMs 

call me maybe 

- Debes desarrollar el hábito de revisar, cuestionar y probar sistemáticamente cualquier contenido generado por la IA. 

- Debes buscar siempre la revisión de otras personas, no te limites a confiar en tu propia validación. 

## ● **Resultados de esta etapa:** 

- Desarrollar habilidades de prompting tanto generales como de ámbito especifico. 

- Aumentar tu productividad con un uso eficaz de las herramientas de IA. 

- Seguir fortaleciendo el pensamiento computacional, la resolución de problemas, la adaptabilidad y la colaboración. 

## ● **Comentarios y ejemplos:** 

- Ten en cuenta que la IA puede no tener la respuesta correcta porque esa respuesta no esté ni siquiera en Internet. Además, si te da soluciones incorrectas, intenta no insistir y busca ayuda entre las personas que te rodean. Vas a ahorrarte tiempo y vas a sumar en compresión. 

- Vas a enfretarte con frecuencia a situaciones (como exámenes o evaluaciones) donde debes demostrar una comprensión real. Prepárate, sigue construyendo tanto tus habilidades técnicas como transversales. 

- Explicar tu razonamiento y debatir con otras personas suele revelar lagunas en tu comprensión de un concepto. Prioriza el aprendizaje entre pares. 

- Lo normal es que la herramienta de IA que utilices no conozca tu contexto específico (a menos que se lo indiques), así que te dará respuestas genéricas. Si buscas información más adecuada y más precisa en relación a tu entorno cercano, confía en el resto de estudiantes. 

- Donde la IA tiende a generar la respuesta más probable, el resto de estudiantes puede proporcionar perspectivas alternativas y matices valiosos. Confía en la comunidad de 42 como un punto de control de calidad. 

#### ✓ **Buenas prácticas:** 

Le pregunto a la IA: "¿Cómo pruebo una función de ordenación?"Me da algunas ideas. Las pruebo y reviso los resultados con otra persona. Refinamos el enfoque de manera conjunta. 

#### ✗ **Mala práctica:** 

Le pido a la IA que escriba una función completa, la copio y la pego en mi proyecto. Durante la evaluación entre pares, no puedo explicar qué hace ni por qué. Pierdo credibilidad. Suspendo mi proyecto. 

4 

Introducción a las llamadas a función en LLMs 

call me maybe 

#### ✓ **Buenas prácticas:** 

Utilizo la IA para ayudarme a diseñar un parser. Luego, reviso la lógica con otra persona. Encontramos dos errores y lo reescribimos juntos: mejor, más limpio y comprendiendo al 100 

#### ✗ **Mala práctica:** 

Dejo que Copilot genere mi código para una parte clave de mi proyecto. Compila, pero no puedo explicar cómo maneja los pipes. Durante la evaluación, no puedo justificarlo y suspendo mi proyecto. 



<!-- Start of picture text -->
5<br><!-- End of picture text -->

# **Capítulo III** 

# **Introducción** 

## **III.1. ¿Qué es la llamada a función?** 

Los Large Language Models (LLMs) son muy potentes para entender y generar lenguaje humano, pero la respuesta no es una salida estructurada y ejecutable por una máquina. Los sistemas de llamada a función cubren esta brecha, traduciendo las peticiones en lenguaje natural en llamadas a funciones precisas con argumentos tipados. 

Por ejemplo: 

Del lenguaje natural a la llamada a función 

```
Peticionenlenguajenatural:"Cualeslasumade40y2?"
RespuestadeunLLMnormal:"Lasumade40y2es42."
Respuestadeunsistemadellamadaafuncion:
{
"function":"add_numbers",
"arguments":{"a":40,"b":2}
}
```

El sistema de llamada a función no responde a la pregunta directamente. En su lugar, proporciona las **herramientas** para resolverla: el nombre correcto de la función y los argumentos adecuados con los tipos apropiados. 

6 



# **Capítulo V** 

# **Parte obligatoria** 

## **V.1. Resumen** 

En este proyecto, se creará una herramienta de llamada a función que traduzca peticiones en lenguaje natural en llamadas a funciones estructuradas. Dada una pregunta como "What is the sum of 40 and 2?", la solución no debe devolver 42, sino proporcionar: 

- El nombre de la función: `fn_add_numbers` 

- Los argumentos: `.`<sup>`a`</sup> `": 40, "b": 2` 

La implementación debe usar **decodificación restringida** para garantizar un JSON válido al 100 %, asegurando una fiabilidad casi perfecta incluso con un modelo pequeño de 0.5B parámetros. 

## **V.2. Archivos de entrada** 

La solución procesará los dos archivos de entrada situados en el directorio `data/input/` : 

### **V.2.1. Pruebas de llamadas a funciones** 

El archivo `data/input/function_calling_tests.json` contiene un array JSON de prompts en lenguaje natural que el programa debe procesar. 

Ejemplo: function_calling_tests.json 

```
[
```

```
"Whatisthesumof2and3?",
```

```
"Reversethestring'hello'",
```

```
"Calculatethefactorialof5"
```

```
]
```

11 

Introducción a las llamadas a función en LLMs 

call me maybe 

### **V.2.2. Definiciones de funciones** 

El archivo `data/input/function_definitions.json` contiene las funciones disponibles que el sistema puede llamar. Cada función incluye: 

- Nombre de la función. 

- Nombres de los argumentos y tipos. 

- Tipo de retorno. 

- Descripción. 

Ejemplo: function_definitions.json 

```
[
{
name":"fn_add_numbers",
"description":"Addtwonumbers",
"parameters":{
"a":{"type":"number"},
"b":{"type":"number"}
},
"returns":{"type":"number"}
},
{
"name":"fn_reverse_string",
"description":"Reverseastring",
"parameters":{
"s":{"type":"string"}
},
"returns":{"type":"string"}
}
]
```



```
Estosejemplossirvencomoreferenciadelniveldecomplejidad
esperado.Sinembargo,lasolucióncreadaseprobarácondistintos
promptsyconjuntosdefunciones.Sedebeimplementarunagestión
adecuadadeerroresdeJSONparalosarchivosdeentrada,yaque
puedencontenerJSONnoválidosodirectamentenoexistir.
```



<!-- Start of picture text -->
12<br><!-- End of picture text -->



<!-- Start of picture text -->
AN<br>O<br><!-- End of picture text -->

Introducción a las llamadas a función en LLMs 

call me maybe 

## **V.4. Formato del archivo de salida** 

El programa producirá un único archivo JSON: `output/function_calling_results.json` . Para cada prompt, se debe añadir un objeto JSON a este archivo. Cada objeto del array debe contener exactamente las siguientes claves: 

- `prompt` (string): la petición original en lenguaje natural. 

- `fn_name` (string): el nombre de la función a llamar. 

- `args` (object): todos los argumentos requeridos con los tipos correctos. 

### **V.4.1. Ejemplo de salida** 

data/output/function_calling_results.json 

```
[
```

```
{
```

```
"prompt":"Whatisthesumof2and3?",
```

```
"fn_name":"fn_add_numbers",
"args":{"a":2.0,"b":3.0}
```

```
},
{
```

```
"prompt":"Reversethestring'hello'",
"fn_name":"fn_reverse_string",
"args":{"s":"hello"}
}
```

```
]
```

### **V.4.2. Reglas de validación** 

- El archivo debe ser un JSON válido (sin comas finales, sin comentarios). 

- Las claves y tipos deben coincidir exactamente con el esquema de `function_definitions.json` . 

- No se permiten claves adicionales ni texto libre en ninguna parte de la salida. 

- Todos los argumentos requeridos deben estar presentes. 

- Los tipos de los argumentos deben coincidir con los de la definición de la función (number, string, boolean, etc.). 



```
Losarchivosdeentradaproporcionadospuedencambiardurantela
evaluación.Nohardcodeessolucionesbasadasenlosejemplosdados.
```

16 

Introducción a las llamadas a función en LLMs 

call me maybe 

## **V.5. Rendimiento y fiabilidad** 

La implementación debería alcanzar: 

- **Precisión casi perfecta** : más del 95 % de selección correctade función y argumentos. 

- **JSON válido al 100 %** : toda la salida debe ser parseable y cumplir el esquema. 

- **Velocidad razonable** : procesar todos los prompts de prueba en menos de 5 minutos. 

- **Gestión robusta de errores** : manejar de forma correcta entradas mal formateadas, archivos ausentes y casos límite. 



```
ElmodeloQwen3-0.6Btienesolo500millonesdeparámetrosy,aun
así,conunadecodificaciónrestringidaadecuadapuedealcanzar
unafiabilidadcomparablealademodelosmuchomásgrandes.Esto
demuestraelpoderdeunaguíaestructuralfrentealafuerzabruta
deltamañodelmodelo.
```

## **V.6. Probar la implementación** 

Para verificar que la solución funciona correctamente, hay que hacer las siguientes comprobaciones: 

1. Asegurar que los archivos de entrada están en el directorio `input/` . 

2. Ejecutar: `uv run python -m src` . 

3. Comprobar que se ha creado `output/function_calling_results.json` . 

4. Validar la estructura y el contenido del JSON. 

5. Verificar que los nombres de las funciones y los tipos de los argumentos coinciden con las definiciones. 



```
Hazpruebascondistintoscasoslímite:cadenasvacías,números
grandes,caracteresespeciales,promptsambiguosyfuncionescon
múltiplesparámetros.
```

17 



# **Capítulo VII** 

# **evaluación entre Entrega y pares** 

Entrega tu trabajo en tu repositorio `Git` como de costumbre. Solo se evaluará durante la defensa el trabajo que haya dentro de tu repositorio. No dudes en comprobar dos veces los nombres de tus archivos para asegurarte de que son correctos. 

Tu repositorio debe contener: 

- El directorio `src/` con tu implementación 

- `pyproject.toml` y `uv.lock` para la gestión de dependencias 

- El directorio `llm_sdk/` (copiado del paquete proporcionado) 

- El directorio `data/input/` con archivos de prueba (para demostración) 

- `README.md` con documentación completa 

- Cualesquiera archivos adicionales necesarios para ejecutar tu solución 



```
Noincluyaseldirectoriooutput/enturepositorio.Segenerará
durantelaevaluación.
```

## **VII.1. Instrucciones de recodificación** 

Durante la evaluación, es posible que se solicite una ligera **modificación del proyecto** . Esto puede consistir en ajustar ligeramente el comportamiento, modificar unas cuantas líneas de código o incorporar una característica fácil de implementar. 

Puede que este paso **no sea necesario en todos los proyectos** , pero hay que tenerlo en cuenta si así se especifica en la hoja de evaluación. 

Este paso sirve para verificar la comprensión real de una parte específica del proyecto. La modificación se puede realizar en cualquier entorno de desarrollo que se elija (por ejemplo, la configuración habitual), y debería ser factible en unos pocos minutos, a menos que se 

20 

