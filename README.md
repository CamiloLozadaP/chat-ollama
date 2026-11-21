# Chat local con Ollama

Chat en la terminal con un modelo de lenguaje que corre **100 % en tu computador**, sin internet y sin costo por tokens, usando [Ollama](https://ollama.com) y Python.

Es el primer proyecto de una serie para aprender a construir aplicaciones con IA.

## Qué aprendí con este proyecto

- **Cómo se llama a un LLM desde código** con la API de chat de Ollama.
- **El modelo no tiene memoria.** Cada vez se le reenvía todo el historial de la conversación, y por eso el contexto (y el costo, en modelos de pago) crece con cada mensaje.
- **El prompt de sistema** define la personalidad y las reglas del asistente.
- **Streaming:** mostrar la respuesta mientras se genera.
- **Modelos que razonan** (como Qwen3): pueden "pensar" antes de responder. Eso mejora algunas respuestas, pero gasta más tokens y tiempo.
- **Métricas:** tokens por segundo y tamaño del contexto usado.

## Requisitos

- [Ollama](https://ollama.com/download) instalado y corriendo
- Python 3.10 o superior
- Un modelo descargado, por ejemplo:

  ```bash
  ollama pull qwen3:4b-instruct
  ```

  `qwen3:4b-instruct` responde directo. La variante `qwen3:4b` es de tipo "Thinking": siempre razona antes de responder, así que es más lenta pero mejor en lógica y matemáticas.

## Instalación

```bash
git clone <url-de-este-repositorio>
cd chat-ollama
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Uso

```bash
python chat.py                    # usa qwen3:4b-instruct
python chat.py --modelo qwen3:4b  # variante que razona antes de responder
```

### Comandos dentro del chat

| Comando | Qué hace |
|---|---|
| `/ayuda` | Muestra los comandos |
| `/limpiar` | Borra el historial: el modelo "olvida" la conversación |
| `/pensar` | Activa o desactiva el razonamiento visible del modelo |
| `/guardar` | Guarda la conversación en `conversaciones/` como JSON |
| `/info` | Muestra cuántos mensajes hay en el historial |
| `/salir` | Termina el programa |

## Probado en

- Portátil con NVIDIA RTX 4050 (6 GB de VRAM) y 16 GB de RAM
- Modelos `qwen3:4b-instruct` y `qwen3:4b`, corriendo 100 % en la GPU

## Próximos pasos

- [ ] Recortar el historial cuando el contexto se acerque al límite del modelo
- [ ] Interfaz web con Streamlit o Gradio
- [ ] Comparar varios modelos con las mismas preguntas
