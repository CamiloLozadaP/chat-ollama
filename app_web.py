"""
Interfaz web del chat, hecha con Gradio.

Usa exactamente la misma lógica que el chat de la terminal (nucleo.py).

Uso:
    python app_web.py
Luego abre http://127.0.0.1:7860 en el navegador (se abre solo).
"""

import gradio as gr

import nucleo


def modelos_disponibles() -> list[str]:
    try:
        return nucleo.listar_modelos() or [nucleo.MODELO_POR_DEFECTO]
    except Exception:
        return [nucleo.MODELO_POR_DEFECTO]


def conversaciones_guardadas() -> list[str]:
    return [ruta.stem for ruta in nucleo.listar_conversaciones()]


def panel_memoria(conv: nucleo.Conversacion | None) -> str:
    """Texto del panel lateral: tokens, memoria y velocidad."""
    if conv is None:
        return "_Todavía no hay conversación._"
    lineas = [
        f"**Tokens enviados al modelo:** ~{conv.tokens_estimados()} (se resume al pasar de {conv.limite_tokens})",
        f"**Mensajes recientes:** {len(conv.mensajes)} · **Veces resumida:** {conv.veces_resumida}",
    ]
    est = conv.ultimas_estadisticas
    if est:
        lineas.append(f"**Última respuesta:** {est.tokens_generados} tokens · {est.tokens_por_segundo:.1f} tokens/s")
    lineas.append("")
    lineas.append("**Resumen guardado:**")
    lineas.append(conv.resumen if conv.resumen else "_Todavía no hay resumen._")
    return "\n\n".join(lineas)


def preparar(conv, modelo, personalidad):
    """Crea la conversación la primera vez, o aplica los cambios de modelo/personalidad."""
    if conv is None:
        return nucleo.Conversacion(modelo=modelo, personalidad=personalidad)
    conv.modelo = modelo
    if personalidad != conv.personalidad:
        conv.cambiar_personalidad(personalidad)
    return conv


def enviar(texto, chat, conv, modelo, personalidad, pensar):
    texto = (texto or "").strip()
    if not texto:
        yield "", chat, conv, panel_memoria(conv)
        return

    conv = preparar(conv, modelo, personalidad)
    chat = chat + [{"role": "user", "content": texto}]
    razonamiento = None
    respuesta = {"role": "assistant", "content": ""}

    try:
        for fragmento in conv.enviar(texto, pensar=pensar):
            if fragmento.tipo == "razonamiento":
                if razonamiento is None:
                    # Gradio muestra los mensajes con "metadata.title" como un bloque plegable.
                    razonamiento = {"role": "assistant", "content": "", "metadata": {"title": "Razonamiento"}}
                razonamiento["content"] += fragmento.texto
            else:
                respuesta["content"] += fragmento.texto
            visibles = ([razonamiento] if razonamiento else []) + [respuesta]
            yield "", chat + visibles, conv, panel_memoria(conv)
    except Exception as error:
        gr.Warning(f"Error al hablar con el modelo: {error}")
        yield texto, chat[:-1], conv, panel_memoria(conv)
        return

    chat = chat + ([razonamiento] if razonamiento else []) + [respuesta]

    if conv.necesita_resumen():
        gr.Info("Resumiendo la memoria para ahorrar tokens...")
        conv.resumir()

    yield "", chat, conv, panel_memoria(conv)


def limpiar(conv):
    if conv is not None:
        conv.limpiar()
    return [], conv, panel_memoria(conv)


def guardar(conv):
    if conv is None or not conv.mensajes:
        gr.Warning("No hay nada que guardar todavía.")
        return gr.update()
    ruta = conv.guardar()
    gr.Info(f"Conversación guardada: {ruta.stem}")
    return gr.update(choices=conversaciones_guardadas(), value=ruta.stem)


def cargar(nombre):
    if not nombre:
        gr.Warning("Elige una conversación guardada.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    ruta = nucleo.CARPETA_CONVERSACIONES / f"{nombre}.json"
    conv = nucleo.Conversacion.cargar(ruta)
    chat = [{"role": m["role"], "content": m["content"]} for m in conv.mensajes]
    if conv.resumen:
        chat.insert(0, {"role": "assistant", "content": conv.resumen,
                        "metadata": {"title": "Memoria de la conversación anterior"}})
    gr.Info(f"Conversación '{nombre}' cargada.")
    personalidad = conv.personalidad if conv.personalidad in nucleo.listar_personalidades() else gr.update()
    return chat, conv, panel_memoria(conv), conv.modelo, personalidad


def crear_interfaz() -> gr.Blocks:
    modelos = modelos_disponibles()
    modelo_inicial = nucleo.MODELO_POR_DEFECTO if nucleo.MODELO_POR_DEFECTO in modelos else modelos[0]

    with gr.Blocks(title="Chat local con Ollama") as demo:
        gr.Markdown("# Chat local con Ollama\nModelos de IA corriendo en tu propio computador, sin costo por token.")
        conv = gr.State(None)

        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(label="Conversación", height=520)
                with gr.Row():
                    entrada = gr.Textbox(placeholder="Escribe tu mensaje y presiona Enter...",
                                         show_label=False, scale=5, autofocus=True)
                    boton_enviar = gr.Button("Enviar", variant="primary", scale=1)

            with gr.Column(scale=1):
                modelo = gr.Dropdown(modelos, value=modelo_inicial, label="Modelo")
                personalidad = gr.Dropdown(nucleo.listar_personalidades(),
                                           value=nucleo.PERSONALIDAD_POR_DEFECTO, label="Personalidad")
                pensar = gr.Checkbox(False, label="Mostrar razonamiento (solo modelos que razonan)")
                boton_limpiar = gr.Button("Limpiar conversación")

                with gr.Accordion("Memoria y tokens", open=True):
                    memoria = gr.Markdown(panel_memoria(None))

                with gr.Accordion("Conversaciones guardadas", open=False):
                    boton_guardar = gr.Button("Guardar conversación actual")
                    guardadas = gr.Dropdown(conversaciones_guardadas(), label="Elegir conversación")
                    boton_cargar = gr.Button("Cargar")

        entradas = [entrada, chat, conv, modelo, personalidad, pensar]
        salidas = [entrada, chat, conv, memoria]
        entrada.submit(enviar, entradas, salidas)
        boton_enviar.click(enviar, entradas, salidas)
        boton_limpiar.click(limpiar, [conv], [chat, conv, memoria])
        boton_guardar.click(guardar, [conv], [guardadas])
        boton_cargar.click(cargar, [guardadas], [chat, conv, memoria, modelo, personalidad])

    return demo


if __name__ == "__main__":
    crear_interfaz().launch(inbrowser=True)
