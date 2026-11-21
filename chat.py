"""
Chat en la terminal con un modelo de lenguaje local usando Ollama.

Conceptos que muestra este proyecto:
  1. Cómo se llama a un LLM desde código (la API de chat).
  2. El "historial": el modelo NO recuerda nada; cada vez le reenviamos
     toda la conversación.
  3. El "prompt de sistema": instrucciones que definen cómo se comporta.
  4. Streaming: mostrar la respuesta palabra por palabra mientras se genera.
  5. Métricas: tokens por segundo y tamaño del contexto.

Uso:
    python chat.py                 # usa el modelo por defecto
    python chat.py --modelo qwen3:8b
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import ollama
from rich.console import Console

# "instruct" responde directo. La variante "qwen3:4b" (Thinking) siempre razona antes
# de responder: es más lenta, pero útil para problemas de lógica o matemáticas.
MODELO_POR_DEFECTO = "qwen3:4b-instruct"

PROMPT_SISTEMA = (
    "Eres un asistente útil y directo. Respondes siempre en español, "
    "de forma clara y breve, salvo que el usuario pida más detalle."
)

CARPETA_CONVERSACIONES = Path(__file__).parent / "conversaciones"

AYUDA = """
Comandos disponibles:
  /ayuda     Muestra esta ayuda
  /limpiar   Borra el historial (el modelo "olvida" la conversación)
  /pensar    Activa/desactiva que el modelo muestre su razonamiento
  /guardar   Guarda la conversación en un archivo JSON
  /info      Muestra cuántos mensajes hay en el historial
  /salir     Termina el programa
"""

consola = Console()


def historial_inicial() -> list[dict]:
    """El historial siempre empieza con el mensaje de sistema."""
    return [{"role": "system", "content": PROMPT_SISTEMA}]


def guardar_conversacion(historial: list[dict], modelo: str) -> Path:
    """Guarda el historial en conversaciones/AAAA-MM-DD_HH-MM-SS.json."""
    CARPETA_CONVERSACIONES.mkdir(exist_ok=True)
    nombre = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json"
    ruta = CARPETA_CONVERSACIONES / nombre
    datos = {"modelo": modelo, "mensajes": historial}
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    return ruta


def responder(modelo: str, historial: list[dict], pensar: bool) -> str:
    """
    Envía TODO el historial al modelo y muestra la respuesta en streaming.
    Devuelve el texto completo de la respuesta para agregarlo al historial.
    """
    respuesta = ""
    mostrando_razonamiento = False
    ultimo = None

    # stream=True devuelve la respuesta en pedazos ("chunks") a medida que se genera.
    # think controla si los modelos que razonan (como Qwen3) piensan antes de responder.
    for pedazo in ollama.chat(model=modelo, messages=historial, stream=True, think=pensar):
        mensaje = pedazo.message

        # El razonamiento llega por un canal separado de la respuesta final.
        if pensar and mensaje.thinking:
            if not mostrando_razonamiento:
                consola.print("[dim italic](pensando...)[/]")
                mostrando_razonamiento = True
            consola.print(mensaje.thinking, end="", style="dim", markup=False, highlight=False)

        if mensaje.content:
            if mostrando_razonamiento:
                consola.print("\n")
                mostrando_razonamiento = False
            consola.print(mensaje.content, end="", markup=False, highlight=False)
            respuesta += mensaje.content

        ultimo = pedazo

    consola.print()

    # El último pedazo trae estadísticas: tokens generados, tiempo, tamaño del contexto.
    if ultimo is not None and ultimo.eval_count and ultimo.eval_duration:
        velocidad = ultimo.eval_count / (ultimo.eval_duration / 1e9)
        contexto = (ultimo.prompt_eval_count or 0) + ultimo.eval_count
        consola.print(
            f"[dim]{ultimo.eval_count} tokens · {velocidad:.1f} tokens/s · "
            f"contexto usado: ~{contexto} tokens[/]"
        )

    # Algunos modelos "filtran" su razonamiento dentro de la respuesta, terminado en </think>.
    # Lo quitamos antes de guardarlo en el historial para no gastar contexto en él.
    if "</think>" in respuesta:
        respuesta = respuesta.split("</think>", 1)[1].strip()

    return respuesta


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local con Ollama")
    parser.add_argument("--modelo", default=MODELO_POR_DEFECTO, help="Modelo de Ollama a usar")
    args = parser.parse_args()
    modelo = args.modelo

    # Comprobamos que Ollama esté corriendo y que el modelo exista.
    try:
        disponibles = [m.model for m in ollama.list().models]
    except Exception:
        consola.print("[red]No pude conectar con Ollama.[/] Abre la app de Ollama o ejecuta: ollama serve")
        return
    if modelo not in disponibles:
        consola.print(f"[red]El modelo '{modelo}' no está descargado.[/] Ejecuta: ollama pull {modelo}")
        consola.print(f"Modelos disponibles: {', '.join(disponibles) or '(ninguno)'}")
        return

    historial = historial_inicial()
    pensar = False

    consola.print(f"[bold green]Chat local con {modelo}[/]  (escribe /ayuda para ver los comandos)\n")

    while True:
        try:
            entrada = consola.input("[bold cyan]Tú:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            consola.print("\n¡Hasta luego!")
            break

        if not entrada:
            continue

        # --- Comandos ---
        if entrada.startswith("/"):
            comando = entrada.lower()
            if comando == "/salir":
                consola.print("¡Hasta luego!")
                break
            elif comando == "/ayuda":
                consola.print(AYUDA)
            elif comando == "/limpiar":
                historial = historial_inicial()
                consola.print("[yellow]Historial borrado.[/]")
            elif comando == "/pensar":
                pensar = not pensar
                consola.print(f"[yellow]Razonamiento {'activado' if pensar else 'desactivado'}.[/]")
            elif comando == "/guardar":
                ruta = guardar_conversacion(historial, modelo)
                consola.print(f"[yellow]Conversación guardada en {ruta}[/]")
            elif comando == "/info":
                consola.print(f"[yellow]Mensajes en el historial: {len(historial)} (incluye el de sistema)[/]")
            else:
                consola.print("[red]Comando desconocido.[/] Escribe /ayuda")
            continue

        # --- Conversación ---
        historial.append({"role": "user", "content": entrada})
        consola.print("[bold magenta]IA:[/] ", end="")
        try:
            respuesta = responder(modelo, historial, pensar)
        except KeyboardInterrupt:
            consola.print("\n[yellow]Respuesta interrumpida.[/]")
            historial.pop()  # quitamos la pregunta sin respuesta
            continue
        except Exception as error:
            consola.print(f"\n[red]Error al hablar con el modelo:[/] {error}")
            historial.pop()
            continue

        historial.append({"role": "assistant", "content": respuesta})
        consola.print()


if __name__ == "__main__":
    main()
