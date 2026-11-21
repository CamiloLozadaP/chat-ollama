"""
Chat en la terminal con un modelo de lenguaje local usando Ollama.

Toda la lógica (historial, memoria resumida, guardar/cargar) está en nucleo.py.
Este archivo solo se encarga de leer lo que escribes y mostrar las respuestas.

Uso:
    python chat.py
    python chat.py --modelo qwen3:4b --personalidad profesor
    python chat.py --limite 800      # resume antes, para ver la memoria en acción
"""

import argparse

from rich.console import Console

import nucleo

AYUDA = """
Comandos disponibles:
  /ayuda                 Muestra esta ayuda
  /limpiar               Borra la conversación y la memoria
  /pensar                Activa/desactiva el razonamiento visible (modelos que razonan)
  /personalidad          Lista las personalidades disponibles
  /personalidad NOMBRE   Cambia de personalidad sin borrar la conversación
  /memoria               Muestra el resumen guardado y los tokens usados
  /resumir               Fuerza un resumen de la conversación ahora
  /guardar               Guarda la conversación en un archivo JSON
  /cargar                Lista las conversaciones guardadas
  /cargar NÚMERO         Retoma una conversación guardada
  /info                  Muestra modelo, personalidad y tamaño del historial
  /salir                 Termina el programa
"""

consola = Console()


def mostrar_respuesta(conversacion: nucleo.Conversacion, texto: str, pensar: bool) -> None:
    """Muestra la respuesta en streaming, con el razonamiento en gris."""
    consola.print("[bold magenta]IA:[/] ", end="")
    pensando = False
    for fragmento in conversacion.enviar(texto, pensar=pensar):
        if fragmento.tipo == "razonamiento":
            if not pensando:
                consola.print("[dim italic](pensando...)[/]")
                pensando = True
            consola.print(fragmento.texto, end="", style="dim", markup=False, highlight=False)
        else:
            if pensando:
                consola.print("\n")
                pensando = False
            consola.print(fragmento.texto, end="", markup=False, highlight=False)
    consola.print()

    est = conversacion.ultimas_estadisticas
    if est:
        consola.print(
            f"[dim]{est.tokens_generados} tokens · {est.tokens_por_segundo:.1f} tokens/s · "
            f"contexto: ~{est.tokens_contexto} tokens (límite para resumir: {conversacion.limite_tokens})[/]"
        )


def resumir_con_aviso(conversacion: nucleo.Conversacion) -> None:
    antes = conversacion.tokens_estimados()
    with consola.status("Resumiendo la memoria..."):
        conversacion.resumir()
    despues = conversacion.tokens_estimados()
    consola.print(
        f"[yellow]Memoria resumida: ~{antes} → ~{despues} tokens "
        f"(se conservan los últimos {conversacion.mensajes_recientes} mensajes completos).[/]"
    )


def ejecutar_comando(entrada: str, conversacion: nucleo.Conversacion, estado: dict):
    """
    Procesa un comando que empieza con '/'.
    Devuelve la conversación (puede ser otra si se cargó una guardada) o None para salir.
    """
    partes = entrada.split(maxsplit=1)
    comando = partes[0].lower()
    argumento = partes[1].strip() if len(partes) > 1 else ""

    if comando == "/salir":
        return None

    if comando == "/ayuda":
        consola.print(AYUDA)

    elif comando == "/limpiar":
        conversacion.limpiar()
        consola.print("[yellow]Conversación y memoria borradas.[/]")

    elif comando == "/pensar":
        estado["pensar"] = not estado["pensar"]
        consola.print(f"[yellow]Razonamiento {'activado' if estado['pensar'] else 'desactivado'}.[/]")
        if estado["pensar"] and "instruct" in conversacion.modelo:
            consola.print("[dim]Nota: los modelos 'instruct' no razonan. Prueba con --modelo qwen3:4b[/]")

    elif comando == "/personalidad":
        if not argumento:
            for nombre in nucleo.listar_personalidades():
                marca = " (actual)" if nombre == conversacion.personalidad else ""
                consola.print(f"  - {nombre}{marca}")
            consola.print("[dim]Uso: /personalidad NOMBRE[/]")
        else:
            try:
                conversacion.cambiar_personalidad(argumento)
                consola.print(f"[yellow]Personalidad cambiada a '{argumento}'.[/]")
            except ValueError as error:
                consola.print(f"[red]{error}[/]")

    elif comando == "/memoria":
        consola.print(f"[yellow]Tokens enviados al modelo: ~{conversacion.tokens_estimados()} "
                      f"(se resume al pasar de {conversacion.limite_tokens})[/]")
        consola.print(f"[yellow]Veces resumida: {conversacion.veces_resumida}[/]")
        if conversacion.resumen:
            consola.print("[bold]Resumen guardado:[/]")
            consola.print(conversacion.resumen, markup=False, highlight=False)
        else:
            consola.print("[dim]Todavía no hay resumen.[/]")

    elif comando == "/resumir":
        if len(conversacion.mensajes) <= conversacion.mensajes_recientes:
            consola.print(f"[yellow]Hay muy pocos mensajes para resumir "
                          f"(se conservan siempre los últimos {conversacion.mensajes_recientes}).[/]")
        else:
            resumir_con_aviso(conversacion)

    elif comando == "/guardar":
        ruta = conversacion.guardar()
        consola.print(f"[yellow]Conversación guardada en {ruta.name}[/]")

    elif comando == "/cargar":
        guardadas = nucleo.listar_conversaciones()
        if not guardadas:
            consola.print("[yellow]No hay conversaciones guardadas.[/]")
        elif not argumento:
            for i, ruta in enumerate(guardadas, start=1):
                consola.print(f"  {i}. {ruta.stem}")
            consola.print("[dim]Uso: /cargar NÚMERO[/]")
        elif argumento.isdigit() and 1 <= int(argumento) <= len(guardadas):
            ruta = guardadas[int(argumento) - 1]
            nueva = nucleo.Conversacion.cargar(
                ruta,
                limite_tokens=conversacion.limite_tokens,
                mensajes_recientes=conversacion.mensajes_recientes,
            )
            consola.print(f"[yellow]Conversación '{ruta.stem}' cargada "
                          f"({len(nueva.mensajes)} mensajes, modelo {nueva.modelo}).[/]")
            return nueva
        else:
            consola.print("[red]Número no válido.[/] Escribe /cargar para ver la lista.")

    elif comando == "/info":
        consola.print(f"[yellow]Modelo: {conversacion.modelo} · Personalidad: {conversacion.personalidad} · "
                      f"Mensajes recientes: {len(conversacion.mensajes)} · "
                      f"Razonamiento: {'sí' if estado['pensar'] else 'no'}[/]")

    else:
        consola.print("[red]Comando desconocido.[/] Escribe /ayuda")

    return conversacion


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat local con Ollama")
    parser.add_argument("--modelo", default=nucleo.MODELO_POR_DEFECTO, help="Modelo de Ollama a usar")
    parser.add_argument("--personalidad", default=nucleo.PERSONALIDAD_POR_DEFECTO,
                        help="Archivo de la carpeta personalidades/ (sin .md)")
    parser.add_argument("--limite", type=int, default=nucleo.LIMITE_TOKENS,
                        help="Tokens a partir de los cuales se resume la memoria")
    args = parser.parse_args()

    # Comprobamos que Ollama esté corriendo y que el modelo exista.
    try:
        disponibles = nucleo.listar_modelos()
    except Exception:
        consola.print("[red]No pude conectar con Ollama.[/] Abre la app de Ollama o ejecuta: ollama serve")
        return
    if args.modelo not in disponibles:
        consola.print(f"[red]El modelo '{args.modelo}' no está descargado.[/] Ejecuta: ollama pull {args.modelo}")
        consola.print(f"Modelos disponibles: {', '.join(disponibles) or '(ninguno)'}")
        return

    try:
        conversacion = nucleo.Conversacion(
            modelo=args.modelo, personalidad=args.personalidad, limite_tokens=args.limite
        )
    except ValueError as error:
        consola.print(f"[red]{error}[/]")
        return

    estado = {"pensar": False}
    consola.print(
        f"[bold green]Chat local con {conversacion.modelo}[/] · personalidad: {conversacion.personalidad}  "
        f"(escribe /ayuda para ver los comandos)\n"
    )

    while True:
        try:
            entrada = consola.input("[bold cyan]Tú:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            consola.print("\n¡Hasta luego!")
            break

        if not entrada:
            continue

        if entrada.startswith("/"):
            conversacion = ejecutar_comando(entrada, conversacion, estado)
            if conversacion is None:
                consola.print("¡Hasta luego!")
                break
            continue

        try:
            mostrar_respuesta(conversacion, entrada, estado["pensar"])
        except KeyboardInterrupt:
            consola.print("\n[yellow]Respuesta interrumpida.[/]")
            continue
        except Exception as error:
            consola.print(f"\n[red]Error al hablar con el modelo:[/] {error}")
            continue

        # Después de cada respuesta revisamos si la memoria creció demasiado.
        if conversacion.necesita_resumen():
            resumir_con_aviso(conversacion)
        consola.print()


if __name__ == "__main__":
    main()
