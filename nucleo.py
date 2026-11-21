"""
Lógica del chat, separada de la interfaz.

La usan tanto el chat de la terminal (chat.py) como la interfaz web (app_web.py).
Tenerla aparte permite probarla con tests sin necesitar Ollama ni una pantalla.

Ideas clave:
  - El modelo NO tiene memoria: cada vez le enviamos el prompt de sistema
    y todos los mensajes de la conversación.
  - Por eso el contexto crece con cada mensaje. Si supera el límite del modelo,
    Ollama recorta lo más antiguo SIN AVISAR, y el modelo "olvida" cosas.
  - Solución: "memoria resumida". Cuando la conversación crece demasiado,
    le pedimos al propio modelo que resuma los mensajes viejos, y guardamos
    solo el resumen y los mensajes más recientes. Así gastamos muchos menos tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import ollama

MODELO_POR_DEFECTO = "qwen3:4b-instruct"
PERSONALIDAD_POR_DEFECTO = "asistente"

RAIZ = Path(__file__).parent
CARPETA_PERSONALIDADES = RAIZ / "personalidades"
CARPETA_CONVERSACIONES = RAIZ / "conversaciones"

# Cuando la conversación supere este número de tokens (aprox.), se resume.
# Ollama usa por defecto 4096 tokens de contexto en tu GPU, así que resumimos antes.
LIMITE_TOKENS = 3000

# Cuántos mensajes recientes se conservan completos al resumir.
MENSAJES_RECIENTES = 4

PROMPT_RESUMEN = (
    "Resume la siguiente conversación entre un usuario y un asistente. "
    "Conserva todos los datos importantes: nombre y preferencias del usuario, "
    "decisiones tomadas, preguntas pendientes y cualquier dato concreto (números, nombres, fechas). "
    "Escribe el resumen en español, en viñetas breves, sin introducciones."
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def estimar_tokens(texto: str) -> int:
    """
    Estimación rápida: en promedio, 1 token ≈ 4 caracteres.
    No es exacta, pero sirve para decidir cuándo resumir sin llamar al modelo.
    """
    return len(texto) // 4 + 1


def quitar_razonamiento(texto: str) -> str:
    """Algunos modelos 'filtran' su razonamiento en la respuesta, terminado en </think>."""
    if "</think>" in texto:
        return texto.split("</think>", 1)[1].strip()
    return texto


def listar_personalidades(carpeta: Path = CARPETA_PERSONALIDADES) -> list[str]:
    """Cada archivo .md de la carpeta 'personalidades' es una personalidad."""
    return sorted(ruta.stem for ruta in carpeta.glob("*.md"))


def cargar_personalidad(nombre: str, carpeta: Path = CARPETA_PERSONALIDADES) -> str:
    """Devuelve el prompt de sistema guardado en personalidades/<nombre>.md."""
    ruta = carpeta / f"{nombre}.md"
    if not ruta.exists():
        disponibles = ", ".join(listar_personalidades(carpeta)) or "(ninguna)"
        raise ValueError(f"No existe la personalidad '{nombre}'. Disponibles: {disponibles}")
    return ruta.read_text(encoding="utf-8").strip()


def listar_conversaciones(carpeta: Path = CARPETA_CONVERSACIONES) -> list[Path]:
    """Conversaciones guardadas, de la más reciente a la más antigua."""
    if not carpeta.exists():
        return []
    return sorted(carpeta.glob("*.json"), reverse=True)


def listar_modelos(cliente=None) -> list[str]:
    """Modelos descargados en Ollama. Lanza una excepción si Ollama no está corriendo."""
    cliente = cliente or ollama.Client()
    return [m.model for m in cliente.list().models]


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

@dataclass
class Fragmento:
    """Un pedazo de respuesta en streaming: 'razonamiento' o 'respuesta'."""
    tipo: str
    texto: str


@dataclass
class Estadisticas:
    """Métricas que Ollama entrega al terminar cada respuesta."""
    tokens_generados: int
    tokens_por_segundo: float
    tokens_contexto: int


# ---------------------------------------------------------------------------
# La conversación
# ---------------------------------------------------------------------------

class Conversacion:
    def __init__(
        self,
        modelo: str = MODELO_POR_DEFECTO,
        personalidad: str = PERSONALIDAD_POR_DEFECTO,
        prompt_sistema: str | None = None,
        limite_tokens: int = LIMITE_TOKENS,
        mensajes_recientes: int = MENSAJES_RECIENTES,
        cliente=None,
    ):
        self.modelo = modelo
        self.personalidad = personalidad
        self.prompt_sistema = prompt_sistema if prompt_sistema is not None else cargar_personalidad(personalidad)
        self.limite_tokens = limite_tokens
        self.mensajes_recientes = mensajes_recientes
        # Permite inyectar un cliente falso en los tests.
        self.cliente = cliente or ollama.Client()

        self.resumen = ""                 # memoria resumida de los mensajes viejos
        self.mensajes: list[dict] = []    # mensajes recientes (usuario y asistente)
        self.ultimas_estadisticas: Estadisticas | None = None
        self.veces_resumida = 0

    # --- Lo que se le envía al modelo -------------------------------------

    def mensajes_para_modelo(self) -> list[dict]:
        """
        Prompt de sistema + resumen (si hay) + mensajes recientes.

        El resumen va en un mensaje de sistema propio y con una instrucción explícita:
        los modelos pequeños tienden a ignorar información "escondida" al final
        del prompt principal y responden que no recuerdan nada.
        """
        mensajes = [{"role": "system", "content": self.prompt_sistema}]
        if self.resumen:
            mensajes.append({
                "role": "system",
                "content": (
                    "MEMORIA: esto es lo que recuerdas de la conversación anterior con este usuario. "
                    "Úsalo para responder como si lo recordaras tú mismo; nunca digas que no lo sabes "
                    "si la respuesta está aquí.\n" + self.resumen
                ),
            })
        return mensajes + self.mensajes

    def tokens_estimados(self) -> int:
        return sum(estimar_tokens(m["content"]) for m in self.mensajes_para_modelo())

    # --- Conversar ---------------------------------------------------------

    def enviar(self, texto: str, pensar: bool = False) -> Iterator[Fragmento]:
        """
        Envía un mensaje y va devolviendo la respuesta en pedazos (streaming).
        Al terminar, guarda la respuesta en el historial.
        Si ocurre un error, quita la pregunta para no dejar el historial a medias.
        """
        self.mensajes.append({"role": "user", "content": texto})
        respuesta = ""
        ultimo = None
        try:
            flujo = self.cliente.chat(
                model=self.modelo,
                messages=self.mensajes_para_modelo(),
                stream=True,
                think=pensar,
            )
            for pedazo in flujo:
                mensaje = pedazo.message
                if pensar and getattr(mensaje, "thinking", None):
                    yield Fragmento("razonamiento", mensaje.thinking)
                if mensaje.content:
                    respuesta += mensaje.content
                    yield Fragmento("respuesta", mensaje.content)
                ultimo = pedazo
        except BaseException:
            # BaseException incluye Ctrl+C (KeyboardInterrupt).
            self.mensajes.pop()
            raise

        self.mensajes.append({"role": "assistant", "content": quitar_razonamiento(respuesta)})

        if ultimo is not None and getattr(ultimo, "eval_count", None) and getattr(ultimo, "eval_duration", None):
            self.ultimas_estadisticas = Estadisticas(
                tokens_generados=ultimo.eval_count,
                tokens_por_segundo=ultimo.eval_count / (ultimo.eval_duration / 1e9),
                tokens_contexto=(ultimo.prompt_eval_count or 0) + ultimo.eval_count,
            )

    # --- Memoria resumida --------------------------------------------------

    def necesita_resumen(self) -> bool:
        """Hay que resumir si se pasó el límite y hay mensajes viejos que resumir."""
        return (
            self.tokens_estimados() > self.limite_tokens
            and len(self.mensajes) > self.mensajes_recientes
        )

    def resumir(self) -> str:
        """
        Resume los mensajes viejos (junto con el resumen anterior, si existe)
        y conserva completos solo los más recientes.
        """
        if len(self.mensajes) <= self.mensajes_recientes:
            return self.resumen

        if self.mensajes_recientes > 0:
            viejos = self.mensajes[: -self.mensajes_recientes]
            recientes = self.mensajes[-self.mensajes_recientes:]
        else:
            viejos, recientes = self.mensajes, []

        texto = ""
        if self.resumen:
            texto += "Resumen previo:\n" + self.resumen + "\n\n"
        texto += "\n".join(
            f"{'Usuario' if m['role'] == 'user' else 'Asistente'}: {m['content']}" for m in viejos
        )

        resultado = self.cliente.chat(
            model=self.modelo,
            messages=[
                {"role": "system", "content": PROMPT_RESUMEN},
                {"role": "user", "content": texto},
            ],
            stream=False,
            think=False,
        )
        self.resumen = quitar_razonamiento(resultado.message.content).strip()
        self.mensajes = recientes
        self.veces_resumida += 1
        return self.resumen

    # --- Administración ----------------------------------------------------

    def limpiar(self) -> None:
        self.resumen = ""
        self.mensajes = []
        self.ultimas_estadisticas = None
        self.veces_resumida = 0

    def cambiar_personalidad(self, nombre: str) -> None:
        """Cambia el prompt de sistema; la conversación se mantiene."""
        self.prompt_sistema = cargar_personalidad(nombre)
        self.personalidad = nombre

    def guardar(self, carpeta: Path = CARPETA_CONVERSACIONES) -> Path:
        carpeta.mkdir(parents=True, exist_ok=True)
        ruta = carpeta / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".json")
        datos = {
            "modelo": self.modelo,
            "personalidad": self.personalidad,
            "prompt_sistema": self.prompt_sistema,
            "resumen": self.resumen,
            "mensajes": self.mensajes,
        }
        ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
        return ruta

    @classmethod
    def cargar(cls, ruta: Path, cliente=None, **opciones) -> "Conversacion":
        datos = json.loads(Path(ruta).read_text(encoding="utf-8"))
        conversacion = cls(
            modelo=datos.get("modelo", MODELO_POR_DEFECTO),
            personalidad=datos.get("personalidad", PERSONALIDAD_POR_DEFECTO),
            prompt_sistema=datos.get("prompt_sistema", ""),
            cliente=cliente,
            **opciones,
        )
        conversacion.resumen = datos.get("resumen", "")
        # Compatibilidad con la versión anterior, que guardaba el mensaje de sistema en la lista.
        conversacion.mensajes = [m for m in datos.get("mensajes", []) if m.get("role") != "system"]
        return conversacion
