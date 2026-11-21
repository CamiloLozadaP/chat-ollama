"""
Tests de nucleo.py.

No necesitan Ollama: usamos un "cliente falso" que imita sus respuestas.
Así los tests son rápidos, siempre dan el mismo resultado y pueden correr
en GitHub Actions, donde no hay GPU ni modelos.

Ejecutar:  python -m pytest
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nucleo  # noqa: E402


# ---------------------------------------------------------------------------
# Cliente falso
# ---------------------------------------------------------------------------

def pedazo(contenido="", razonamiento=None, final=False):
    """Imita un 'chunk' de la respuesta en streaming de Ollama."""
    datos = {"message": SimpleNamespace(content=contenido, thinking=razonamiento)}
    if final:
        datos.update(eval_count=10, eval_duration=500_000_000, prompt_eval_count=40)
    return SimpleNamespace(**datos)


class ClienteFalso:
    """
    Devuelve respuestas predefinidas y guarda cada llamada para poder revisarla.
    - respuestas: lista de textos para las llamadas en streaming (el chat)
    - resumen: texto que devuelve cuando se le pide un resumen (stream=False)
    """

    def __init__(self, respuestas=None, resumen="- resumen de prueba", razonamiento=None, falla=False):
        self.respuestas = list(respuestas or ["respuesta"])
        self.resumen = resumen
        self.razonamiento = razonamiento
        self.falla = falla
        self.llamadas = []

    def chat(self, model, messages, stream, think):
        self.llamadas.append({"model": model, "messages": messages, "stream": stream, "think": think})
        if self.falla:
            raise ConnectionError("Ollama no responde")
        if not stream:
            return SimpleNamespace(message=SimpleNamespace(content=self.resumen))
        texto = self.respuestas.pop(0) if self.respuestas else "respuesta"
        return self._flujo(texto)

    def _flujo(self, texto):
        if self.razonamiento:
            yield pedazo(razonamiento=self.razonamiento)
        mitad = len(texto) // 2
        yield pedazo(texto[:mitad])
        yield pedazo(texto[mitad:], final=True)


def nueva(cliente=None, **opciones):
    return nucleo.Conversacion(prompt_sistema="Eres de prueba.", cliente=cliente or ClienteFalso(), **opciones)


# ---------------------------------------------------------------------------
# Conversación básica
# ---------------------------------------------------------------------------

def test_enviar_guarda_pregunta_y_respuesta():
    conv = nueva(ClienteFalso(["Hola, ¿en qué te ayudo?"]))
    texto = "".join(f.texto for f in conv.enviar("Hola"))

    assert texto == "Hola, ¿en qué te ayudo?"
    assert conv.mensajes == [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, ¿en qué te ayudo?"},
    ]


def test_el_historial_completo_se_envia_cada_vez():
    """La lección clave: el modelo no recuerda; le reenviamos todo."""
    cliente = ClienteFalso(["uno", "dos"])
    conv = nueva(cliente)
    list(conv.enviar("primera"))
    list(conv.enviar("segunda"))

    enviados = cliente.llamadas[-1]["messages"]
    assert [m["content"] for m in enviados] == ["Eres de prueba.", "primera", "uno", "segunda"]


def test_estadisticas():
    conv = nueva()
    list(conv.enviar("Hola"))
    est = conv.ultimas_estadisticas
    assert est.tokens_generados == 10
    assert est.tokens_por_segundo == pytest.approx(20.0)
    assert est.tokens_contexto == 50


def test_error_quita_la_pregunta_sin_respuesta():
    conv = nueva(ClienteFalso(falla=True))
    with pytest.raises(ConnectionError):
        list(conv.enviar("Hola"))
    assert conv.mensajes == []


def test_razonamiento_se_muestra_pero_no_se_guarda():
    conv = nueva(ClienteFalso(["391"], razonamiento="17*23 = 340 + 51"))
    fragmentos = list(conv.enviar("¿17*23?", pensar=True))

    assert fragmentos[0] == nucleo.Fragmento("razonamiento", "17*23 = 340 + 51")
    assert conv.mensajes[-1] == {"role": "assistant", "content": "391"}


def test_razonamiento_filtrado_en_la_respuesta_se_recorta():
    conv = nueva(ClienteFalso(["pensando en voz alta...</think>\n\nverde"]))
    list(conv.enviar("¿Color?"))
    assert conv.mensajes[-1]["content"] == "verde"


# ---------------------------------------------------------------------------
# Memoria resumida
# ---------------------------------------------------------------------------

def test_no_resume_si_la_conversacion_es_corta():
    conv = nueva(limite_tokens=3000)
    list(conv.enviar("Hola"))
    assert not conv.necesita_resumen()


def test_resume_al_pasar_el_limite_y_conserva_los_recientes():
    cliente = ClienteFalso(["r1", "r2", "r3"], resumen="- El usuario se llama Ana")
    conv = nueva(cliente, limite_tokens=10, mensajes_recientes=2)
    for pregunta in ["p1", "p2", "p3"]:
        list(conv.enviar(pregunta * 20))

    assert conv.necesita_resumen()
    conv.resumir()

    assert conv.resumen == "- El usuario se llama Ana"
    assert conv.veces_resumida == 1
    assert [m["content"] for m in conv.mensajes] == ["p3" * 20, "r3"]
    # La llamada de resumen no usa streaming y recibe los mensajes viejos.
    llamada = cliente.llamadas[-1]
    assert llamada["stream"] is False
    assert "p1" in llamada["messages"][1]["content"]
    assert "p3" not in llamada["messages"][1]["content"]


def test_el_resumen_se_envia_al_modelo_como_memoria():
    conv = nueva()
    conv.resumen = "- El usuario se llama Ana"
    enviados = conv.mensajes_para_modelo()

    assert enviados[0]["content"] == "Eres de prueba."
    assert enviados[1]["role"] == "system"
    assert "MEMORIA" in enviados[1]["content"]
    assert "Ana" in enviados[1]["content"]


def test_resumir_incluye_el_resumen_anterior():
    cliente = ClienteFalso(resumen="- nuevo")
    conv = nueva(cliente, mensajes_recientes=0)
    conv.resumen = "- viejo"
    conv.mensajes = [{"role": "user", "content": "hola"}]
    conv.resumir()
    assert "- viejo" in cliente.llamadas[-1]["messages"][1]["content"]
    assert conv.mensajes == []


def test_limpiar_borra_todo():
    conv = nueva()
    list(conv.enviar("Hola"))
    conv.resumen = "algo"
    conv.limpiar()
    assert conv.mensajes == [] and conv.resumen == "" and conv.ultimas_estadisticas is None


# ---------------------------------------------------------------------------
# Personalidades
# ---------------------------------------------------------------------------

def test_personalidades_incluidas():
    nombres = nucleo.listar_personalidades()
    assert {"asistente", "profesor", "programador", "traductor"} <= set(nombres)
    for nombre in nombres:
        assert nucleo.cargar_personalidad(nombre)


def test_personalidad_inexistente(tmp_path):
    with pytest.raises(ValueError):
        nucleo.cargar_personalidad("no-existe", carpeta=tmp_path)


def test_cambiar_personalidad_mantiene_la_conversacion():
    conv = nucleo.Conversacion(personalidad="asistente", cliente=ClienteFalso())
    list(conv.enviar("Hola"))
    conv.cambiar_personalidad("traductor")
    assert conv.personalidad == "traductor"
    assert "traductor" in conv.prompt_sistema.lower()
    assert len(conv.mensajes) == 2


# ---------------------------------------------------------------------------
# Guardar y cargar
# ---------------------------------------------------------------------------

def test_guardar_y_cargar(tmp_path):
    conv = nueva(ClienteFalso(["respuesta"]))
    list(conv.enviar("Hola"))
    conv.resumen = "- dato importante"
    ruta = conv.guardar(carpeta=tmp_path)

    assert nucleo.listar_conversaciones(tmp_path) == [ruta]
    cargada = nucleo.Conversacion.cargar(ruta, cliente=ClienteFalso())
    assert cargada.mensajes == conv.mensajes
    assert cargada.resumen == "- dato importante"
    assert cargada.prompt_sistema == "Eres de prueba."


def test_cargar_formato_antiguo(tmp_path):
    """La primera versión guardaba el mensaje de sistema dentro de la lista."""
    ruta = tmp_path / "vieja.json"
    ruta.write_text(
        '{"modelo": "qwen3:4b", "mensajes": [{"role": "system", "content": "s"},'
        ' {"role": "user", "content": "hola"}]}',
        encoding="utf-8",
    )
    cargada = nucleo.Conversacion.cargar(ruta, cliente=ClienteFalso())
    assert cargada.mensajes == [{"role": "user", "content": "hola"}]


def test_listar_conversaciones_vacia(tmp_path):
    assert nucleo.listar_conversaciones(tmp_path / "no-existe") == []


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("texto, esperado", [("", 1), ("abcd", 2), ("a" * 400, 101)])
def test_estimar_tokens(texto, esperado):
    assert nucleo.estimar_tokens(texto) == esperado
