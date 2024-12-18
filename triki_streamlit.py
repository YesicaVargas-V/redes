import numpy as np
import pandas as pd
import streamlit as st
import os
from tensorflow.keras.models import load_model

# Inicializar variables de sesión
if "tablero" not in st.session_state:
    st.session_state.tablero = np.zeros(9, dtype=int)  # Tablero vacío
    st.session_state.turno_jugador = True  # Empieza el jugador
    st.session_state.movimiento_ia = None  # Último movimiento de la IA

# Cargar modelo entrenado
def cargar_modelo():
    if os.path.exists("triki_modelo_entrenado_con_reales.h5"):
        return load_model("triki_modelo_entrenado_con_reales.h5")
    return None

modelo = cargar_modelo()

# Guardar jugadas en archivo CSV
def guardar_jugada(tablero, movimiento, jugador):
    with open("jugadas_reales.csv", "a", newline="") as archivo:
        pd.DataFrame([list(tablero) + [movimiento, jugador]]).to_csv(archivo, header=False, index=False)

# Verificar ganador
def verificar_ganador(tablero):
    combinaciones = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for combo in combinaciones:
        valores = [tablero[i] for i in combo]
        if valores == [1, 1, 1]: return 1  # Jugador gana
        if valores == [-1, -1, -1]: return -1  # IA gana
    if 0 not in tablero: return 0  # Empate
    return None

# Movimiento de la IA
def movimiento_ia(tablero):
    movimientos_validos = np.where(tablero == 0)[0]
    if modelo:
        prediccion = modelo.predict(tablero.reshape(1, 9))
        mejor_movimiento = max(movimientos_validos, key=lambda x: prediccion[0][x])
    else:
        mejor_movimiento = np.random.choice(movimientos_validos)  # Movimiento aleatorio
    return mejor_movimiento

# Renderizar tablero
def renderizar_tablero():
    st.write("### Tu turno: Elige una casilla")
    cols = st.columns(3)
    for i in range(9):
        color = "red" if st.session_state.tablero[i] == 1 else ("green" if st.session_state.tablero[i] == -1 else "white")
        with cols[i % 3]:
            if st.session_state.tablero[i] == 0 and st.session_state.turno_jugador:
                if st.button(" ", key=i, help="Casilla libre", use_container_width=True):
                    st.session_state.tablero[i] = 1
                    guardar_jugada(st.session_state.tablero, i, 1)
                    st.session_state.turno_jugador = False
                    st.rerun()
            else:
                st.markdown(f"<div style='text-align:center; color:{color}; font-size:30px;'>{'X' if st.session_state.tablero[i] == 1 else ('O' if st.session_state.tablero[i] == -1 else ' ')}</div>", unsafe_allow_html=True)

# Lógica del juego
st.title("Triki (Tres en Raya) contra la IA")

# Mostrar tablero y actualizar juego
ganador = verificar_ganador(st.session_state.tablero)
if ganador is None:
    renderizar_tablero()
    if not st.session_state.turno_jugador:
        movimiento = movimiento_ia(st.session_state.tablero)
        st.session_state.tablero[movimiento] = -1
        guardar_jugada(st.session_state.tablero, movimiento, -1)
        st.session_state.turno_jugador = True
        st.rerun()
else:
    st.write("### ¡Juego terminado!")
    if ganador == 1:
        st.success("🎉 ¡Felicidades, ganaste!")
    elif ganador == -1:
        st.error("🤖 La IA ganó. ¡Suerte la próxima vez!")
    else:
        st.warning("😐 ¡Es un empate!")
    if st.button("Reiniciar juego"):
        st.session_state.tablero = np.zeros(9, dtype=int)
        st.session_state.turno_jugador = True
        st.rerun()
