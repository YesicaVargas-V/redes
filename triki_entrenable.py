import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import csv
import os

# Función para imprimir el tablero
def imprimir_tablero(tablero):
    simbolos = {1: 'X', -1: 'O', 0: ' '}
    for i in range(3):
        fila = [simbolos[tablero[j]] for j in range(i * 3, (i + 1) * 3)]
        print(" | ".join(fila))
        if i < 2:
            print("-" * 9)

# Función para verificar si hay un ganador o empate
def verificar_ganador(tablero):
    combinaciones = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Filas
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columnas
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]
    for combo in combinaciones:
        valores = [tablero[i] for i in combo]
        if valores == [1, 1, 1]:
            return 1  # Ganaste
        elif valores == [-1, -1, -1]:
            return -1  # La IA ganó
    if 0 not in tablero:
        return 0  # Empate
    return None  # El juego continúa

# Guardar jugadas en un archivo CSV
def guardar_jugada(tablero, movimiento, jugador):
    with open("jugadas_reales.csv", mode="a", newline="") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(tablero.tolist() + [movimiento, jugador])

# Cargar las jugadas reales desde un archivo CSV
def cargar_jugadas():
    if not os.path.exists("jugadas_reales.csv"):
        print("No hay jugadas reales guardadas.")
        return None, None
    datos = np.genfromtxt("jugadas_reales.csv", delimiter=",")
    X = datos[:, :9]
    y_movimientos = datos[:, 9].astype(int)
    y = np.zeros((len(y_movimientos), 9))
    for i, movimiento in enumerate(y_movimientos):
        y[i, movimiento] = 1
    return X, y

# Crear el modelo
def crear_modelo():
    modelo = Sequential([
        Dense(128, input_dim=9, activation='relu'),
        Dense(64, activation='relu'),
        Dense(9, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo

# Entrenar el modelo con jugadas reales
def entrenar_modelo():
    X, y = cargar_jugadas()
    if X is None or y is None:
        print("No hay datos suficientes para entrenar.")
        return
    modelo = crear_modelo()
    modelo.fit(X, y, epochs=20, batch_size=32)
    modelo.save("triki_modelo_entrenado_con_reales.h5")
    print("Modelo entrenado y guardado como 'triki_modelo_entrenado_con_reales.h5'.")

# Movimiento de la IA
def movimiento_ia(modelo, tablero):
    prediccion = modelo.predict(tablero.reshape(1, 9))
    movimientos_validos = np.where(tablero == 0)[0]
    prediccion_valida = [(i, prediccion[0][i]) for i in movimientos_validos]
    mejor_movimiento = max(prediccion_valida, key=lambda x: x[1])[0]
    return mejor_movimiento

# Jugar contra la IA
def jugar():
    # Cargar modelo (usa el modelo entrenado si existe)
    if os.path.exists("triki_modelo_entrenado_con_reales.h5"):
        modelo = load_model("triki_modelo_entrenado_con_reales.h5")
        print("Cargando modelo entrenado con jugadas reales.")
    else:
        modelo = crear_modelo()
        print("Modelo inicial cargado.")

    print("¡Bienvenido a Triki contra la IA!")
    print("Tú eres 'X' y la IA es 'O'.")
    print("Las posiciones del tablero están numeradas así:")
    print("0 | 1 | 2\n---------\n3 | 4 | 5\n---------\n6 | 7 | 8\n")
    
    tablero = np.zeros(9, dtype=int)  # Tablero vacío
    turno_jugador = True

    while True:
        imprimir_tablero(tablero)
        ganador = verificar_ganador(tablero)
        if ganador is not None:
            if ganador == 1:
                print("¡Felicidades, ganaste!")
            elif ganador == -1:
                print("La IA ganó. ¡Suerte la próxima vez!")
            else:
                print("¡Es un empate!")
            break

        if turno_jugador:
            try:
                movimiento = int(input("Elige tu movimiento (0-8): "))
                if tablero[movimiento] == 0:
                    tablero[movimiento] = 1  # Marca tu movimiento con '1'
                    guardar_jugada(tablero, movimiento, 1)  # Guardar jugada del jugador
                    turno_jugador = False
                else:
                    print("Movimiento inválido. La casilla ya está ocupada.")
            except (ValueError, IndexError):
                print("Por favor, elige un número válido entre 0 y 8.")
        else:
            print("La IA está pensando...")
            movimiento = movimiento_ia(modelo, tablero)
            tablero[movimiento] = -1  # La IA marca con '-1'
            guardar_jugada(tablero, movimiento, -1)  # Guardar jugada de la IA
            turno_jugador = True

# Menú principal
if __name__ == "__main__":
    while True:
        print("\n1. Jugar contra la IA")
        print("2. Entrenar modelo con jugadas reales")
        print("3. Salir")
        opcion = input("Selecciona una opción: ")
        if opcion == "1":
            jugar()
        elif opcion == "2":
            entrenar_modelo()
        elif opcion == "3":
            print("¡Adiós!")
            break
        else:
            print("Opción no válida. Intenta de nuevo.")
