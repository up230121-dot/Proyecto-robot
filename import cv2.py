import subprocess, sys

def instalar(paquete):
    subprocess.check_call([sys.executable, "-m", "pip", "install", paquete])

try:
    import cv2
except ModuleNotFoundError:
    instalar("opencv-python")
    import cv2

try:
    import numpy as np
except ModuleNotFoundError:
    instalar("numpy")
    import numpy as np
import cv2
import numpy as np
 
# ---------- CONFIGURACIÓN ----------
CAMARA = 0              # 0 = cámara por defecto; prueba 1, 2... si tienes varias
AREA_MINIMA = 500       # ignora manchas pequeñas (ruido)
USAR_SERIAL = False     # pon True si vas a enviar a un Arduino
PUERTO = "COM3"         # Windows: COM3, Linux: /dev/ttyUSB0, Mac: /dev/cu.usbmodem...
BAUDIOS = 9600
 
# Rangos HSV (H: 0-179, S: 0-255, V: 0-255). Ajústalos con tu iluminación.
COLORES = {
    "rojo": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (179, 255, 255))],
    "verde": [((40, 80, 70), (80, 255, 255))],
    "azul": [((100, 120, 70), (130, 255, 255))],
}
 
# Color de dibujo (BGR) para cada color detectado
DIBUJO = {"rojo": (0, 0, 255), "verde": (0, 255, 0), "azul": (255, 0, 0)}
 
 
def crear_mascara(hsv, rangos):
    mascara = None
    for bajo, alto in rangos:
        m = cv2.inRange(hsv, np.array(bajo), np.array(alto))
        mascara = m if mascara is None else cv2.bitwise_or(mascara, m)
    # Limpia ruido
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return mascara
 
 
def encontrar_centros(mascara):
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centros = []
    for c in contornos:
        if cv2.contourArea(c) < AREA_MINIMA:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centros.append((cx, cy))
    return centros
 
 
def main():
    serial_port = None
    if USAR_SERIAL:
        import serial
        serial_port = serial.Serial(PUERTO, BAUDIOS, timeout=1)
 
    cap = cv2.VideoCapture(CAMARA)
    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return
 
    print("Presiona 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
 
        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7, 7), 0), cv2.COLOR_BGR2HSV)
 
        for nombre, rangos in COLORES.items():
            mascara = crear_mascara(hsv, rangos)
            for (x, y) in encontrar_centros(mascara):
                # Dibuja en pantalla
                cv2.circle(frame, (x, y), 8, DIBUJO[nombre], -1)
                cv2.putText(frame, f"{nombre} ({x},{y})", (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, DIBUJO[nombre], 2)
 
                # Mensaje con las coordenadas: color,x,y
                mensaje = f"{nombre},{x},{y}\n"
                print(mensaje.strip())
                if serial_port:
                    serial_port.write(mensaje.encode())
 
        cv2.imshow("Detector de colores", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
 
    cap.release()
    cv2.destroyAllWindows()
    if serial_port:
        serial_port.close()
 
 
if __name__ == "__main__":
    main()
 