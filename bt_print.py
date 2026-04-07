import serial

ser = serial.Serial("COM3", baudrate=9600, timeout=1)

print("Listening on COM3...")

while True:
    data = ser.readline().decode(errors="ignore").strip()
    if data:
        print(data)
