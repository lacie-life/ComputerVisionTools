import serial


# this port address is for the serial tx/rx pins on the GPIO header
SERIAL_PORT = '/dev/ttyACM0'
# be sure to set this to the same rate used on the Arduino
SERIAL_RATE = 9600


def main():
    ser = serial.Serial(SERIAL_PORT, SERIAL_RATE)
    while True:
        # using ser.readline() assumes each line contains a single reading
        # sent using Serial.println() on the Arduino
        try:
                reading = ser.readline().decode('utf-8')
        except:
            print("An exception occurred")
        # reading is a string...do whatever you want from here
        print(reading + "\n")
        print("\n")


if __name__ == "__main__":
    main()
