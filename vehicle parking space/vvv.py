import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure the serial port and baud rate
SERIAL_PORT = '/dev/ttyUSB0'  # Update this to your serial port
BAUD_RATE = 9600

# Initialize the serial connection
ser = serial.Serial(SERIAL_PORT, BAUD_RATE)

# Setup the plot
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], 'r-', animated=True)
plt.title('Heart Beat Monitoring System')
plt.xlabel('Time (s)')
plt.ylabel('Signal')

# Function to initialize the plot
def init():
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1024)
    return ln,

# Function to update the plot
def update(frame):
    data = ser.readline().decode('utf-8').strip()
    try:
        value = int(data)
    except ValueError:
        return ln,
    
    xdata.append(frame * 0.1)
    ydata.append(value)
    
    if len(xdata) > 100:
        ax.set_xlim(frame * 0.1 - 10, frame * 0.1)
    
    ln.set_data(xdata, ydata)
    return ln,

# Create an animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 100, 0.1), init_func=init, blit=True)

# Display the plot
plt.show()

# Close the serial connection when done
ser.close()