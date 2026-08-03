"""
motor_test.py
===============
L298N two-motor test for the kids-robot chassis.

WIRING (Raspberry Pi 5 -> L298N)
--------------------------------
    L298N     Pi physical pin    GPIO (BCM)
    ENA       29                 5
    IN1       11                 17
    IN2       13                 27
    IN3       15                 22
    IN4       16                 23
    ENB       31                 6
    GND       25                 -

IMPORTANT - COUNTING PHYSICAL PINS
----------------------------------
Pin 1 is at the END of the header where pins 2 and 4 both
read 5V. Verify with a multimeter before trusting any
layout diagram:

    1. Find two adjacent pins in the same column reading 5V
       -> those are pins 2 and 4
    2. The next pin down in that column is pin 6 = GROUND
    3. Sanity check: pin 6 to pin 9 must read 0.00V
       (ground to ground). If it does not, you are counting
       from the wrong end.

Counting from the wrong end mirrors every pin number and
silently breaks everything. This cost three nights.

BOARD SETUP
-----------
    - 5V-EN jumper: ON (board regulator powers its logic;
      red LED should be lit)
    - ENA / ENB: no caps, no loops - driven from the Pi
    - Battery pack -> +12V terminal (6xAA, ~8-9V)
    - Motor A -> OUT1/OUT2, Motor B -> OUT3/OUT4

Run:
    python3 motor_test.py
"""

from gpiozero import OutputDevice
import time

ENA = OutputDevice(5)
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)
ENB = OutputDevice(6)

ALL = (ENA, IN1, IN2, IN3, IN4, ENB)


def stop():
    """Everything off. Safe resting state."""
    for pin in ALL:
        pin.off()


def motor_a(forward=True, seconds=2):
    ENA.on()
    (IN1 if forward else IN2).on()
    time.sleep(seconds)
    IN1.off()
    IN2.off()
    ENA.off()


def motor_b(forward=True, seconds=2):
    ENB.on()
    (IN3 if forward else IN4).on()
    time.sleep(seconds)
    IN3.off()
    IN4.off()
    ENB.off()


def both(forward=True, seconds=2):
    ENA.on()
    ENB.on()
    (IN1 if forward else IN2).on()
    (IN3 if forward else IN4).on()
    time.sleep(seconds)
    stop()


if __name__ == "__main__":
    try:
        stop()
        print("idle 2s - nothing should move")
        time.sleep(2)

        print("Motor A forward");  motor_a(forward=True);  time.sleep(1)
        print("Motor A reverse");  motor_a(forward=False); time.sleep(1)
        print("Motor B forward");  motor_b(forward=True);  time.sleep(1)
        print("Motor B reverse");  motor_b(forward=False); time.sleep(1)
        print("Both forward");     both(forward=True);     time.sleep(1)
        print("Both reverse");     both(forward=False)

        print("done")
    finally:
        stop()
