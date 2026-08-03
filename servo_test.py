"""
servo_test.py
===============
PCA9685 + MG996R servo test for the kids-robot pan-tilt neck.

WIRING (Raspberry Pi 5 -> PCA9685)
----------------------------------
    PCA9685   Pi physical pin
    VCC       1    (3.3V - chip logic only)
    SDA       3
    SCL       5
    GND       6

    Servo power: 4xAA pack (~6V) -> green terminal block
    NOTE: V+ is 6V MAX on this board. Never connect the
    9V pack used for the L298N motors.

SERVO CONNECTIONS
-----------------
    Each channel is a 3-pin group, matched by colour:
        brown/black  -> GND  (black row)
        red          -> V+   (red row)
        orange/yellow-> PWM  (yellow row)

    Channel 0 = pan, Channel 1 = tilt (by convention here)

VERIFY BEFORE RUNNING
---------------------
    sudo i2cdetect -y 1     -> expect 40 (and 70, all-call)

    Measure V+ against GND with a multimeter: ~6V.
    A pack under ~5V will leave every LED lit while the
    servo refuses to move. Depleted batteries look exactly
    like a code problem.

MG996R LIMITS
-------------
    180 degree servo. Commanding outside 0-180 drives it
    against an internal mechanical stop - it draws hard,
    heats up, and strips gear teeth over time. Angles are
    clamped below.

Run:
    python3 servo_test.py
"""

from adafruit_servokit import ServoKit
import time

PAN = 0
TILT = 1

CENTER = 90
MIN_ANGLE = 0
MAX_ANGLE = 180

kit = ServoKit(channels=16)


def clamp(angle):
    """Never send a 180 degree servo past its stops."""
    return max(MIN_ANGLE, min(MAX_ANGLE, angle))


def move(channel, angle, settle=0.5):
    angle = clamp(angle)
    kit.servo[channel].angle = angle
    time.sleep(settle)
    return angle


def center_all():
    move(PAN, CENTER)
    move(TILT, CENTER)


def sweep(channel, start=30, end=150, step=10, settle=0.15):
    """Slow sweep - watch for binding or stalling."""
    for a in range(start, end + 1, step):
        move(channel, a, settle)
    for a in range(end, start - 1, -step):
        move(channel, a, settle)


def look(pan_angle, tilt_angle):
    """Point the head. Useful once the camera is mounted."""
    move(PAN, pan_angle)
    move(TILT, tilt_angle)


if __name__ == "__main__":
    try:
        print("centering both")
        center_all()
        time.sleep(1)

        print("pan: left, center, right")
        move(PAN, 30);  time.sleep(0.5)
        move(PAN, 90);  time.sleep(0.5)
        move(PAN, 150); time.sleep(0.5)
        move(PAN, 90);  time.sleep(1)

        print("tilt: down, center, up")
        move(TILT, 60);  time.sleep(0.5)
        move(TILT, 90);  time.sleep(0.5)
        move(TILT, 120); time.sleep(0.5)
        move(TILT, 90);  time.sleep(1)

        print("pan sweep")
        sweep(PAN)

        print("tilt sweep")
        sweep(TILT, start=60, end=120)

        print("recentering")
        center_all()
        print("done")

    except KeyboardInterrupt:
        print("\ninterrupted - recentering")
        center_all()
