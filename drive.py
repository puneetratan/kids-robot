"""
drive.py
==========
Keyboard driving for the kids-robot chassis.

Non-blocking: a key press sets the motor pins and returns
immediately, so the robot keeps moving while you type the
next command. Nothing here calls time.sleep() on the main
path - that matters later, when the voice pipeline needs
to answer questions WHILE the robot is driving.

CONTROLS
--------
    a   forward
    b   backward
    c   stop
    l   turn left
    r   turn right
    q   quit (stops motors first)

WIRING - see motor_test.py for the full pin map and the
note about counting physical pins from the correct end.

Run:
    python3 drive.py
"""

from gpiozero import OutputDevice
import sys
import termios
import tty

ENA = OutputDevice(5)
IN1 = OutputDevice(17)
IN2 = OutputDevice(27)
IN3 = OutputDevice(22)
IN4 = OutputDevice(23)
ENB = OutputDevice(6)

ALL = (ENA, IN1, IN2, IN3, IN4, ENB)


def stop():
    for pin in ALL:
        pin.off()


def _drive(a_fwd, b_fwd):
    """Set both channels. True = forward, False = reverse,
    None = that motor holds still."""
    IN1.off(); IN2.off(); IN3.off(); IN4.off()

    if a_fwd is None:
        ENA.off()
    else:
        ENA.on()
        (IN1 if a_fwd else IN2).on()

    if b_fwd is None:
        ENB.off()
    else:
        ENB.on()
        (IN3 if b_fwd else IN4).on()


def forward():
    _drive(True, True)


def backward():
    _drive(False, False)


def turn_left():
    """Right wheel forward, left wheel back - spins in place.
    If it turns the wrong way, swap these two arguments."""
    _drive(False, True)


def turn_right():
    _drive(True, False)


def _read_key():
    """One keypress, no Enter needed."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


COMMANDS = {
    "a": ("forward",  forward),
    "b": ("backward", backward),
    "c": ("stop",     stop),
    "l": ("left",     turn_left),
    "r": ("right",    turn_right),
}


def main():
    print("a forward  b backward  c stop  l left  r right  q quit")
    stop()
    try:
        while True:
            key = _read_key().lower()
            if key == "q":
                break
            if key in COMMANDS:
                label, action = COMMANDS[key]
                action()
                print(f"\r{label:<10}", end="", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop()
        print("\rstopped    ")


if __name__ == "__main__":
    main()
