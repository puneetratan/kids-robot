import RPi.GPIO as GPIO
import time

# Pin definitions
IN1 = 17
IN2 = 18
IN3 = 27
IN4 = 24

GPIO.setmode(GPIO.BCM)

# Set all as outputs, start LOW
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)

print("=" * 50)
print("DIAGNOSTIC TEST - Motor Control")
print("=" * 50)

# Check current pin states (should all be LOW)
print("\n[1] Current pin states (should all be 0):")
print(f"GPIO 17 (IN1): {GPIO.input(IN1)}")
print(f"GPIO 18 (IN2): {GPIO.input(IN2)}")
print(f"GPIO 27 (IN3): {GPIO.input(IN3)}")
print(f"GPIO 24 (IN4): {GPIO.input(IN4)}")

print("\n[2] Testing Motor A (IN1/IN2) - Forward for 3 seconds...")
GPIO.output(IN1, GPIO.HIGH)
GPIO.output(IN2, GPIO.LOW)
print(f"  IN1=HIGH, IN2=LOW (GPIO 17: {GPIO.input(IN1)}, GPIO 18: {GPIO.input(IN2)})")
time.sleep(3)

print("  Stopping Motor A...")
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
time.sleep(1)

print("\n[3] Testing Motor A - Backward for 3 seconds...")
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.HIGH)
print(f"  IN1=LOW, IN2=HIGH (GPIO 17: {GPIO.input(IN1)}, GPIO 18: {GPIO.input(IN2)})")
time.sleep(3)

print("  Stopping Motor A...")
GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
time.sleep(1)

print("\n[4] Testing Motor B (IN3/IN4) - Forward for 3 seconds...")
GPIO.output(IN3, GPIO.HIGH)
GPIO.output(IN4, GPIO.LOW)
print(f"  IN3=HIGH, IN4=LOW (GPIO 27: {GPIO.input(IN3)}, GPIO 24: {GPIO.input(IN4)})")
time.sleep(3)

print("  Stopping Motor B...")
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)
time.sleep(1)

print("\n[5] Testing Motor B - Backward for 3 seconds...")
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.HIGH)
print(f"  IN3=LOW, IN4=HIGH (GPIO 27: {GPIO.input(IN3)}, GPIO 24: {GPIO.input(IN4)})")
time.sleep(3)

print("  Stopping Motor B...")
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)

print("\n[6] Final pin states (should all be 0):")
print(f"GPIO 17 (IN1): {GPIO.input(IN1)}")
print(f"GPIO 18 (IN2): {GPIO.input(IN2)}")
print(f"GPIO 27 (IN3): {GPIO.input(IN3)}")
print(f"GPIO 24 (IN4): {GPIO.input(IN4)}")

print("\n" + "=" * 50)
print("DIAGNOSTIC COMPLETE")
print("=" * 50)

GPIO.cleanup()
