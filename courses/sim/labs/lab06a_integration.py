"""Lab 6, part A: Euler against Runge-Kutta 2 in
a field where the exact answer is a circle. Plain NumPy; the
simulator is not started.

Run (env isaac):  python labs/lab06a_integration.py
"""
import vizcore as vc

print(" dt     method  radius after one turn   error")
for dt, method, radius, _path in vc.circle_test():
    print(f" {dt:<5} {method:6s}  {radius:.5f}"
          f"                {abs(radius - 1):.5f}")
