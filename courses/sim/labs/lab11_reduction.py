"""Lab 11 (handbook 1.5.7): data reduction by subsampling. How
much error does each level of reduction cost? Plain NumPy; the
simulator is not started.

Run (env isaac):  python labs/lab11_reduction.py
"""
import vizcore as vc

print("stride  points   quads   max error   rms error")
for stride, points, quads, worst, rms, _Xs, _Zs in vc.reduction_table():
    print(f"{stride:5d}  {points:6d}  {quads:6d}"
          f"   {worst:.4f} m   {rms:.4f} m")
