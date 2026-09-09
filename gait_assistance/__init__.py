"""Real-time hemiparetic gait assistance on the Log-Euclidean SPD manifold.

The package is organised as two decoupled control loops:

* ``LowLevelLoop``  (100-1000 Hz) - acquisition, safety, gait phase,
  belt-target interpolation and impedance/current control.
* ``HighLevelLoop`` (0.5-2 Hz, stride triggered) - features, SPD covariance,
  matrix logarithm, gait-state clustering, deviation and assist gain.

No supervised machine learning is used anywhere: the patient model is built
online from the first strides of the session.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
