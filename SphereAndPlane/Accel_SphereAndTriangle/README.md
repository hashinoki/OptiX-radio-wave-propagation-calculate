## SphereAndPlane.cpp
File describing the setup and startup of ray tracing with OptiX.

## SphereAndPlane.cu
This file is the calculation kernel for ray tracing.
This is where the radio propagation loss calculations are implemented.
Each of the two primitives implements its own collision detection in OptiX.

## SphereAndPlane.h
Header files for SphereAndPlane.cpp and SphereAndPlane.cu.
