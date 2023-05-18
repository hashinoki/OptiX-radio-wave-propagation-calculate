# calc.cu
This file is the calculation kernel for ray tracing.  
This is where the radio propagation loss calculations are implemented.  
The collision detection for triangles uses a ray tracing framework by OptiX. On the other hand, collisions with spheres are implemented directly in the Closest-Hit and Miss processes, respectively.

# main.cpp
