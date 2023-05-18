# main.cpp
File describing the setup and startup of ray tracing with OptiX.

# calc.cu
This file is the calculation kernel for ray tracing.  
This is where the radio propagation loss calculations are implemented.  
The collision detection for triangles uses a ray tracing framework by OptiX. On the other hand, collisions with spheres are implemented directly in the Closest-Hit and Miss processes, respectively.

# rtxFunctions.cpp
This file defines various functions using OptiX API to simplify the configuration in main.cpp.
