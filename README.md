# PETSc Demos

This is a place to adapt or create PETSc examples.

Follow these steps to create and build programs, assuming you are in the cloned repository

* For new programs
  * Create source code in `src`
  * Add a target to `src/makefile` with an executable of the same name
* For all programs
  * Make sure `PETSC_DIR` and `PETSC_ARCH` are set
  * Run `./setup TARGET`, where `TARGET` is the `make` target
  * Move to `programs/TARGET`
  * Run `./TARGET` or `mpirun -n N TARGET`

