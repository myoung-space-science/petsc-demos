This is a place to adapt or create PETSc examples.

Steps (starting from `~/sandbox/petsc-demos`)

* For new programs
  * Create source code in `src`
  * Add a target to `src/makefile` with an executable of the same name
* For all programs
  * Make sure `PETSC_DIR` and `PETSC_ARCH` are set
  * Run `./setup TARGET`, where `TARGET` is the `make` target
  * Move to `programs/TARGET`
  * Run `./TARGET` or `mpirun -n N TARGET`

