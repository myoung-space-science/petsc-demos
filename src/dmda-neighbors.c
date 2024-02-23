static char help[] = \
"Example using DMDAGetNeighbors.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>

#define NDIM 3

int main(int argc, char **args)
{
  DM dmda;
  PetscMPIInt        rank;
  PetscInt           nx=32;
  PetscInt           ny=32;
  PetscInt           nz=32;
  DMBoundaryType     xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType     yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType     zBC=DM_BOUNDARY_PERIODIC;
  PetscInt           dof=1;
  PetscInt           width=1;
  PetscReal          x0=0.0;
  PetscReal          x1=1.0;
  PetscReal          y0=0.0;
  PetscReal          y1=1.0;
  PetscReal          z0=0.0;
  PetscReal          z1=1.0;
  const PetscMPIInt *ranks;
  PetscInt           ix, iy, iz, ir;
  PetscMPIInt        neighbor;
  const char        *ijk[NDIM*NDIM*NDIM]={"(i-1, j-1, k-1)", \
                                          "(i  , j-1, k-1)", \
                                          "(i+1, j-1, k-1)", \
                                          "(i-1, j  , k-1)", \
                                          "(i  , j  , k-1)", \
                                          "(i+1, j  , k-1)", \
                                          "(i-1, j+1, k-1)", \
                                          "(i  , j+1, k-1)", \
                                          "(i+1, j+1, k-1)", \
                                          "(i-1, j-1, k  )", \
                                          "(i  , j-1, k  )", \
                                          "(i+1, j-1, k  )", \
                                          "(i-1, j  , k  )", \
                                          "(i  , j  , k  )", \
                                          "(i+1, j  , k  )", \
                                          "(i-1, j+1, k  )", \
                                          "(i  , j+1, k  )", \
                                          "(i+1, j+1, k  )", \
                                          "(i-1, j-1, k+1)", \
                                          "(i  , j-1, k+1)", \
                                          "(i+1, j-1, k+1)", \
                                          "(i-1, j  , k+1)", \
                                          "(i  , j  , k+1)", \
                                          "(i+1, j  , k+1)", \
                                          "(i-1, j+1, k+1)", \
                                          "(i  , j+1, k+1)", \
                                          "(i+1, j+1, k+1)"};

  PetscFunctionBeginUser;

  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, &dmda));
  PetscCall(DMDASetElementType(dmda, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(dmda));
  PetscCall(DMSetUp(dmda));
  PetscCall(DMDASetUniformCoordinates(dmda, x0, x1, y0, y1, z0, z1));
  PetscCall(DMView(dmda, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMDAGetNeighbors(dmda, &ranks));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d]\n", rank));
  for (iz=0; iz<NDIM; iz++) {
    for (iy=0; iy<NDIM; iy++) {
      for (ix=0; ix<NDIM; ix++) {
        ir = iz*NDIM*NDIM + iy*NDIM + ix;
        neighbor = ranks[ir];
        PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "neighbor at %s is ranks[%02d] = %02d\n", ijk[ir], ir, neighbor));
      }
    }
  }
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "\n"));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  PetscCall(PetscFinalize());

  return 0;
}