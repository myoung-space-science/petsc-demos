static char help[] = "Examine DMSWARM particle initialzation methods.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


#define NDIM 3

typedef struct {
  PetscInt nx, ny, nz;
  PetscReal Lx, Ly, Lz;
} Grid;

typedef struct {
  PetscInt n;
} Particles;

typedef struct {
  Grid grid;
  Particles particles;
} UserContext;


static PetscErrorCode
ProcessOptions(UserContext *options)
{
  PetscFunctionBeginUser;

  PetscInt intArg;
  PetscReal realArg;
  PetscBool found;
  PetscInt  np;

  options->grid.nx = 7;
  options->grid.ny = 7;
  options->grid.nz = 7;
  np = options->grid.nx * options->grid.ny * options->grid.nz;
  options->particles.n = np;
  options->grid.Lx = 1.0;
  options->grid.Ly = 1.0;
  options->grid.Lz = 1.0;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nx", &intArg, &found));
  if (found) {
    options->grid.nx = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ny", &intArg, &found));
  if (found) {
    options->grid.ny = intArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nz", &intArg, &found));
  if (found) {
    options->grid.nz = intArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lx", &realArg, &found));
  if (found) {
    options->grid.Lx = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Ly", &realArg, &found));
  if (found) {
    options->grid.Ly = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-Lz", &realArg, &found));
  if (found) {
    options->grid.Lz = realArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &intArg, &found));
  if (found) {
    options->particles.n = intArg;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=user->grid.nx;
  PetscInt       ny=user->grid.ny;
  PetscInt       nz=user->grid.nz;
  DMBoundaryType xBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType yBC=DM_BOUNDARY_PERIODIC;
  DMBoundaryType zBC=DM_BOUNDARY_PERIODIC;
  PetscInt       dof=2;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(
            PETSC_COMM_WORLD,
            xBC, yBC, zBC,
            DMDA_STENCIL_BOX,
            nx, ny, nz,
            PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
            dof, width,
            NULL, NULL, NULL,
            mesh));
  PetscCall(DMDASetElementType(*mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*mesh));
  PetscCall(DMSetUp(*mesh));
  PetscCall(DMDASetUniformCoordinates(
            *mesh,
            0.0, user->grid.Lx,
            0.0, user->grid.Ly,
            0.0, user->grid.Lz));
  PetscCall(DMSetApplicationContext(*mesh, user));
  PetscCall(DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateSwarmDM(DM *swarm, DM *mesh, UserContext *user)
{
  PetscInt dim;
  PetscInt bufsize=0;
  PetscInt np;
  MPI_Comm comm;
  int      size;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "Ions"));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(
            *swarm, "potential", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(
            *swarm, "density", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(PetscObjectGetComm((PetscObject)*mesh, &comm));
  MPI_Comm_size(comm, &size);
  np = user->particles.n / size;
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Setting local sizes <<<\n"));
  PetscCall(DMSwarmSetLocalSizes(*swarm, np, bufsize));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticles(DM *mesh, DM *swarm, UserContext *user, PetscInt n0pc)
{
  PetscReal min[NDIM], max[NDIM];
  PetscInt ndir[NDIM];

  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Inserting points using cell DM <<<\n"));
  PetscCall(DMSwarmInsertPointsUsingCellDM(
            *swarm, DMSWARMPIC_LAYOUT_REGULAR, n0pc));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  min[0] = 0.0;
  max[0] = 1.0;
  min[1] = 0.0;
  max[1] = 1.0;
  min[2] = 0.7;
  max[2] = 0.9;
  ndir[0] = ndir[1] = ndir[2] = 10;
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Adding points using uniform coordinates <<<\n"));
  PetscCall(DMSwarmSetPointsUniformCoordinates(
            *swarm, min, max, ndir, ADD_VALUES));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_FALSE));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  min[0] = 0.0;
  max[0] = 1.0;
  min[1] = 0.0;
  max[1] = 1.0;
  min[2] = 0.7;
  max[2] = 0.9;
  ndir[0] = ndir[1] = ndir[2] = 10;
  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Adding points using uniform coordinates <<<\n"));
  PetscCall(DMSwarmSetPointsUniformCoordinates(
            *swarm, min, max, ndir, ADD_VALUES));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(DMView(*swarm, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;
  KSP         ksp;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n********** START **********\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete mesh.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle swarm.
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Set initial particle positions and velocities.
  PetscCall(InitializeParticles(&mesh, &swarm, &user, 1));

  // Free memory.
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n*********** END ***********\n"));
  PetscCall(PetscFinalize());

  return 0;
}