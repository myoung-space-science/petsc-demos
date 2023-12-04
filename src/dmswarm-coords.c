static char help[] = "MWE to initialize DMSwarm from coordinates.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>

#define NDIM 3

typedef struct {
  PetscInt       nx, ny, nz;
  PetscReal      Lx, Ly, Lz;
  PetscReal      x0, y0, z0;
  PetscReal      x1, y1, z1;
  DMBoundaryType bc;
} Grid;

typedef struct {
  PetscInt np;
} Particles;

typedef struct {
  Grid      grid;
  Particles particles;
} UserContext;


/* Read parameter values from the options database.

-nx: The number of grid cells in the x dimension

-ny: The number of grid cells in the y dimension

-nz: The number of grid cells in the z dimension

-Lx: The length of the x dimension

-Ly: The length of the y dimension

-Lz: The length of the z dimension

-x0: The lower bound of particle positions in the x dimension

-y0: The lower bound of particle positions in the y dimension

-z0: The lower bound of particle positions in the z dimension

-x1: The upper bound of particle positions in the x dimension

-y1: The upper bound of particle positions in the y dimension

-z1: The upper bound of particle positions in the z dimension

--x-shift: The amount by which to shift particle positions in the x dimension

--y-shift: The amount by which to shift particle positions in the y dimension

--z-shift: The amount by which to shift particle positions in the z dimension

-np: The number of particles along every dimension

--periodic: Use periodic boundaries for the grid
*/
static PetscErrorCode
ProcessOptions(UserContext *options)
{
  PetscFunctionBeginUser;

  PetscInt  intArg;
  PetscReal realArg;
  PetscBool boolArg, found;

  options->grid.nx = 7;
  options->grid.ny = 7;
  options->grid.nz = 7;
  options->grid.Lx = 1.0;
  options->grid.Ly = 1.0;
  options->grid.Lz = 1.0;
  options->grid.x0 = 0.0;
  options->grid.y0 = 0.0;
  options->grid.z0 = 0.0;
  options->grid.x1 = options->grid.Lx;
  options->grid.y1 = options->grid.Ly;
  options->grid.z1 = options->grid.Lz;

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
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x0", &realArg, &found));
  if (found) {
    options->grid.x0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y0", &realArg, &found));
  if (found) {
    options->grid.y0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z0", &realArg, &found));
  if (found) {
    options->grid.z0 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-x1", &realArg, &found));
  if (found) {
    options->grid.x1 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-y1", &realArg, &found));
  if (found) {
    options->grid.y1 = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-z1", &realArg, &found));
  if (found) {
    options->grid.z1 = realArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &intArg, &found));
  if (found) {
    options->particles.np = intArg;
  } else {
    options->particles.np = -1;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--periodic", &boolArg, &found));
  if (found) {
    options->grid.bc = DM_BOUNDARY_PERIODIC;
  } else {
    options->grid.bc = DM_BOUNDARY_GHOSTED;
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateMeshDM(DM *mesh, UserContext *user)
{
  PetscInt       nx=user->grid.nx;
  PetscInt       ny=user->grid.ny;
  PetscInt       nz=user->grid.nz;
  DMBoundaryType xBC=user->grid.bc;
  DMBoundaryType yBC=user->grid.bc;
  DMBoundaryType zBC=user->grid.bc;
  PetscInt       dof=2;
  PetscInt       width=1;

  PetscFunctionBeginUser;

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, xBC, yBC, zBC, DMDA_STENCIL_BOX, nx, ny, nz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, width, NULL, NULL, NULL, mesh));
  PetscCall(DMDASetElementType(*mesh, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*mesh));
  PetscCall(DMSetUp(*mesh));
  PetscCall(DMDASetUniformCoordinates(*mesh, 0.0, user->grid.Lx, 0.0, user->grid.Ly, 0.0, user->grid.Lz));
  PetscCall(DMSetApplicationContext(*mesh, user));
  PetscCall(DMView(*mesh, PETSC_VIEWER_STDOUT_WORLD));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ViewSwarm(DM swarm, const char *filestem, UserContext user)
{
  char        binname[PETSC_MAX_PATH_LEN]="";
  char        xmfname[PETSC_MAX_PATH_LEN]="";
  PetscInt    npG, npL;
  PetscViewer viewer;
  Vec         target;
  int         rank;

  PetscFunctionBeginUser;

  // Build the binary file name.
  PetscCall(PetscStrcat(binname, filestem));
  PetscCall(PetscStrcat(binname, ".bin"));

  // Echo swarm information.
  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmGetSize(swarm, &npG));
  PetscCall(DMSwarmGetLocalSize(swarm, &npL));
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] Total number of particles: %d (local) / %d (global)\n", rank, npG, npL));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));

  // Create the binary viewer.
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, binname, FILE_MODE_WRITE, &viewer));

  // Write particle positions to a binary file.
  PetscCall(DMSwarmCreateGlobalVectorFromField(swarm, DMSwarmPICField_coor, &target));
  PetscCall(PetscObjectSetName((PetscObject)target, "position"));
  PetscCall(VecView(target, viewer));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(swarm, DMSwarmPICField_coor, &target));

  // Destroy the binary viewer.
  PetscCall(PetscViewerDestroy(&viewer));

  // Destroy the temporary vector object.
  PetscCall(VecDestroy(&target));

  // Build the XDMF file name.
  PetscCall(PetscStrcat(xmfname, filestem));
  PetscCall(PetscStrcat(xmfname, ".xmf"));

  // Write particle positions to an XDMF file.
  PetscCall(DMSwarmViewXDMF(swarm, xmfname));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
CreateSwarmDM(DM *swarm, DM *mesh, UserContext *user)
{
  PetscInt dim;
  PetscInt bufsize;
  PetscInt np;
  MPI_Comm comm;
  int      size;

  PetscFunctionBeginUser;

  PetscCall(DMCreate(PETSC_COMM_WORLD, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)(*swarm), "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "ions"));
  PetscCall(DMGetDimension(*mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, *mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(PetscObjectGetComm((PetscObject)*mesh, &comm));
  MPI_Comm_size(comm, &size);
  if (user->particles.np < 0) {
    np = user->grid.nx * user->grid.ny * user->grid.nz;
    bufsize = np;
  } else {
    np = user->particles.np;
    bufsize = 0;
  }
  PetscCall(DMSwarmSetLocalSizes(*swarm, np / size, bufsize));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(ViewSwarm(*swarm, "coords-0", *user));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticles(DM *swarm, UserContext *user)
{
  PetscReal min[NDIM], max[NDIM];
  PetscInt ndir[NDIM];

  PetscFunctionBeginUser;

  min[0] = user->grid.x0;
  max[0] = user->grid.x1;
  min[1] = user->grid.y0;
  max[1] = user->grid.y1;
  min[2] = user->grid.z0;
  max[2] = user->grid.z1;
  if (user->particles.np < 0) {
    ndir[0] = user->grid.nx;
    ndir[1] = user->grid.ny;
    ndir[2] = user->grid.nz;
  } else {
    ndir[0] = user->particles.np;
    ndir[1] = user->particles.np;
    ndir[2] = user->particles.np;
  }

  PetscCall(DMSwarmSetPointsUniformCoordinates(*swarm, min, max, ndir, INSERT_VALUES));
  PetscCall(ViewSwarm(*swarm, "coords-1", *user));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(ViewSwarm(*swarm, "coords-2", *user));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete-mesh DM.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle-swarm DM.
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Initialize particle coordinates.
  PetscCall(InitializeParticles(&swarm, &user));

  // Free memory.
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscFinalize());

  return 0;
}

