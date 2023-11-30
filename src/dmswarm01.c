static char help[] = "Examine DMSWARM particle initialzation methods.";

#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmswarm.h>


#define NDIM 3

typedef struct {
  PetscInt nx, ny, nz;
  PetscReal Lx, Ly, Lz;
  PetscReal x0, y0, z0;
  PetscReal x1, y1, z1;
  DMBoundaryType bc;
} Grid;

typedef struct {
  PetscInt np;
  PetscInt npc;
  PetscInt npd[NDIM];
  PetscBool remove;
} Particles;

typedef struct {
  Grid grid;
  Particles particles;
  PetscReal xshift;
  PetscReal yshift;
  PetscReal zshift;
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

-np: The total number of particles

-npc: The number of particles per cell when using
`DMSwarmInsertPointsUsingCellDM`

-npd: The number of particles along each dimension when using
`DMSwarmSetPointsUniformCoordinates`

--periodic: Use periodic boundaries for the grid

--remove: Remove particles from their original MPI rank after migration
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
  options->xshift = 0.0;
  options->yshift = 0.0;
  options->zshift = 0.0;
  options->particles.remove = PETSC_FALSE;

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
  PetscCall(PetscOptionsGetReal(NULL, NULL, "--x-shift", &realArg, &found));
  if (found) {
    options->xshift = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "--y-shift", &realArg, &found));
  if (found) {
    options->yshift = realArg;
  }
  PetscCall(PetscOptionsGetReal(NULL, NULL, "--z-shift", &realArg, &found));
  if (found) {
    options->zshift = realArg;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &intArg, &found));
  if (found) {
    options->particles.np = intArg;
  } else {
    options->particles.np = options->grid.nx * options->grid.ny * options->grid.nz;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npc", &intArg, &found));
  if (found) {
    options->particles.npc = intArg;
  } else {
    options->particles.npc = 1;
  }
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-npd", &intArg, &found));
  if (found) {
    options->particles.npd[0] = intArg;
    options->particles.npd[1] = intArg;
    options->particles.npd[2] = intArg;
  } else {
    options->particles.npd[0] = options->grid.nx;
    options->particles.npd[1] = options->grid.ny;
    options->particles.npd[2] = options->grid.nz;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--periodic", &boolArg, &found));
  if (found) {
    options->grid.bc = DM_BOUNDARY_PERIODIC;
  } else {
    options->grid.bc = DM_BOUNDARY_GHOSTED;
  }
  PetscCall(PetscOptionsGetBool(NULL, NULL, "--remove", &boolArg, &found));
  if (found) {
    options->particles.remove = boolArg;
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
ViewSwarm(DM swarm)
{
  PetscInt npT;

  PetscFunctionBeginUser;

  PetscCall(DMView(swarm, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMSwarmGetSize(swarm, &npT));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total number of particles: %d\n", npT));

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
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm, "potential", 1, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*swarm, "density", 1, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(PetscObjectGetComm((PetscObject)*mesh, &comm));
  MPI_Comm_size(comm, &size);
  np = user->particles.np / size;
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Setting local sizes <<<\n"));
  PetscCall(DMSwarmSetLocalSizes(*swarm, np, bufsize));
  PetscCall(DMSwarmMigrate(*swarm, PETSC_TRUE));
  PetscCall(ViewSwarm(*swarm));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
ShiftParticles(DM swarm, UserContext user, PetscBool rmpart)
{
  PetscReal *coords;
  PetscInt   ip, np;
  PetscReal  xshift=user.xshift;
  PetscReal  yshift=user.yshift;
  PetscReal  zshift=user.zshift;
  PetscReal  Lx=user.grid.Lx;
  PetscReal  Ly=user.grid.Ly;
  PetscReal  Lz=user.grid.Lz;
  PetscReal  x, y, z;

  PetscFunctionBeginUser;

  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetLocalSize(swarm, &np));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Shifting particle positions by (%g, %g, %g) <<<\n", xshift, yshift, zshift));
  for (ip=0; ip<np; ip++) {
    x = coords[ip*NDIM + 0];
    y = coords[ip*NDIM + 1];
    z = coords[ip*NDIM + 2];
    x += xshift;
    while (x > Lx) {x -= Lx;}
    while (x < 0)  {x += Lx;}
    y += yshift;
    while (y > Ly) {y -= Ly;}
    while (y < 0)  {y += Ly;}
    z += zshift;
    while (z > Lz) {z -= Lz;}
    while (z < 0)  {z += Lz;}
    coords[ip*NDIM + 0] = x;
    coords[ip*NDIM + 1] = y;
    coords[ip*NDIM + 2] = z;
  }
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(ViewSwarm(swarm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(swarm, rmpart));
  PetscCall(ViewSwarm(swarm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating again <<<\n"));
  PetscCall(DMSwarmMigrate(swarm, rmpart));
  PetscCall(ViewSwarm(swarm));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticlesFromCellDM(DM *swarm, UserContext *user, PetscBool rmpart)
{
  PetscFunctionBeginUser;

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Inserting points using cell DM <<<\n"));
  PetscCall(DMSwarmInsertPointsUsingCellDM(*swarm, DMSWARMPIC_LAYOUT_REGULAR, user->particles.npc));
  PetscCall(ViewSwarm(*swarm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, rmpart));
  PetscCall(ViewSwarm(*swarm));
  PetscCall(ShiftParticles(*swarm, *user, rmpart));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
InitializeParticlesFromCoordinates(DM *swarm, UserContext *user, PetscBool rmpart)
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
  ndir[0] = user->particles.npd[0];
  ndir[1] = user->particles.npd[1];
  ndir[2] = user->particles.npd[2];

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Inserting points using uniform coordinates <<<\n"));
  PetscCall(DMSwarmSetPointsUniformCoordinates(*swarm, min, max, ndir, INSERT_VALUES));
  PetscCall(ViewSwarm(*swarm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n>>> Migrating <<<\n"));
  PetscCall(DMSwarmMigrate(*swarm, rmpart));
  PetscCall(ViewSwarm(*swarm));
  PetscCall(ShiftParticles(*swarm, *user, rmpart));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode
OutputSwarmBinary(DM *swarm, const char *insert, UserContext *user)
{
  char          posfn[PETSC_MAX_PATH_LEN]="position";
  PetscViewer   viewer;
  Vec           target;

  PetscFunctionBeginUser;

  // Build the full file name.
  PetscCall(PetscStrcat(posfn, insert));
  PetscCall(PetscStrcat(posfn, ".bin"));

  // Create the binary viewer.
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, posfn, FILE_MODE_WRITE, &viewer));

  // Write particle positions to a binary file.
  PetscCall(DMSwarmCreateGlobalVectorFromField(*swarm, DMSwarmPICField_coor, &target));
  PetscCall(PetscObjectSetName((PetscObject)target, "position"));
  PetscCall(VecView(target, viewer));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(*swarm, DMSwarmPICField_coor, &target));

  // Destroy the binary viewer.
  PetscCall(PetscViewerDestroy(&viewer));

  // Destroy the temporary vector object.
  PetscCall(VecDestroy(&target));

  PetscFunctionReturn(PETSC_SUCCESS);
}


int main(int argc, char **args)
{
  UserContext user;
  DM          mesh, swarm;
  char        rmstr[2048]="";

  PetscFunctionBeginUser;

  // Initialize PETSc and MPI.
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n************************* START **********************\n\n"));

  // Assign parameter values from user arguments or defaults.
  PetscCall(ProcessOptions(&user));

  // Set up discrete mesh.
  PetscCall(CreateMeshDM(&mesh, &user));

  // Set up particle swarm.
  PetscCall(CreateSwarmDM(&swarm, &mesh, &user));

  // Output initial positions.
  PetscCall(OutputSwarmBinary(&swarm, "-setup", &user));

  // Set initial particle positions and velocities.
  if (user.particles.remove) {
    strcpy(rmstr, "-------------------- WITH REMOVAL -----------------------");
  } else {
    strcpy(rmstr, "-------------------- WITHOUT REMOVAL --------------------");
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n%s\n\n", rmstr));
  PetscCall(InitializeParticlesFromCellDM(&swarm, &user, user.particles.remove));
  PetscCall(OutputSwarmBinary(&swarm, "-from-celldm", &user));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n\n%s\n\n", rmstr));
  PetscCall(InitializeParticlesFromCoordinates(&swarm, &user, user.particles.remove));
  PetscCall(OutputSwarmBinary(&swarm, "-from-coords", &user));

  // Free memory.
  PetscCall(DMDestroy(&mesh));
  PetscCall(DMDestroy(&swarm));

  // Finalize PETSc and MPI.
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\n************************* END *************************\n"));
  PetscCall(PetscFinalize());

  return 0;
}