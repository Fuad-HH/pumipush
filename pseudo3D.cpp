#include <iostream>

#include "pumipush.h"

/**
 * Pseudo Monte Carlo simulation to simulate particle transport
 * The following steps will be performed:
 * 1. Initial source sampling
 * 2. Particles moved to new location
 * 3. Particles are tracked
 * 4. Particles are tallied
 */
int main(int argc, char* argv[]) {
  pumipic::Library pic_lib(&argc, &argv);
  Omega_h::Library& lib = pic_lib.omega_h_lib();
  int comm_rank, comm_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  if (!comm_rank)
    std::cout << "PUMIPic Monte Carlo Transport Simulation on " << comm_size
              << " ranks\n";
  // ******************* Argument Parsing ******************* //

  if (argc < 6) {
    std::cerr << "Error: missing required arguments\n";
    std::cout << "Usage: " << argv[0]
              << " <mesh> <partition_file> <num_particles> <num_iterations> "
                 "<lambda>\n";
    return 1;
  }

  std::string mesh_file_name = argv[1];
  // mesh file name has to be non-empty
  if (mesh_file_name.empty()) {
    std::cerr << "Error: mesh file name cannot be empty\n";
    return 1;
  }
  int num_particles = atoi(argv[3]);
  int num_iterations = atoi(argv[4]);
  double lambda = atof(argv[5]);
  std::string partition_file = argv[2];

  // ******************* Mesh Loading ******************* //
  o::Mesh full_mesh = readMesh(mesh_file_name, lib);
  // o::reorder_by_hilbert(&full_mesh);
  // if (comm_rank == 0) std::cout << "Mesh Hilbert reordered\n";
  if (comm_rank == 0) {
    std::cout << "Mesh loaded sucessfully with " << full_mesh.nelems()
              << " elements\n\t\t\t"
                 "and "
              << full_mesh.nverts() << " vertices\n";
  }

  // ******** Mesh bounding box ******** //
  // Omega_h::BBox<3> bb = Omega_h::find_bounding_box<3>(full_mesh.coords());

  // ******************* Partition Loading ******************* //
  o::Write<o::LO> owners = o::Write<o::LO>(full_mesh.nelems(), 0);

  partitionMeshEqually(full_mesh, owners, comm_size, comm_rank);
#ifdef DEBUG
  printf("INFO: Owners list of the mesh created successfully\n");
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  p::Mesh picparts(full_mesh, owners, 3, 2);
  printf("Partitioned the mesh successfully\n");

  o::Mesh* mesh = picparts.mesh();

  /// Omega_h::vtk::write_parallel("initialPartition.vtk", mesh, 2);
  // std::cout << "Initial partition written to initialPartition.vtk\n";

  // ******************* Particle Initialization ******************* //
  Omega_h::Int ne = mesh->nelems();

#ifdef DEBUG
  MPI_Barrier(MPI_COMM_WORLD);
  // printf("Number of particles: \t%d in rank \t%d\n", num_particles,
  // comm_rank);
  printf("INFO: Number of elements: \t%d in rank %d\n", ne, comm_rank);
#endif

  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(  // ? what is this doing? copying from GOs to kkGidView
      ne,
      OMEGA_H_LAMBDA(const int& i) { element_gids(i) = mesh_element_gids[i]; });

  // int numPtcls = setSourceElements(picparts, ptcls_per_elem, 0,
  // num_particles);
  int total_num_particles = num_particles;
  // if (comm_rank != 0) num_particles = 0;
  {
    double element_fraction = double(ne) / full_mesh.nelems();
    num_particles = int(num_particles * element_fraction);
  }
  int setPtcls =
      distributeParticlesEqually(picparts, ptcls_per_elem, num_particles);
  printf("INFO: Number of particles set to elements: \t %d in rank %d\n",
         setPtcls, comm_rank);

  Kokkos::TeamPolicy<ExeSpace> policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
  // Sell-C-Sigma particle structure: see the pumipic paper for more details
  PS* ptcls = new SellCSigma<Particle>(policy, INT_MAX, 10, ne, setPtcls,
                                       ptcls_per_elem, element_gids);
#ifdef DEBUG
  printf("INFO: Particle structure created successfully in rank %d\n",
         comm_rank);
#endif

  setInitialPtclCoords(picparts, ptcls);
#ifdef DEBUG
  printf("INFO: Initial positions of particles set... in rank %d\n", comm_rank);
#endif
  setPtclIds(ptcls);
#ifdef DEBUG
  printf("INFO: Particles initialized with ids... in rank %d\n", comm_rank);
#endif

  o::LOs elmTags(ne, -1, "elmTagVals");
  mesh->add_tag(o::FACE, "has_particles", 1,
                elmTags);  // ncomp is number of components
  mesh->add_tag(o::VERT, "avg_density", 1, o::Reals(mesh->nverts(), 0));
  render(picparts, 0, comm_rank);

  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;

  int iter;  // iteration number
  int np;    //
  int ps_np;

  // ******************* Monte Carlo Transport Simulation ******************* //
  for (iter = 1; iter <= num_iterations; ++iter) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0) printf("-----------------------------\n");
    printf("Iteration: %d Rank: %d\n", iter, comm_rank);
    // 1. check the remaining number of particles
    ps_np = ptcls->nPtcls();
    std::cout << "Number of particles: " << ps_np << " in rank " << comm_rank
              << '\n';
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    timer.reset();
    // 2. push particles
    pseudo2Dpush(ptcls, lambda);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0)
      fprintf(stderr, "TIME: push and transfer (seconds) %f\n",
              timer.seconds());
    timer.reset();
    // 3. search for the new element
    search(picparts, ptcls, false);
    if (comm_rank == 0)
      fprintf(stderr, "search, rebuild, and transfer (seconds) %f\n",
              timer.seconds());
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    // 4. tag the parent elements
    tagParentElements(picparts, ptcls, iter);
    render(picparts, iter, comm_rank);
  }

  Omega_h::vtk::write_parallel("pseudoMC_vtk", mesh, picparts.dim());
  if (comm_rank == 0)
    fprintf(stderr, "%d iterations of pseudopush (seconds) %f\n", iter,
            fullTimer.seconds());

  // cleanup
  delete ptcls;

  pumipic::SummarizeTime();
  if (!comm_rank) fprintf(stderr, "done\n");
  return 0;
}