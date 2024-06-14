#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_mesh.hpp>
#include <fstream>
#include <particle_structs.hpp>
#include <pumipic_kktypes.hpp>
#include <random>

#include "pumipic_adjacency.hpp"
#include "pumipic_mesh.hpp"
#include "pumipic_ptcl_ops.hpp"
// #include <team_policy.hpp>
#define NUM_ITERATIONS 30

using particle_structs::lid_t;
using particle_structs::MemberTypes;
using particle_structs::SellCSigma;
using pumipic::fp_t;
using pumipic::Vector3d;

namespace o = Omega_h;
namespace p = pumipic;
namespace ps = particle_structs;

// To demonstrate push and adjacency search we store:
//-two fp_t[3] arrays, 'Vector3d', for the current and
//  computed (pre adjacency search) positions, and
//-an integer to store the particles id
typedef MemberTypes<Vector3d, Vector3d, int> Particle;
typedef ps::ParticleStructure<Particle> PS;
typedef Kokkos::DefaultExecutionSpace ExeSpace;

// ******************** Function Prototypes ******************** //
/**
 * @brief Generate a random path length using an exponential distribution
 * TODO: make this lambda a mean free path length
 */
inline double random_path_length(double lambda);

o::Mesh readMesh(std::string meshFile, o::Library& lib);

/**
 * Populate the particles equally to all elements
 */
int distributeParticlesEqually(const p::Mesh& picparts, PS::kkLidView ppe,
                               const int numPtcls);

/**
 * Set the initial particle coordinates to the centroid of the parent element
 */
void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls);

/**
 * Set the particle ids to the particle index
 */
void setPtclIds(PS* ptcls);

/**
 * Render the mesh with the particles
 */
void render(p::Mesh& picparts, int iter, int comm_rank);

/**
 * Print the timing of the operation
 */
void printTiming(const char* name, double t);

/**
 * Push the particles in the direction of the vector
 */
void push(PS* ptcls, int np, double lambda);

/**
 * Get a random direction uniformly distributed on the unit sphere
 */
inline std::vector<double> getDirection(const double A = 1.0);

/**
 * Update the particle positions to the new target positions
 */
void updatePtclPositions(PS* ptcls);

/**
 * Rebuild the particle structure with the new element ids
 * It just calls the *migrate_lb_ptcls* function
 */
void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, const bool output);

/**
 * Search for the new element for the particles
 */
void search(p::Mesh& picparts, PS* ptcls, bool output);

/**
 * Tag the parent elements with the number of particles
 */
void tagParentElements(p::Mesh& picparts, PS* ptcls, int loop);

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
  o::reorder_by_hilbert(&full_mesh);
  if (comm_rank == 0) std::cout << "Mesh Hilbert reordered\n";
  std::cout << "Mesh loaded sucessfully with " << full_mesh.nelems()
            << " elements\n\t\t\t"
               "and "
            << full_mesh.nverts() << " vertices\n";

  // ******************* Mesh Partitioning ******************* //
  // ? Does the owner file has mapping for each element to a rank?
  Omega_h::HostWrite<Omega_h::LO> host_owners(full_mesh.nelems());
  if (comm_size > 1) {
    std::ifstream in_str(partition_file);
    if (!in_str) {
      if (!comm_rank)
        std::cerr << "Error: could not open partition file " << partition_file
                  << '\n';
      return EXIT_FAILURE;
    }
    int own;
    int index = 0;
    while (in_str >> own) host_owners[index++] = own;
  } else {
    for (int i = 0; i < full_mesh.nelems(); ++i) host_owners[i] = 0;
  }
  Omega_h::Write<Omega_h::LO> owner(host_owners);

  // *********** Create Picparts with the full mesh
  p::Mesh picparts(full_mesh, owner);  // Constucts PIC parts with a core and
                                       // the entire mesh as buffer/safe
  o::Mesh* mesh = picparts.mesh();

  // ******************* Particle Initialization ******************* //
  if (comm_rank != 0) num_particles = 0;
  Omega_h::Int ne = mesh->nelems();
  if (comm_rank == 0) {
    std::cout << "Number of particles: \t" << num_particles << '\n';
    std::cout << "Number of elements: \t" << ne << '\n';
  }

  PS::kkLidView ptcls_per_elem("ptcls_per_elem", ne);
  PS::kkGidView element_gids("element_gids", ne);
  Omega_h::GOs mesh_element_gids = picparts.globalIds(picparts.dim());
  Omega_h::parallel_for(  // ? what is this doing? copying from GOs to kkGidView
      ne,
      OMEGA_H_LAMBDA(const int& i) { element_gids(i) = mesh_element_gids[i]; });

  // int numPtcls = setSourceElements(picparts, ptcls_per_elem, 0,
  // num_particles);
  int setPtcls =
      distributeParticlesEqually(picparts, ptcls_per_elem, num_particles);
  if (comm_rank == 0) {
    std::cout << "Number of particles set to elements: \t" << setPtcls << '\n';
  }

  Kokkos::TeamPolicy<ExeSpace> policy =
      Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(10000, Kokkos::AUTO());
  // Sell-C-Sigma particle structure: see the pumipic paper for more details
  PS* ptcls = new SellCSigma<Particle>(policy, INT_MAX, 10, ne, setPtcls,
                                       ptcls_per_elem, element_gids);
  setInitialPtclCoords(picparts, ptcls);
  setPtclIds(ptcls);

  o::LOs elmTags(ne, -1, "elmTagVals");
  mesh->add_tag(o::REGION, "has_particles", 1,
                elmTags);  // ncomp is number of components
  mesh->add_tag(o::VERT, "avg_density", 1, o::Reals(mesh->nverts(), 0));
  render(picparts, 0, comm_rank);

  Kokkos::Timer timer;
  Kokkos::Timer fullTimer;

  int iter;  // iteration number
  int np;    //
  int ps_np;

  // ******************* Monte Carlo Transport Simulation ******************* //
  for (iter = 0; iter < num_iterations; ++iter) {
    if (comm_rank == 0) std::cout << "Iteration: " << iter << '\n';
    // 1. check the remaining number of particles
    ps_np = ptcls->nPtcls();
    MPI_Allreduce(&ps_np, &np, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (np == 0) {
      fprintf(stderr, "No particles remain... exiting push loop\n");
      break;
    }
    timer.reset();
    // 2. push particles
    push(ptcls, np, lambda);
    MPI_Barrier(MPI_COMM_WORLD);
    if (comm_rank == 0)
      fprintf(stderr, "push and transfer (seconds) %f\n", timer.seconds());
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

// ******************** Function Definitions ******************** //

inline double random_path_length(double lambda) {
  // ref: https://docs.openmc.org/en/stable/methods/neutron_physics.html#sampling-distance-to-next-collision
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  double l = - std::log(dis(gen)) * lambda;
  return l;
}

o::Mesh readMesh(std::string meshFile, o::Library& lib) {
  (void)lib;
  std::string fn(meshFile);
  // if there is a forward slash at the end of the filename, remove the forward
  // slash e.g.  coarseMesh.osh/  ->  coarseMesh.osh
  if (fn.back() == '/') {
    fn.pop_back();
  }

  auto ext = fn.substr(fn.find_last_of(".") + 1);
  if (ext == "osh") {
    std::cout << "Reading omegah mesh " << meshFile << "\n";
    return Omega_h::binary::read(meshFile, lib.self());
  } else {
    std::cout
        << "Only supports Omega_h mesh. \nerror: unrecognized mesh extension \'"
        << ext << "\'\n";
    exit(EXIT_FAILURE);
  }
}

int distributeParticlesEqually(const p::Mesh& picparts, PS::kkLidView ppe,
                               const int numPtcls) {
  auto ne = picparts.mesh()->nelems();
  auto numPpe = numPtcls / ne;
  auto numPpeR = numPtcls % ne;
  o::parallel_for(
      ne, OMEGA_H_LAMBDA(const int i) {
        if (i == ne - 1)
          ppe[i] = numPpe + numPpeR;
        else
          ppe[i] = numPpe;
      });
  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(
      ppe.size(),
      OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) { lsum += ppe[i]; },
      totPtcls);
  assert(totPtcls == numPtcls);
  return totPtcls;
}

// HACK to avoid having an unguarded comma in the PS PARALLEL macro
OMEGA_H_DEVICE o::Matrix<3, 4> gatherVectors(o::Reals const& a,
                                             o::Few<o::LO, 4> v) {
  return o::gather_vectors<4, 3>(a, v);
}

void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls) {
  // get centroid of parent element and set the child particle coordinates
  // most of this is copied from Omega_h_overlay.cpp get_cell_center_location
  // It isn't clear why the template parameter for gather_[verts|vectors] was
  // sized eight... maybe something to do with the 'Overlay'.  Given that there
  // are four vertices bounding a tet, I'm setting that parameter to four below.
  o::Mesh* mesh = picparts.mesh();
  auto cells2nodes = mesh->get_adj(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh->coords();
  // set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();  // ? I am not quite sure what get does:
                                  // ? may be gets the position, 1 means the
                                  // velocity and 2 means the id
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
      auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
      auto center = average(cell_nodes2coords);
      for (int i = 0; i < 3; i++) x_ps_d(pid, i) = center[i];
    }
  };
  ps::parallel_for(ptcls, lamb);
}

void setPtclIds(PS* ptcls) {
  auto pid_d = ptcls->get<2>();
  auto setIDs = PS_LAMBDA(const int& eid, const int& pid, const bool& mask) {
    pid_d(pid) = pid;
  };
  ps::parallel_for(ptcls, setIDs);
}

void render(p::Mesh& picparts, int iter, int comm_rank) {
  std::stringstream ss;
  ss << "pseudoMC_t" << iter << "_r" << comm_rank;
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, picparts.mesh(), picparts.dim());
}
void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void push(PS* ptcls, int np, double lambda) {
  Kokkos::Timer timer;
  auto position_d = ptcls->get<0>();
  auto new_position_d = ptcls->get<1>();

  double totTime = 0;
  timer.reset();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      // fp_t dir[3];
      double distance = random_path_length(lambda);
      std::vector<double> disp_d = getDirection();
      new_position_d(pid, 0) = position_d(pid, 0) + distance * disp_d[0];
      new_position_d(pid, 1) = position_d(pid, 1) + distance * disp_d[1];
      new_position_d(pid, 2) = position_d(pid, 2) + distance * disp_d[2];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
}

inline std::vector<double> getDirection(const double A) {
  // ref https://docs.openmc.org/en/stable/methods/neutron_physics.html#isotropic-angular-distribution
  // ref
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  double mu = 2 * dis(gen) - 1;
  // cosine in the particles incident direction
  double mu_lab = (1 + A*mu) / std::sqrt(1 + 2*A*mu + A*A);
  // cosine with the plane of the collision
  double nu_lab = 2 * dis (gen) - 1;
  std::vector<double> dir(3);

  // TODO: replace this dummy direction with the actual direction
  // actual direction needs the incident direction
  dir[0] = std::sqrt(1 - mu_lab * mu_lab) * std::cos(2 * M_PI * nu_lab);
  dir[1] = std::sqrt(1 - mu_lab * mu_lab) * std::sin(2 * M_PI * nu_lab);
  dir[2] = mu_lab;
  return dir;
}

void updatePtclPositions(PS* ptcls) {
  auto x_ps_d = ptcls->get<0>();
  auto xtgt_ps_d = ptcls->get<1>();
  auto updatePtclPos = PS_LAMBDA(const int&, const int& pid, const bool&) {
    x_ps_d(pid, 0) = xtgt_ps_d(pid, 0);
    x_ps_d(pid, 1) = xtgt_ps_d(pid, 1);
    x_ps_d(pid, 2) = xtgt_ps_d(pid, 2);
    xtgt_ps_d(pid, 0) = 0;
    xtgt_ps_d(pid, 1) = 0;
    xtgt_ps_d(pid, 2) = 0;
  };
  ps::parallel_for(ptcls, updatePtclPos);
}

void rebuild(p::Mesh& picparts, PS* ptcls, o::LOs elem_ids, const bool output) {
  updatePtclPositions(ptcls);
  const int ps_capacity = ptcls->capacity();
  auto ids = ptcls->get<2>();
  auto printElmIds = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (output && mask > 0)
      printf("elem_ids[%d] %d ptcl_id:%d\n", pid, elem_ids[pid], ids(pid));
  };
  ps::parallel_for(ptcls, printElmIds);

  // PS::kkLidView ps_elem_ids("ps_elem_ids", ps_capacity);
  // auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
  //   if (mask) {
  //     int new_elem = elem_ids[pid];
  //     ps_elem_ids(pid) = new_elem;
  //   }
  // };
  // ps::parallel_for(ptcls, lamb);

  pumipic::migrate_lb_ptcls(picparts, ptcls, elem_ids, 1.05);
  pumipic::printPtclImb(ptcls);

  int comm_rank = picparts.comm()->rank();

  printf("PS on rank %d has Elements: %d. Ptcls %d. Capacity %d. Rows %d.\n",
         comm_rank, ptcls->nElems(), ptcls->nPtcls(), ptcls->capacity(),
         ptcls->numRows());
  ids = ptcls->get<2>();
  if (output) {
    auto printElms = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0)
        printf("Rank %d Ptcl: %d has Element %d and id %d\n", comm_rank, pid, e,
               ids(pid));
    };
    ps::parallel_for(ptcls, printElms);
  }
}

void search(p::Mesh& picparts, PS* ptcls, bool output) {
  o::Mesh* mesh = picparts.mesh();
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 100;
  const auto psCapacity = ptcls->capacity();
  o::Write<o::LO> elem_ids;
  Kokkos::Timer timer;
  auto x = ptcls->get<0>();
  auto xtgt = ptcls->get<1>();
  auto pid = ptcls->get<2>();
  o::Write<o::Real> xpoints_d(3 * psCapacity, "intersection points");
  o::Write<o::LO> xface_id(psCapacity, "intersection faces");
  // all sides are considered zone boundaries
  o::Read<o::I8> zone_boundary_sides(mesh->nfaces(), 1);
  bool isFound = p::search_mesh_with_zone<Particle>(
      *mesh, ptcls, x, xtgt, pid, elem_ids, xpoints_d, xface_id,
      zone_boundary_sides, maxLoops);
  fprintf(stderr, "search_mesh (seconds) %f\n", timer.seconds());
  assert(isFound);
  // rebuild the PS to set the new element-to-particle lists
  timer.reset();
  rebuild(picparts, ptcls, elem_ids, output);
  fprintf(stderr, "rebuild (seconds) %f\n", timer.seconds());
}

void tagParentElements(p::Mesh& picparts, PS* ptcls, int loop) {
  o::Mesh* mesh = picparts.mesh();
  // read from the tag
  o::LOs ehp_nm1 = mesh->get_array<o::LO>(picparts.dim(), "has_particles");
  o::Write<o::LO> ehp_nm0(ehp_nm1.size());
  auto set_ehp = OMEGA_H_LAMBDA(o::LO i) { ehp_nm0[i] = ehp_nm1[i]; };
  o::parallel_for(ehp_nm1.size(), set_ehp, "set_ehp");

  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    (void)pid;
    if (mask > 0) ehp_nm0[e] = loop;
  };
  ps::parallel_for(ptcls, lamb);

  o::LOs ehp_nm0_r(ehp_nm0);
  mesh->set_tag(o::REGION, "has_particles", ehp_nm0_r);
}