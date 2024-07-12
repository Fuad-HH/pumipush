#include "pumipush.h"

inline double random_path_length(double lambda) {
  // ref:
  // https://docs.openmc.org/en/stable/methods/neutron_physics.html#sampling-distance-to-next-collision
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  double l = -std::log(dis(gen)) * lambda;
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
    // std::cout << "Reading omegah mesh " << meshFile << "\n";
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
        ppe[i] = numPpe + (i < numPpeR);
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
  // ref
  // https://docs.openmc.org/en/stable/methods/neutron_physics.html#isotropic-angular-distribution
  // ref
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);
  double mu = 2 * dis(gen) - 1;
  // cosine in the particles incident direction
  double mu_lab = (1 + A * mu) / std::sqrt(1 + 2 * A * mu + A * A);
  // cosine with the plane of the collision
  double nu_lab = 2 * dis(gen) - 1;
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

inline void ownerFromCPN(const std::string cpn_file_name,
                         o::Write<o::LO>& owners) {
  std::ifstream cpn_file(cpn_file_name);
  if (!cpn_file.is_open()) {
    std::cerr << "Error: cannot open the cpn file\n";
    exit(1);
  }
  int own;
  int index = -1;
  while (cpn_file >> own) owners[index++] = own;

  if (index != owners.size()) {
    std::cout << "******************* Warning *******************\n"
                 "The number of elements in the cpn file does not match the "
                 "number of elements in the mesh\n";
  }
}

 void partitionMeshEqually(o::Mesh& mesh, o::Write<o::LO> owners, int comm_size, int comm_rank){
  Omega_h::BBox<3> bb = Omega_h::find_bounding_box<3>(mesh.coords());
  // create a vector of 0 comm_size-1
  std::vector<int> ranks(comm_size);
  std::iota(ranks.begin(), ranks.end(), 0);
  // create a vector of cut surfaces for recursive coordinate bisection
  double startx = bb.min[0]; double starty = bb.min[1];
  double deltax = bb.max[0] - bb.min[0]; double deltay = bb.max[1] - bb.min[1];
  int ncuts_x = std::sqrt(comm_size); 
  // decrement ncuts_x until it divides comm_size
  while (comm_size % ncuts_x != 0) {
    ncuts_x--;
  }
  
  int ncuts_y = comm_size/ncuts_x;

  // print the number of cuts in x and y
  std::cout << "Number of cuts in x and y: " << ncuts_x << " " << ncuts_y << '\n';

  double dx = deltax/ncuts_x; double dy = deltay/ncuts_y;

  auto cells2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  auto nodes2coords = mesh.coords();
  auto lamb = OMEGA_H_LAMBDA(o::LO i){
      auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(i));
      auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
      auto center = average(cell_nodes2coords);
      //std::array<double, 3> center_arr = {center[0], center[1], center[2]};
      std::array<double, 2> center2d {center[0], center[1]};
      // assert that the center is within the bounding box
      assert(center[0] >= startx && center[0] <= bb.max[0]);
      assert(center[1] >= starty && center[1] <= bb.max[1]);
      // get rank based on the location of the center starting from the lower-left corner and moving up and right
      int rank = int((center[0] - startx)/dx) + ncuts_x*int((center[1] - starty)/dy);
      assert((rank < comm_size) && (rank >= 0));
      owners[i] = rank;
    };
  o::parallel_for(mesh.nelems(), lamb);
 }