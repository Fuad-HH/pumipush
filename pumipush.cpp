#include "pumipush.h"

#include <Segment.h>
#include <ppMacros.h>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_MinMax.hpp>
#include <Omega_h_adj.hpp>
#include <Omega_h_array.hpp>
#include <Omega_h_bbox.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_shape.hpp>
#include <Omega_h_vector.hpp>
#include <cstdlib>
#include <particle_structure.hpp>
#include <pumipic_kktypes.hpp>

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
  // printf("Number of elements: %d\n", ne);
  auto numPpe = numPtcls / ne;
  auto numPpeR = numPtcls % ne;
  o::parallel_for(
      ne, OMEGA_H_LAMBDA(const int i) { ppe[i] = numPpe + (i < numPpeR); });
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
  int dim = mesh->dim();
  o::LOs cells2nodes;
  if (dim == 3) {
    cells2nodes = mesh->ask_down(o::REGION, o::VERT).ab2b;
  } else if (dim == 2) {
    cells2nodes = mesh->ask_down(o::FACE, o::VERT).ab2b;
  } else {
    std::cerr << "Error: unsupported dimension\n";
    exit(1);
  }
  auto nodes2coords = mesh->coords();
  // set particle positions and parent element ids
  auto x_ps_d = ptcls->get<0>();  // ? I am not quite sure what get does:
                                  // ? may be gets the position, 1 means the
                                  // velocity and 2 means the id
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      if (dim == 2) {
        o::Few<o::Real, 2> center;
        get_tri_centroid(cells2nodes, e, nodes2coords, center);
        // random theta from -pi to pi
        double random_theta = 2 * M_PI * (rand() / (RAND_MAX + 1.0)) - M_PI;
        // r, theta, z
        x_ps_d(pid, 0) = center[0];
        x_ps_d(pid, 1) = random_theta;
        x_ps_d(pid, 2) = center[1];
      } else if (dim == 3) {
        o::Few<o::Real, 3> center;
        get_tet_centroid(cells2nodes, e, nodes2coords, center);
        // x, y, z
        x_ps_d(pid, 0) = center[0];
        x_ps_d(pid, 1) = center[1];
        x_ps_d(pid, 2) = center[2];
      } else {
        std::cerr << "Error: unsupported dimension\n";
        exit(1);
      }
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
  ss << "pseudoMC_t" << iter << "_r" << comm_rank << ".vtk";
  std::string s = ss.str();
  Omega_h::vtk::write_parallel(s, picparts.mesh(), picparts.dim());
}
void printTiming(const char* name, double t) {
  fprintf(stderr, "kokkos %s (seconds) %f\n", name, t);
}

void pseudo2Dpush(PS* ptcls, double lambda) {
  Kokkos::Timer timer;
  auto position_d = ptcls->get<0>();
  auto new_position_d = ptcls->get<1>();

  double totTime = 0;
  timer.reset();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      // fp_t dir[3];
      double distance = random_path_length(lambda);
      o::Vector<3> cyl_coords = {position_d(pid, 0), position_d(pid, 1),
                                 position_d(pid, 2)};
      o::Vector<3> cart_coords;
      cylindrical2cartesian(cyl_coords, cart_coords);

      o::Vector<3> direction_vec = sampleRandomDirection();
      o::Vector<3> new_position_cart = cart_coords + (distance * direction_vec);
      cartesian2cylindrical(new_position_cart, cyl_coords);
      new_position_d(pid, 0) = cyl_coords[0];
      new_position_d(pid, 1) = cyl_coords[1];
      new_position_d(pid, 2) = cyl_coords[2];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
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
      o::Vector<3> disp_d = sampleRandomDirection();
      new_position_d(pid, 0) = position_d(pid, 0) + distance * disp_d[0];
      new_position_d(pid, 1) = position_d(pid, 1) + distance * disp_d[1];
      new_position_d(pid, 2) = position_d(pid, 2) + distance * disp_d[2];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
}

inline o::Vector<3> sampleRandomDirection(const double A) {
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
  o::Vector<3> dir;

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

template <class Vec>
OMEGA_H_DEVICE bool all_positive(const Vec a, Omega_h::Real tol = EPSILON) {
  auto isPos = 1;
  for (Omega_h::LO i = 0; i < a.size(); ++i) {
    const auto gtez = Omega_h::are_close(a[i], 0.0, tol, tol) || a[i] > 0;
    isPos = isPos && gtez;
  }
  return isPos;
}

OMEGA_H_DEVICE bool counter_clockwise(const Omega_h::Vector<2>& a,
                                      const Omega_h::Vector<2>& b,
                                      const Omega_h::Vector<2>& c) {
  return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0]);
}
/// ref: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
OMEGA_H_DEVICE double find_intersection_distance_tri(
    const Omega_h::Vector<2>& a, const Omega_h::Vector<2>& b,
    const Omega_h::Vector<2>& c, const Omega_h::Vector<2>& d) {
  bool intersects = counter_clockwise(a, c, d) != counter_clockwise(b, c, d) &&
                    counter_clockwise(a, b, c) != counter_clockwise(a, b, d);

  // get the intersection distance from the start point a
  Omega_h::Vector<2> u = b - a;
  Omega_h::Vector<2> v = d - c;
  Omega_h::Vector<2> w = a - c;
  double s = (v[0] * w[1] - v[1] * w[0]) / (v[1] * u[0] - v[0] * u[1]);
  return intersects ? s : -1;
}

bool search_adj_elements(o::Mesh& mesh, PS* ptcls,
                         p::Segment<double[3], Kokkos::HostSpace> x,
                         p::Segment<double[3], Kokkos::HostSpace> xtgt,
                         p::Segment<int, Kokkos::HostSpace> pid,
                         o::Write<o::LO> elem_ids, o::Write<o::Real> xpoints_d,
                         o::Write<o::LO> xface_id,
                         o::Read<o::I8> zone_boundary_sides,
                         int looplimit = 10) {
  OMEGA_H_CHECK(mesh.dim() == 2);  // only for pseudo3D now
  o::Real tol = 1.0e-12;
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  const auto coords = mesh.coords();
  const auto faces2nodes = mesh.ask_verts_of(o::FACE);
  const auto face2faceFace = mesh.ask_dual().ab2b;
  const auto face2faceOffsets = mesh.ask_dual().a2ab;
  const auto face2nodeNode =
      mesh.ask_down(o::FACE, o::VERT).ab2b;  // always 3; offset not needed
  const auto node2faceFace = mesh.ask_up(o::VERT, o::FACE).ab2b;
  const auto node2faceOffsets = mesh.ask_up(o::VERT, o::FACE).a2ab;
  const auto psCapacity = ptcls->capacity();

  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_done");
  o::Write<o::LO> elem_ids_next(psCapacity);

  // bool set_ids = false;
  // if (elem_ids.size() == 0) { // ? why this is done
  //   std::cerr << "Warning!!! elem_ids size is zero\n";
  //   elem_ids = o::Write<o::LO>(psCapacity, -1);
  //   set_ids = true;
  // }
  auto fill = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      elem_ids[pid] = e;
    } else {
      ptcl_done[pid] = 1;
    }
  };
  parallel_for(ptcls, fill, "searchMesh_fill_elem_ids");

  bool found = false;

  {  // original search in pumipush does it in a loop but here it is searched
    // only once since it only moves to the adj faces
    auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        auto elmId = elem_ids[pid];
        OMEGA_H_CHECK(elmId >= 0);
        // get the corners of the element
        auto tri2verts = o::gather_verts<3>(faces2nodes, e);
        o::Vector<3> dest = {xtgt(pid, 0), 0.0, xtgt(pid, 2)};
        o::Vector<3> start = {x(pid, 0), 0.0, x(pid, 2)};
        o::Vector<3> bcc;
        Omega_h::Few<Omega_h::Vector<2>, 3> tri_verts;
        for (int i = 0; i < 3; ++i) {
          tri_verts[i] = Omega_h::get_vector<2>(coords, tri2verts[i]);
        }
        {  // check if the particle is in the element
          // find barrycentric coordinates of the
          // find_barycentric_triangle(abc, start, bcc);
          bcc =
              o::barycentric_from_global<2, 2>({start[0], start[2]}, tri_verts);
          if (!all_positive(bcc, tol)) {
            // printf("Starting Position of pid %d in el %d : %f %f %f\n", pid,
            // elmId, start[0], start[1], start[2]);
            //// print abc
            // printf("Start elm vertex %d: %f %f %f\n",tri2verts[0], abc[0][0],
            // abc[0][1], abc[0][2]); printf("Start elm vertex %d: %f %f
            // %f\n",tri2verts[1], abc[1][0], abc[1][1], abc[1][2]);
            // printf("Start elm vertex %d: %f %f %f\n",tri2verts[2], abc[2][0],
            // abc[2][1], abc[2][2]);
            printf(
                "Error: Particle not in this "
                "element"
                "\tpid %d elem %d\n",
                pid, elmId);
            printf("bcc %f %f %f\n", bcc[0], bcc[1], bcc[2]);
            OMEGA_H_CHECK(false);
          }
        }
        // find the intersection point
        double intersection_distance = -1.0;
        int edge_id = -1;
        Omega_h::Vector<2> start_2d = {start[0], start[2]};
        Omega_h::Vector<2> dest_2d = {dest[0], dest[2]};
        for (int i = 0; i < 3; i++) {
          int j = (i + 1) % 3;
          double dist = find_intersection_distance_tri(
              start_2d, dest_2d, tri_verts[i], tri_verts[j]);
          if (dist >= 0) {
            intersection_distance = dist;
            /// printf("! Intersected: Intersection distance %f\n",
            /// intersection_distance);
            edge_id = i;
            break;
          }
        }
        // get intersetion edge
        if (intersection_distance > 0) {
          // printf("Intersection distance %f\n", intersection_distance);
          //  move the particle to the next element
          intersection_distance += tol;  // to move inside the next element
          // update the xtgt to the new position
          o::Vector<3> new_position_cyl = {xtgt(pid, 0), xtgt(pid, 1),
                                           xtgt(pid, 2)};
          auto new_position_cyl_cp =
              new_position_cyl;  // todo remove this after testing
          o::Vector<3> new_position_cart;
          cylindrical2cartesian(new_position_cyl, new_position_cart);
          // move the particle to the new position
          o::Vector<3> orig_position_cyl = {x(pid, 0), x(pid, 1), x(pid, 2)};
          // it's the new position but with the same theta
          o::Vector<3> new_position_cyl_cotheta = {
              new_position_cyl[0], orig_position_cyl[1], new_position_cyl[2]};
          o::Vector<3> new_position_cart_cotheta;
          o::Vector<3> orig_position_cart;
          cylindrical2cartesian(orig_position_cyl, orig_position_cart);
          cylindrical2cartesian(new_position_cyl_cotheta,
                                new_position_cart_cotheta);
          auto direction = (new_position_cart - orig_position_cart) /
                           o::norm(new_position_cart - orig_position_cart);
          // assert that the direction is unit vector
          OMEGA_H_CHECK(std::abs(o::norm(direction) - 1.0) <
                        1.0e-10);  // todo remove this check and the next
          auto rz_plane_direction =
              (new_position_cart_cotheta - orig_position_cart) /
              o::norm(new_position_cart_cotheta - orig_position_cart);
          // assert that the rz_plane_direction is unit vector
          OMEGA_H_CHECK(std::abs(o::norm(rz_plane_direction) - 1.0) < 1.0e-10);
          double dot_product = o::inner_product(direction, rz_plane_direction);
          double intersection_distance_3d = intersection_distance / dot_product;
          // OMEGA_H_CHECK(intersection_distance_3d > 0); // ? can it be
          // negative OMEGA_H_CHECK(intersection_distance_3d >=
          // intersection_distance); // ? can it be less than the intersection
          // distance
          new_position_cart =
              orig_position_cart + (direction * intersection_distance_3d);
          cartesian2cylindrical(new_position_cart, new_position_cyl);
          // check if the projection is correct
          // print the new position
          auto cyl_direction = (new_position_cyl_cp - orig_position_cyl) /
                               o::norm(new_position_cyl_cp - orig_position_cyl);
          double temp_R =
              orig_position_cyl[0] + (intersection_distance * cyl_direction[0]);
          double temp_Z =
              orig_position_cyl[2] + (intersection_distance * cyl_direction[2]);
          // printf("New position of pid %d, el %d:             %f %f %f\n",
          // pid,e, new_position_cyl[0], new_position_cyl[1],
          // new_position_cyl[2]); printf("New position of pid %d, el %d in RZ
          // plane: %f        %f\n", pid, e, temp_R, temp_Z);
          if (std::abs(temp_R - new_position_cyl[0]) > 1.0e-4 ||
              std::abs(temp_Z - new_position_cyl[2]) > 1.0e-5) {
            // printf("Error: The new position is not correct: diff = %f %f \n",
            // std::abs(temp_R - new_position_cyl[0]), std::abs(temp_Z -
            // new_position_cyl[2]));
          }

          // OMEGA_H_CHECK(std::abs(temp_R - new_position_cyl[0])<10e-4);
          // OMEGA_H_CHECK(std::abs(temp_Z  - new_position_cyl[2])<10e-4);

          xtgt(pid, 0) = new_position_cyl[0];
          xtgt(pid, 1) = new_position_cyl[1];
          xtgt(pid, 2) = new_position_cyl[2];
          Omega_h::Vector<3> new_position = {new_position_cyl[0], 0.0,
                                             new_position_cyl[2]};
          // get the next element: loop throught the adjaecnt faces current face
          // and check the bcc of the new position
          int n_adj_faces = face2faceOffsets[e + 1] - face2faceOffsets[e];

          // loop thought the faces and check if the bcc in the new position is
          // all positive
          bool found_next_face = false;
          for (int i = 0; i < n_adj_faces; i++) {
            int adj_face = face2faceFace[face2faceOffsets[e] + i];
            auto adj_tri2verts = o::gather_verts<3>(faces2nodes, adj_face);
            Omega_h::Few<Omega_h::Vector<2>, 3> adj_tri_verts;
            for (int j = 0; j < 3; ++j) {
              adj_tri_verts[j] =
                  Omega_h::get_vector<2>(coords, adj_tri2verts[j]);
            }
            o::Vector<3> bcc = o::barycentric_from_global<2, 2>(
                {new_position[0], new_position[2]}, adj_tri_verts);
            if (all_positive(bcc, tol)) {
              elem_ids_next[pid] = adj_face;
              found_next_face = true;
              // printf("PID %d, Element %d, Next Element %d\n", pid, e,
              // adj_face);
              //// print new position
              // printf("New position of pid %d, el %d: %f %f %f\n", pid, e,
              // new_position[0], new_position[1], new_position[2]);
              // printf("Dest elm vertex %d: %f %f %f\n", adj_tri2verts[0],
              // abc[0][0], abc[0][1], abc[0][2]); printf("Dest elm vertex %d:
              // %f %f %f\n", adj_tri2verts[1], abc[1][0], abc[1][1],
              // abc[1][2]); printf("Dest elm vertex %d: %f %f %f\n",
              // adj_tri2verts[2], abc[2][0], abc[2][1], abc[2][2]);
              break;
            }
          }

          if (!found_next_face && n_adj_faces == 3) {  // todo now make it lost
            // printf("Error: Particle next position not in the neighbour faces
            // of "
            //         "element %d, pid %d\n", e, pid);
            elem_ids_next[pid] = -1;
            // OMEGA_H_CHECK(false);
          }
          // if not found and number of adj_faces is less than 3, that means the
          // partice moved out of the boundary
          if (!found_next_face &&
              n_adj_faces < 3) {  // TODO better way to check out of boundary
            elem_ids_next[pid] = -1;  // out of boundary
          }

          // done with the particle
          ptcl_done[pid] = 1;

        } else {  // particle remained in the same element
          elem_ids_next[pid] = e;
          ptcl_done[pid] = 1;
        }
      }
    };
    parallel_for(ptcls, lamb, "adj_search");
    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA(o::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    o::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");
  }
  return found;
}

void search(p::Mesh& picparts, PS* ptcls, bool output) {
  o::Mesh* mesh = picparts.mesh();
  assert(ptcls->nElems() == mesh->nelems());
  Omega_h::LO maxLoops = 100;
  const auto psCapacity = ptcls->capacity();
  o::Write<o::LO> elem_ids(psCapacity, -1, "elem_ids");
  // printf("INFO: Size of elem_ids: %d\n", elem_ids.size());
  Kokkos::Timer timer;
  auto x = ptcls->get<0>();
  auto xtgt = ptcls->get<1>();
  auto pid = ptcls->get<2>();
  o::Write<o::Real> xpoints_d(3 * psCapacity, "intersection points");
  o::Write<o::LO> xface_id(psCapacity, "intersection faces");
  // all sides are considered zone boundaries
  o::Read<o::I8> zone_boundary_sides(mesh->nfaces(), 1);
  /*
  bool isFound = p::search_mesh_with_zone<Particle>(
      *mesh, ptcls, x, xtgt, pid, elem_ids, xpoints_d, xface_id,
      zone_boundary_sides, maxLoops);
  */
  bool isFound =
      search_adj_elements(*mesh, ptcls, x, xtgt, pid, elem_ids, xpoints_d,
                          xface_id, zone_boundary_sides, maxLoops);
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
  int tagged_ent = (mesh->dim() == 3) ? o::REGION : o::FACE;
  mesh->set_tag(tagged_ent, "has_particles", ehp_nm0_r);
}

void ownerFromCPN(const std::string cpn_file_name, o::Write<o::LO>& owners) {
  std::ifstream cpn_file(cpn_file_name);
  if (!cpn_file.is_open()) {
    std::cerr << "Error: cannot open the cpn file\n";
    exit(1);
  }
  int own;
  int index = 0;
  while (cpn_file >> own) {
    owners[index] = own;
    index++;
  }

  if (index != owners.size()) {
    std::cout << "******************* Warning *******************\n"
                 "The number of elements in the cpn file does not match the "
                 "number of elements in the mesh: owner size, cpn size: : "
              << owners.size() << " " << index - 1 << '\n';
  }
}

void partitionMeshEqually(o::Mesh& mesh, o::Write<o::LO> owners, int comm_size,
                          int comm_rank) {
  int dim = mesh.dim();
  o::Vector<2> min;
  o::Vector<2> max;
  get_bounding_box_in_xy_plane(mesh, min, max);
  // create a vector of 0 comm_size-1
  std::vector<int> ranks(comm_size);
  std::iota(ranks.begin(), ranks.end(), 0);
  // create a vector of cut surfaces for recursive coordinate bisection
  double startx = min[0];
  double starty = min[1];
  double deltax = max[0] - min[0];
  double deltay = max[1] - min[1];
  int ncuts_x, ncuts_y;
  create_int_rectangle(comm_size, ncuts_x, ncuts_y);

  // print the number of cuts in x and y
  std::cout << "Number of cuts in x and y direction: " << ncuts_x << " "
            << ncuts_y << '\n';

  double dx = deltax / ncuts_x;
  double dy = deltay / ncuts_y;

  o::LOs cells2nodes;
  if (dim == 3) {
    cells2nodes = mesh.ask_down(o::REGION, o::VERT).ab2b;
  } else if (dim == 2) {
    cells2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  } else {
    std::cerr << "Error: unsupported dimension\n";
    exit(1);
  }

  auto nodes2coords = mesh.coords();
  auto lamb = OMEGA_H_LAMBDA(o::LO i) {
    std::array<double, 2> center2d;
    if (dim == 3) {
      auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(i));
      auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
      auto center = average(cell_nodes2coords);
      center2d = {center[0], center[1]};
    } else if (dim == 2) {
      auto cell_nodes2nodes = o::gather_verts<3>(cells2nodes, o::LO(i));
      auto cell_nodes2coords =
          o::gather_vectors<3, 2>(nodes2coords, cell_nodes2nodes);
      auto center = average(cell_nodes2coords);
      center2d = {center[0], center[1]};
    }

    // std::array<double, 3> center_arr = {center[0], center[1], center[2]};

    // assert that the center is within the bounding box
    if (center2d[0] < startx || center2d[0] > max[0] || center2d[1] < starty ||
        center2d[1] > max[1]) {
      printf("Error: center is outside the bounding box\n");
      printf("Center: %f %f\n", center2d[0], center2d[1]);
      printf("Bounding box: %f %f %f %f and dim = %d\n", startx, max[0], starty,
             max[1], dim);
      exit(1);
    }
    // get rank based on the location of the center2d starting from the
    // lower-left corner and moving up and right
    int rank = int((center2d[0] - startx) / dx) +
               ncuts_x * int((center2d[1] - starty) / dy);
    if (rank >= comm_size || rank < 0) {
      printf("Error: rank is out of bounds\n");
      printf("Rank: %d comm_size: %d\n", rank, comm_size);
      exit(1);
    }
    owners[i] = rank;
  };
  o::parallel_for(mesh.nelems(), lamb);
  if (comm_rank == 0) {
    varify_balance_of_partitions(owners, comm_size);
  }
}

void get_bounding_box_in_xy_plane(Omega_h::Mesh& mesh, o::Vector<2>& min,
                                  o::Vector<2>& max) {
  int dim = mesh.dim();
  if (dim == 2) {
    Omega_h::BBox<2> bb = Omega_h::get_bounding_box<2>(&mesh);
    min = bb.min;
    max = bb.max;
  } else if (dim == 3) {
    Omega_h::BBox<3> bb = Omega_h::get_bounding_box<3>(&mesh);
    min = {bb.min[0], bb.min[1]};
    max = {bb.max[0], bb.max[1]};
  } else {
    std::cerr << "Error: unsupported dimension\n";
    exit(1);
  }
#ifdef DEBUG
  prettyPrintBB(min, max);
#endif
}

void create_int_rectangle(const int total, int& nrows, int& ncols) {
  nrows = std::sqrt(total);
  while (total % nrows != 0) {
    nrows--;
  }
  ncols = total / nrows;
}

void varify_balance_of_partitions(o::Write<o::LO>& owners, int comm_size) {
  std::vector<int> counts(comm_size, 0);
  for (int i = 0; i < owners.size(); i++) {
    counts[owners[i]]++;
  }

  int fair_amount = owners.size() / comm_size;

  for (int i = 0; i < comm_size; i++) {
    // printf("Rank %d has %d elements\n", i, counts[i]);
    if (counts[i] < fair_amount * 0.05 || counts[1] > fair_amount * 2) {
      printf("Error: Rank %d has %d elements out of %d\n", i, counts[i],
             owners.size());
      exit(1);
    } else if (counts[i] < fair_amount * 0.5 || counts[1] > fair_amount * 2) {
      printf("Warning: Rank %d has %d elements out of %d\n", i, counts[i],
             owners.size());
    }
  }
}

template <typename T>
void prettyPrintBB(T min, T max) {
  printf("\nBounding box: \n");
  printf("\t\t\t----------------------(%8.4f, %8.4f)\n", max[0], max[1]);
  printf("\t\t\t|                    |\n");
  printf("\t\t\t|                    |\n");
  printf("\t\t\t|                    |\n");
  printf("\t\t\t|                    |\n");
  printf("\t\t\t|                    |\n");
  printf("(%8.4f, %8.4f)\t---------------------|\n", min[0], min[1]);
}

void get_tet_centroid(const o::LOs& cells2nodes, o::LO e,
                      const o::Reals& nodes2coords,
                      o::Few<o::Real, 3>& center) {
  auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
  auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
  center = average(cell_nodes2coords);
}

void get_tri_centroid(const o::LOs& cells2nodes, o::LO e,
                      const o::Reals& nodes2coords,
                      o::Few<o::Real, 2>& center) {
  auto cell_nodes2nodes = o::gather_verts<3>(cells2nodes, o::LO(e));
  auto cell_nodes2coords =
      o::gather_vectors<3, 2>(nodes2coords, cell_nodes2nodes);
  center = average(cell_nodes2coords);
}

OMEGA_H_DEVICE void cylindrical2cartesian(const o::Vector<3> cyl,
                                          o::Vector<3>& cartesian) {
  OMEGA_H_CHECK(cyl.size() == 3);

  cartesian[0] = cyl[0] * std::cos(cyl[1]);
  cartesian[1] = cyl[0] * std::sin(cyl[1]);
  cartesian[2] = cyl[2];
}

OMEGA_H_DEVICE void cartesian2cylindrical(const o::Vector<3> cartesian,
                                          o::Vector<3>& cyl) {
  cyl[0] = std::sqrt(cartesian[0] * cartesian[0] + cartesian[1] * cartesian[1]);
  cyl[1] = std::atan2(cartesian[1], cartesian[0]);
  cyl[2] = cartesian[2];
}