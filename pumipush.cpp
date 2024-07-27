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
#include <optional>
#include <particle_structure.hpp>
#include <pumipic_kktypes.hpp>

std::random_device rd;
auto seed = 0;
std::mt19937 gen(seed);
std::uniform_real_distribution<> dis(0, 1);

inline double random_path_length(double lambda) {
  // ref:
  // https://docs.openmc.org/en/stable/methods/neutron_physics.html#sampling-distance-to-next-collision

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
  // std::random_device rd;
  // std::mt19937 gen(0);
  // std::uniform_real_distribution<> dis(0, 1);
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

OMEGA_H_DEVICE std::optional<o::Vector<2>> find_intersection_point(
    o::Few<o::Vector<2>, 2> line1, o::Few<o::Vector<2>, 2> line2) {
  auto b = line2[0] - line1[0];
  o::Matrix<2, 2> A;
  A[0] = line1[1] - line1[0];
  A[1] = line2[0] - line2[1];
  // A = o::transpose(A);
  //  print the matrix
  // printf("Matrix A: \n");
  // for (int i = 0; i < 2; i++){
  //   for (int j = 0; j < 2; j++){
  //     printf("%f ", A[i][j]);
  //   }
  //   printf("\n");
  // }
  auto det = o::determinant(A);
  if (std::abs(det) < 10e-10) {
    return {};
  }
  o::Vector<2> x = o::invert(A) * b;
  if (x[0] < 0 || x[0] > 1 || x[1] < 0 || x[1] > 1) {
    return {};
  }
  o::Vector<2> intersection_point = (1 - x[0]) * line1[0] + x[0] * line1[1];
  return intersection_point;
}

OMEGA_H_DEVICE double distance_between_points(o::Vector<2> p1,
                                              o::Vector<2> p2) {
  return o::norm(p1 - p2);
}

/// ref: https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
OMEGA_H_DEVICE double find_intersection_distance_tri(
    const Omega_h::Few<Omega_h::Vector<2>, 2>& start_dest,
    const o::Few<o::Vector<2>, 2>& tri_edge) {
  // test_intersection();
  if (auto intersection_point = find_intersection_point(start_dest, tri_edge)) {
    return distance_between_points(start_dest[0], intersection_point.value());
  } else {
    return -1.0;
  }
}

OMEGA_H_DEVICE void search_through_mesh(const o::Mesh& mesh, o::Vector<3> x) {
  o::Vector<2> x_rz = {x[0], x[2]};
  auto coords = mesh.coords();
  auto n_faces = mesh.nfaces();
  auto face2nodes = mesh.get_adj(o::FACE, o::VERT).ab2b;

  for (int i = 0; i < n_faces; i++) {
    auto face_nodes = o::gather_verts<3>(face2nodes, i);
    Omega_h::Few<Omega_h::Vector<2>, 3> face_coords;
    face_coords = o::gather_vectors<3, 2>(coords, face_nodes);
    auto bcc = o::barycentric_from_global<2, 2>(x_rz, face_coords);
    if (all_positive(bcc)) {
      printf("Found the particle in element: %d\n", i);
      printf("Barycentric coordinates: %f %f %f\n", bcc[0], bcc[1], bcc[2]);
      printf("Coordinates of the face: \n");
      for (int j = 0; j < 3; j++) {
        printf("%.16f %.16f\n", face_coords[j][0], face_coords[j][1]);
      }
      printf("Position of the particle was: %.16f %.16f\n", x_rz[0], x_rz[1]);
    }
  }
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
  o::Real tol = 1.0e-10;
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
        o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};
        const o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        o::Vector<3> bcc;
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);

        {  // check if the particle is in the element
          bcc = o::barycentric_from_global<2, 2>(
              {origin_rThz[0], origin_rThz[2]}, current_el_vert_coords);
          if (!all_positive(bcc)) {
            printf(
                "Error: Particle not in this "
                "element"
                "\tpid %d elem %d\n",
                pid, elmId);
            printf("bcc %.16f %.16f %.16f\n", bcc[0], bcc[1], bcc[2]);
            printf("Position of the element: %.16f %.16f\n", origin_rThz[0],
                   origin_rThz[2]);
            printf("Face vertex ids: %d %d %d\n", current_el_verts[0],
                   current_el_verts[1], current_el_verts[2]);
            // grab the location and find out the element
            search_through_mesh(mesh, origin_rThz);
            OMEGA_H_CHECK(false);
          }
        }

        bool remains_in_el = false;
        {
          // check if the destination is in the element
          bcc = o::barycentric_from_global<2, 2>({dest_rThz[0], dest_rThz[2]},
                                                 current_el_vert_coords);
          remains_in_el = all_positive(bcc);
        }

        // find the intersection distance
        double intersection_distance = -1.0;
        int edge_id = -1;
        Omega_h::Vector<2> origin_rz = {origin_rThz[0], origin_rThz[2]};
        Omega_h::Vector<2> dest_rz = {dest_rThz[0], dest_rThz[2]};
        for (int i = 0; i < 3; i++) {
          int j = (i + 1) % 3;
          intersection_distance = find_intersection_distance_tri(
              {origin_rz, dest_rz},
              {current_el_vert_coords[i], current_el_vert_coords[j]});
          if (intersection_distance > 0) {
            /// printf("! Intersected: Intersection distance %f\n",
            /// intersection_distance);
            edge_id = i;
            break;
          }
        }

        // if the particle remains in the element, and didn't intersect any edge
        // then it's done
        if (remains_in_el &&
            intersection_distance < 0) {  // particle remained in the same
                                          // element: checked previously
          // with bcc and intersection distance
          elem_ids_next[pid] = e;
          ptcl_done[pid] = 1;
        }

        // get intersetion edge: if it intersected or didn't remain in the same
        // element (starts on edge)
        if ((intersection_distance > 0 && edge_id != -1) ||
            (intersection_distance < 0 && !remains_in_el)) {
          // printf("Intersection distance %f\n", intersection_distance);

          intersection_distance += tol;  // to move inside the next element
          o::Vector<3> new_position_cyl_cp = {xtgt(pid, 0), xtgt(pid, 1),
                                              xtgt(pid, 2)};
          o::Vector<3> dest_xyz;
          cylindrical2cartesian(dest_rThz, dest_xyz);
          // it's the new position but with the same theta
          o::Vector<3> dest_rThz_Th_plane = {dest_rThz[0], origin_rThz[1],
                                             dest_rThz[2]};
          o::Vector<3> dest_xyz_Th_plane;
          o::Vector<3> origin_xyz;
          cylindrical2cartesian(origin_rThz, origin_xyz);
          cylindrical2cartesian(dest_rThz_Th_plane, dest_xyz_Th_plane);
          auto direction_xyz =
              (dest_xyz - origin_xyz) / o::norm(dest_xyz - origin_xyz);
#ifdef DEBUG
          // assert that the direction is unit vector
          OMEGA_H_CHECK(std::abs(o::norm(direction_xyz) - 1.0) <
                        1.0e-10);  // todo remove this check and the next
#endif
          auto direction_Th_plane = (dest_xyz_Th_plane - origin_xyz) /
                                    o::norm(dest_xyz_Th_plane - origin_xyz);
#ifdef DEBUG
          OMEGA_H_CHECK(std::abs(o::norm(direction_Th_plane) - 1.0) < 1.0e-10);
#endif
          double dot_product =
              o::inner_product(direction_xyz, direction_Th_plane);
          double intersection_distance_xyz =
              intersection_distance / dot_product;
          // OMEGA_H_CHECK(intersection_distance_3d > 0); // ? can it be
          // negative OMEGA_H_CHECK(intersection_distance_3d >=
          // intersection_distance); // ? can it be less than the intersection
          // distance
          dest_xyz = origin_xyz + (direction_xyz * intersection_distance_xyz);
          cartesian2cylindrical(dest_xyz, dest_rThz);
          // check if the projection is correct
          // print the new position
          auto cyl_direction = (new_position_cyl_cp - origin_rThz) /
                               o::norm(new_position_cyl_cp - origin_rThz);
          double temp_R =
              origin_rThz[0] + (intersection_distance * cyl_direction[0]);
          double temp_Z =
              origin_rThz[2] + (intersection_distance * cyl_direction[2]);
          // printf("New position of pid %d, el %d:             %f %f %f\n",
          // pid,e, new_position_cyl[0], new_position_cyl[1],
          // new_position_cyl[2]); printf("New position of pid %d, el %d in RZ
          // plane: %f        %f\n", pid, e, temp_R, temp_Z);
          if (std::abs(temp_R - dest_rThz[0]) > 1.0e-4 ||
              std::abs(temp_Z - dest_rThz[2]) > 1.0e-4) {  // todo remove this
            // printf("Error: The new position is not correct: diff = %f %f \n",
            // std::abs(temp_R - new_position_cyl[0]), std::abs(temp_Z -
            // new_position_cyl[2]));
          }

          // OMEGA_H_CHECK(std::abs(temp_R - new_position_cyl[0])<10e-4);
          // OMEGA_H_CHECK(std::abs(temp_Z  - new_position_cyl[2])<10e-4);

          xtgt(pid, 0) = dest_rThz[0];
          xtgt(pid, 1) = dest_rThz[1];
          xtgt(pid, 2) = dest_rThz[2];
          dest_rz = {dest_rThz[0], dest_rThz[2]};
          // get the next element: loop throught the adjaecnt faces current face
          // and check the bcc of the new position
          int n_adj_faces = face2faceOffsets[e + 1] - face2faceOffsets[e];

          // loop thought the faces and check if the bcc in the new position is
          // all positive
          bool found_next_face = false;
          for (int i = 0; i < n_adj_faces; i++) {  // edge adj faces
            int adj_face = face2faceFace[face2faceOffsets[e] + i];
            auto adj_face_verts = o::gather_verts<3>(faces2nodes, adj_face);
            Omega_h::Few<Omega_h::Vector<2>, 3> adj_face_vert_coords =
                o::gather_vectors<3, 2>(coords, adj_face_verts);
            o::Vector<3> bcc =
                o::barycentric_from_global<2, 2>(dest_rz, adj_face_vert_coords);
            if (all_positive(bcc, tol)) {
              elem_ids_next[pid] = adj_face;
              found_next_face = true;
              break;
            }
          }

          if (!found_next_face) {  // node adj faces
            // printf("*");
            for (int node = 0; node < 3; node++) {
              int cur_vert = current_el_verts[node];
              int n_node_adj_face =
                  node2faceOffsets[cur_vert + 1] - node2faceOffsets[cur_vert];
              for (int i = 0; i < n_node_adj_face; i++) {
                auto node_adj_face =
                    node2faceFace[node2faceOffsets[cur_vert] + i];
                auto face_verts =
                    o::gather_verts<3>(faces2nodes, node_adj_face);
                auto face_vert_coords =
                    o::gather_vectors<3, 2>(coords, face_verts);
                o::Vector<3> bcc =
                    o::barycentric_from_global<2, 2>(dest_rz, face_vert_coords);
                if (all_positive(bcc, tol)) {
                  // printf("Face Vertices: %d %d %d\n", face_verts[0],
                  //        face_verts[1], face_verts[2]);
                  elem_ids_next[pid] = node_adj_face;
                  found_next_face = true;
                  break;
                }
              }
            }
            if (!found_next_face) {  // todo handle this case
              elem_ids_next[pid] = -1;
              // printf(".");
            }
            // elem_ids_next[pid] = -1;
            // printf(".");
            //  OMEGA_H_CHECK(false);
          }
          // if not found and number of adj_faces is less than 3, that means the
          // partice moved out of the boundary
          if (!found_next_face &&
              n_adj_faces < 3) {  // TODO better way to check out of boundary
            elem_ids_next[pid] = -1;  // out of boundary
          }

          // done with the particle
          ptcl_done[pid] = 1;

        }  // if intersected or not in the same element
      }  // search each particle
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

OMEGA_H_DEVICE void get_tet_centroid(const o::LOs& cells2nodes, o::LO e,
                                     const o::Reals& nodes2coords,
                                     o::Few<o::Real, 3>& center) {
  auto cell_nodes2nodes = o::gather_verts<4>(cells2nodes, o::LO(e));
  auto cell_nodes2coords = gatherVectors(nodes2coords, cell_nodes2nodes);
  center = average(cell_nodes2coords);
}

OMEGA_H_DEVICE void get_tri_centroid(const o::LOs& cells2nodes, o::LO e,
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