#include "pumipush.h"

#include <ppMacros.h>

Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace> TeamPolicyAutoSelect(
    int league_size, int team_size) {
#ifdef PP_USE_GPU
  return Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(league_size,
                                                           team_size);
#else
  return Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>(league_size,
                                                           Kokkos::AUTO());
#endif
}

o::Mesh readMesh(std::string meshFile, o::Library& lib) {
  (void)lib;
  std::string fn(meshFile);
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

int distributeParticlesBasesOnArea(const p::Mesh& picparts, PS::kkLidView ppe,
                                   const int numPtcls) {
  o::Mesh* mesh = picparts.mesh();
  o::LO ne = mesh->nelems();
  o::Real mesh_area = area_of_2d_mesh(*mesh);
  OMEGA_H_CHECK(mesh_area > 0.0);

  auto coords = mesh->coords();
  auto face2nodes = mesh->ask_down(o::FACE, o::VERT).ab2b;

  auto distribute_based_on_area = OMEGA_H_LAMBDA(o::LO e) {
    auto verts = o::gather_verts<3>(face2nodes, e);
    auto vert_coords = o::gather_vectors<3, 2>(coords, verts);
    o::Real area = area_tri(vert_coords);
#ifdef DEBUG
    OMEGA_H_CHECK(area > 0.0);
#endif
    o::Real area_fraction = area / mesh_area;
    ppe[e] = std::round(numPtcls * area_fraction);
  };
  o::parallel_for(ne, distribute_based_on_area);

  Omega_h::LO totPtcls = 0;
  Kokkos::parallel_reduce(
      ppe.size(),
      OMEGA_H_LAMBDA(const int i, Omega_h::LO& lsum) { lsum += ppe[i]; },
      totPtcls);
  // assert that the difference not more than 0.01% of the total number of
  // particles
  o::Real discrepenacy = 100. * (std::abs(totPtcls - numPtcls) / numPtcls);
  bool high_discrepancy = discrepenacy > 0.1;
  if (high_discrepancy) {
    printf("Error: High discrepancy (%f %) in particle distribution\n",
           discrepenacy);
    printf("Intended particles %d, distributed particles %d\n", numPtcls,
           totPtcls);
  }
  OMEGA_H_CHECK(!high_discrepancy);
  return totPtcls;
}

int distributeParticlesEqually(const p::Mesh& picparts, PS::kkLidView ppe,
                               const int numPtcls) {
  auto ne = picparts.mesh()->nelems();
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

void setUniformPtclCoords(p::Mesh& picparts, PS* ptcls,
                          random_pool_t random_pool) {
  o::Mesh* mesh = picparts.mesh();
  int dim = mesh->dim();
  OMEGA_H_CHECK(dim == 2);

  auto cells2nodes = mesh->ask_down(o::FACE, o::VERT).ab2b;
  auto coords = mesh->coords();

  auto x_ps_d = ptcls->get<0>();

  auto set_initial_positions =
      PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      auto gen = random_pool.get_state();
      o::Vector<3> random_bcc{0.0, 0.0, 0.0};
      random_bcc[0] = gen.drand(0.0, 1.0);
      random_bcc[1] = gen.drand(0.0, 1.0);
      o::Real complimentary0 = 1.0 - random_bcc[0];
      o::Real complimentary1 = 1.0 - random_bcc[1];
      random_bcc[0] = (random_bcc[0] + random_bcc[1] > 1.0) ? complimentary0
                                                            : random_bcc[0];
      random_bcc[1] = (random_bcc[0] + random_bcc[1] > 1.0) ? complimentary1
                                                            : random_bcc[1];
      random_bcc[2] = 1.0 - random_bcc[0] - random_bcc[1];
      o::Real random_theta = gen.drand(-M_PI, M_PI);
      random_pool.free_state(gen);

      auto verts = o::gather_verts<3>(cells2nodes, e);
      auto vert_coords = o::gather_vectors<3, 2>(coords, verts);

      auto real_loc = barycentric2real(vert_coords, random_bcc);
      x_ps_d(pid, 0) = real_loc[0];
      x_ps_d(pid, 1) = random_theta;
      x_ps_d(pid, 2) = real_loc[1];
    }
  };
  ps::parallel_for(ptcls, set_initial_positions);
}

void setInitialPtclCoords(p::Mesh& picparts, PS* ptcls,
                          random_pool_t random_pool) {
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
  auto x_ps_d = ptcls->get<0>();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      if (dim == 2) {
        o::Few<o::Real, 2> center;
        get_tri_centroid(cells2nodes, e, nodes2coords, center);
        // random theta from -pi to pi
        auto gen = random_pool.get_state();
        double rn = gen.drand(-1., 1.);
        double random_theta = rn * M_PI;
        random_pool.free_state(gen);
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
        printf("Error: unsupported dimension\n");
        OMEGA_H_CHECK(false);
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

void pseudo2Dpush(PS* ptcls, double lambda, random_pool_t pool) {
  Kokkos::Timer timer;
  auto position_d = ptcls->get<0>();
  auto new_position_d = ptcls->get<1>();

  double totTime = 0;
  timer.reset();
  // random_pool_t pool(34973947);
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      double distance = random_path_length(lambda, pool);
      // printf("Push distance for particle %d: %f\n", pid, distance);
      o::Vector<3> cyl_coords = {position_d(pid, 0), position_d(pid, 1),
                                 position_d(pid, 2)};
      o::Vector<3> cart_coords;
      cylindrical2cartesian(cyl_coords, cart_coords);

      o::Vector<3> direction_vec = sampleRandomDirection(1, pool);
      o::Vector<3> new_position_cart = cart_coords + (distance * direction_vec);
      cartesian2cylindrical(new_position_cart, cyl_coords);
      // printf("INFO: Old position: %.16f %.16f %.16f\n", position_d(pid, 0),
      //        position_d(pid, 1), position_d(pid, 2));
      // printf("INFO: New position: %.16f %.16f %.16f\n", cyl_coords[0],
      //        cyl_coords[1], cyl_coords[2]);
      new_position_d(pid, 0) = cyl_coords[0];
      new_position_d(pid, 1) = cyl_coords[1];
      new_position_d(pid, 2) = cyl_coords[2];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
}

void push(PS* ptcls, int np, double lambda, random_pool_t pool) {
  Kokkos::Timer timer;
  auto position_d = ptcls->get<0>();
  auto new_position_d = ptcls->get<1>();

  double totTime = 0;
  // random_pool_t pool(1937493);
  timer.reset();
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask) {
      // fp_t dir[3];
      double distance = random_path_length(lambda, pool);
      o::Vector<3> disp_d = sampleRandomDirection(1, pool);
      new_position_d(pid, 0) = position_d(pid, 0) + distance * disp_d[0];
      new_position_d(pid, 1) = position_d(pid, 1) + distance * disp_d[1];
      new_position_d(pid, 2) = position_d(pid, 2) + distance * disp_d[2];
    }
  };
  ps::parallel_for(ptcls, lamb);

  totTime += timer.seconds();
  printTiming("ps push", totTime);
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

bool search_adj_elements(o::Mesh& mesh, PS* ptcls,
                         p::Segment<double[3], Kokkos::CudaSpace> x,
                         p::Segment<double[3], Kokkos::CudaSpace> xtgt,
                         p::Segment<int, Kokkos::CudaSpace> pid,
                         o::Write<o::LO> elem_ids, o::Write<o::Real> xpoints_d,
                         o::Write<o::LO> xface_id,
                         o::Read<o::I8> zone_boundary_sides,
                         int looplimit = 10) {
  OMEGA_H_CHECK(mesh.dim() == 2);  // only for pseudo3D now
  const auto elemArea = o::measure_elements_real(&mesh);
  o::Real tol = p::compute_tolerance_from_area(elemArea);
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  const auto coords = mesh.coords();
  const auto faces2nodes = mesh.ask_verts_of(o::FACE);
  const auto face2faceFace = mesh.ask_dual().ab2b;
  const auto face2faceOffsets = mesh.ask_dual().a2ab;
  const auto face2nodeNode =
      mesh.ask_down(o::FACE, o::VERT).ab2b;  // always 3; offset not needed
  const auto node2faceFace = mesh.ask_up(o::VERT, o::FACE).ab2b;
  const auto node2faceOffsets = mesh.ask_up(o::VERT, o::FACE).a2ab;
  const auto face2edgeEdge = mesh.ask_down(o::FACE, o::EDGE).ab2b;
  const auto psCapacity = ptcls->capacity();

  const auto exposed_edges = o::mark_exposed_sides(&mesh);

  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_done");
  o::Write<o::LO> elem_ids_next(psCapacity, -1, "elem_ids_next");

  auto fill = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      elem_ids[pid] = e;
      printf("INFO: Particle %d is in element %d\n", pid, e);
    } else {
      ptcl_done[pid] = 1;
    }
  };
  parallel_for(ptcls, fill, "searchMesh_fill_elem_ids");
  printf("INFO: Starting search for the particles\n");

  bool found = false;

  {  // original search in pumipush does it in a loop but here it is searched
     // only once since it only moves to the adj faces
    o::Write<o::LO> remains_in_el(psCapacity, 0, "remains_in_el");
    o::Write<o::LO> particle_on_edge(psCapacity, 0, "particle_on_edge");
    auto check_initial_position =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        auto elmId = elem_ids[pid];
        OMEGA_H_CHECK(elmId >= 0);
        // get the corners of the element
        o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};
        const o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        printf("INFO: Checking particle %d in element %d\n", pid, elmId);
        printf("INFO: Origin position: %.16f %.16f %.16f\n", origin_rThz[0],
               origin_rThz[1], origin_rThz[2]);
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        // check if the particle is in the element
        o::Vector<3> bcc = o::barycentric_from_global<2, 2>(
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
          // search_through_mesh(mesh, origin_rThz);
          OMEGA_H_CHECK(false);
        }
        // check remains in the element
        bcc = o::barycentric_from_global<2, 2>({dest_rThz[0], dest_rThz[2]},
                                               current_el_vert_coords);
        if (all_positive(bcc, 0.0)) {
          printf("INFO: Particle %d remains in element %d\n", pid, elmId);
        } else {
          printf("INFO: Particle %d does not remain in element %d\n", pid,
                 elmId);
        }
        remains_in_el[pid] = int(all_positive(bcc));
      }
    };
    parallel_for(ptcls, check_initial_position, "check_initial_postion");

    // find the intersection distance
    o::Write<o::Real> intersection_distance(psCapacity, -1.0,
                                            "intersection_distance");

    auto get_intersection_distance =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const o::Few<o::LO, 3> current_el_edges = {face2edgeEdge[3 * e],
                                                   face2edgeEdge[3 * e + 1],
                                                   face2edgeEdge[3 * e + 2]};
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        const o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1),
                                        xtgt(pid, 2)};
        const o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        const o::Vector<2> origin_rz = {origin_rThz[0], origin_rThz[2]};
        const o::Vector<2> dest_rz = {dest_rThz[0], dest_rThz[2]};
        const o::Vector<2> direction_rz =
            (dest_rz - origin_rz) / norm(dest_rz - origin_rz);

        auto bcc =
            o::barycentric_from_global<2, 2>(dest_rz, current_el_vert_coords);
        o::Few<o::LO, 3> potential_intersected_edges{-1, -1, -1};
        for (int i = 0; i < 3; i++) {
          potential_intersected_edges[i] =
              (bcc[i] < 0.0) ? current_el_edges[i] : -1;
        }
        printf("INFO: Current element edges: %d %d %d\n", current_el_edges[0],
               current_el_edges[1], current_el_edges[2]);
        printf("INFO: Potential intersected edges: %d %d %d\n",
               potential_intersected_edges[0], potential_intersected_edges[1],
               potential_intersected_edges[2]);

        o::LO intersected_edge = -1;
        o::Vector<2> intersection_point = {0.0, 0.0};
        for (int edge = 0; edge < 3; edge++) {
          auto intersection_result = find_intersection_point(
              {origin_rz, dest_rz}, {current_el_vert_coords[edge],
                                     current_el_vert_coords[(edge + 1) % 3]});
          bool intersected = intersection_result.exists;
          intersected_edge =
              (intersected) ? current_el_edges[edge] : intersected_edge;
          intersection_point =
              (intersected) ? intersection_result.point : intersection_point;
        }
        printf("INFO: Particle %d intersects at %.16f %.16f\n", pid,
               intersection_point[0], intersection_point[1]);

        o::Real cur_int_dist = -1.0;
        for (int i = 0; i < 3; i++) {  // todo expand the loop and if
          int j = (i + 1) % 3;
          printf("INFO: Checking edge %d %d\n", i, j);
          printf("INFO: Edge coords %.16f %.16f %.16f %.16f\n",
                 current_el_vert_coords[i][0], current_el_vert_coords[i][1],
                 current_el_vert_coords[j][0], current_el_vert_coords[j][1]);
          cur_int_dist = find_intersection_distance_tri(
              {origin_rz, dest_rz},
              {current_el_vert_coords[i], current_el_vert_coords[j]});
          auto intersection_point = find_intersection_point(
              {origin_rz, dest_rz},
              {current_el_vert_coords[i], current_el_vert_coords[j]});
          o::Real destination_theta = get_dest_theta(origin_rThz, dest_rThz);
          bool intersected = intersection_point.exists;
          if (intersected) {
            printf("INFO: Particle %d intersects at %.16f %.16f\n", pid,
                   intersection_point.point[0], intersection_point.point[1]);
          }
          xtgt(pid, 1) = (intersected) ? destination_theta : xtgt(pid, 1);
          intersection_point.point[0] =
              intersection_point.point[0] + direction_rz[0] * tol;
          intersection_point.point[1] =
              intersection_point.point[1] + direction_rz[1] * tol;
          xtgt(pid, 0) =
              (intersected) ? intersection_point.point[0] : xtgt(pid, 0);
          xtgt(pid, 2) =
              (intersected) ? intersection_point.point[1] : xtgt(pid, 2);
          if (intersection_point.exists) {
            printf("INFO: Intersection point: %.16f %.16f\n",
                   intersection_point.point[0], intersection_point.point[1]);
          }
          o::Real temp_dist = cur_int_dist + tol;
          intersection_distance[pid] =
              (cur_int_dist > -tol) ? temp_dist : intersection_distance[pid];
        }
        printf("INFO: Particle %d intersection distance: %f\n", pid,
               intersection_distance[pid]);
      }
    };
    parallel_for(ptcls, get_intersection_distance, "get_intersection_distance");

    // if the particle remains in the element, and didn't intersect any edge
    // then it's done
    auto if_in_same_el =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        o::Real cur_int_dist = intersection_distance[pid];
        printf("INFO: Particle %d intersection distance: %f\n", pid,
               cur_int_dist);
        o::LO cur_remain_in_el = remains_in_el[pid];
        bool didntMove = cur_remain_in_el && cur_int_dist < 0;
        elem_ids_next[pid] = (didntMove) ? e : elem_ids_next[pid];
        ptcl_done[pid] = (didntMove) ? 1 : ptcl_done[pid];
        if (didntMove) {
          printf("INFO: Particle %d remains in element %d\n", pid, e);
        }
      }
    };
    parallel_for(ptcls, if_in_same_el, "if_in_same_el");

    // auto update_dest_position =
    //     PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    //   if (mask > 0 && ptcl_done[pid] == 0) {
    //     o::Real cur_int_dist = intersection_distance[pid];
    //     printf("INFO: Particle %d intersection distance: %f\n", pid,
    //            cur_int_dist);
    //     o::LO cur_remain_in_el = remains_in_el[pid];
    //     OMEGA_H_CHECK(cur_int_dist > -1.0 && cur_remain_in_el == 0);
    //     // if (cur_int_dist > 0 && !cur_remain_in_el) {
    //     //  printf("Intersection distance %f\n", cur_int_dist);
    //
    //    cur_int_dist += tol;  // to move inside the next element
    //    intersection_distance[pid] = cur_int_dist;
    //    o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};
    //    o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
    //    o::Vector<3> modified_rThz = move_particle_accross_boundary(pid,
    //    origin_rThz, dest_rThz, cur_int_dist);
    //
    //    xtgt(pid, 0) = modified_rThz[0];
    //    xtgt(pid, 1) = modified_rThz[1];
    //    xtgt(pid, 2) = modified_rThz[2];
    //    //}  // if intersected or not in the same element
    //  }  // if mask
    //};  // update_dest_position lambda
    // parallel_for(ptcls, update_dest_position, "update_dest_position");
    // printf("INFO: Finished updating the destination position on rank %d\n",
    //       mesh.comm()->rank());

    auto find_next_element_in_ajd_faces =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        // o::Real cur_int_dist = intersection_distance[pid];
        o::LO cur_remain_in_el = remains_in_el[pid];

        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        o::Vector<2> dest_rz = {xtgt(pid, 0), xtgt(pid, 2)};
        o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};

        int n_adj_faces = face2faceOffsets[e + 1] - face2faceOffsets[e];

        for (int i = 0; i < n_adj_faces; i++) {  // edge adj faces
          int adj_face = face2faceFace[face2faceOffsets[e] + i];
          auto adj_face_verts = o::gather_verts<3>(faces2nodes, adj_face);
          Omega_h::Few<Omega_h::Vector<2>, 3> adj_face_vert_coords =
              o::gather_vectors<3, 2>(coords, adj_face_verts);
          o::Vector<3> bcc =
              o::barycentric_from_global<2, 2>(dest_rz, adj_face_vert_coords);
          elem_ids_next[pid] =
              (all_positive(bcc, tol)) ? adj_face : elem_ids_next[pid];
          ptcl_done[pid] = (all_positive(bcc, tol)) ? 1 : ptcl_done[pid];
          if (all_positive(bcc, tol)) {
            printf(
                "INFO: (Adj faces) Particle %d in element %d moves to element "
                "%d\n",
                pid, e, adj_face);
          }
        }
      }
    };
    parallel_for(ptcls, find_next_element_in_ajd_faces,
                 "find_next_element_in_ajd_faces");

    auto find_next_element_in_node_adj_faces =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        o::Real cur_int_dist = intersection_distance[pid];
        o::LO cur_remain_in_el = remains_in_el[pid];
        // if (cur_int_dist > 0 && !cur_remain_in_el) {
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        o::Vector<2> dest_rz = {xtgt(pid, 0), xtgt(pid, 2)};
        // o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        // o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};

        int n_adj_faces = face2faceOffsets[e + 1] - face2faceOffsets[e];
        for (int node = 0; node < 3; node++) {
          int cur_vert = current_el_verts[node];
          int n_node_adj_face =
              node2faceOffsets[cur_vert + 1] - node2faceOffsets[cur_vert];
          for (int i = 0; i < n_node_adj_face; i++) {
            auto node_adj_face = node2faceFace[node2faceOffsets[cur_vert] + i];
            auto face_verts = o::gather_verts<3>(faces2nodes, node_adj_face);
            auto face_vert_coords = o::gather_vectors<3, 2>(coords, face_verts);
            o::Vector<3> bcc =
                o::barycentric_from_global<2, 2>(dest_rz, face_vert_coords);
            bool found_next_el = all_positive(bcc, EPSILON);
            bool valid_next_el = node_adj_face != e;
            OMEGA_H_CHECK(!(found_next_el && !valid_next_el));
            elem_ids_next[pid] = (found_next_el && valid_next_el)
                                     ? node_adj_face
                                     : elem_ids_next[pid];
            ptcl_done[pid] =
                (found_next_el && valid_next_el) ? 1 : ptcl_done[pid];
            if (found_next_el && valid_next_el) {
              printf(
                  "INFO: (Node adj faces) Particle %d in element %d moves to "
                  "element %d\n",
                  pid, e, node_adj_face);
            }
          }
        }
        bool leaked = (n_adj_faces < 3 && ptcl_done[pid] == 0);
        if (leaked) {
          printf("INFO: Particle %d leaked from element %d\n", pid, e);
        }
        elem_ids_next[pid] =
            (leaked) ? -1 : elem_ids_next[pid];  // out of boundary
        ptcl_done[pid] = (leaked) ? 1 : ptcl_done[pid];
      }  // search node adj faces
    };  // find_next_element lambda
    parallel_for(ptcls, find_next_element_in_node_adj_faces,
                 "find_next_element_in_node_adj_faces");
    found = true;
    auto cp_elm_ids = OMEGA_H_LAMBDA(o::LO i) {
      elem_ids[i] = elem_ids_next[i];
    };
    o::parallel_for(elem_ids.size(), cp_elm_ids, "copy_elem_ids");
  }
  return found;
}

// OMEGA_H_DEVICE o::LO get_potential_next_face(o::LO e, o::LO face, o::LO
// &edge2faceFace, o::LO &edge2faceFaceOffsets)
//{
//   o::LO potential_next_face = -1;
//   o::LO n_adj_faces = edge2faceFaceOffsets[face + 1] -
//   edge2faceFaceOffsets[face]; OMEGA_H_CHECK(n_adj_faces == 1 || n_adj_faces
//   == 2);
//   // n_adj_faces is either 1 or 2
//   potential_next_face = (n_adj_faces == 2 &&
//   edge2faceFace[edge2faceFaceOffsets[face]] == e) ?
//   edge2faceFace[edge2faceFaceOffsets[face] + 1] :
//   edge2faceFace[edge2faceFaceOffsets[face]]; return potential_next_face;
// }
//  Function to check if three points are collinear
OMEGA_H_DEVICE bool areCollinear(const o::Vector<2>& p1, const o::Vector<2>& p2,
                                 const o::Vector<2>& p3) {
  // Calculate the area of the triangle formed by the points
  // If the area is zero, the points are collinear
  return std::abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) +
                   p3[0] * (p1[1] - p2[1])) /
                  2.0) < EPSILON * 100;
}

// Function to check if a point lies on a line segment
OMEGA_H_DEVICE bool isPointOnLineSegment(const o::Vector<2>& lineEnd1,
                                         const o::Vector<2>& lineEnd2,
                                         const o::Vector<2>& point) {
  // Check if the point is collinear with the line segment
  bool are_coollinear = areCollinear(lineEnd1, lineEnd2, point);

  // Check if the point lies within the bounds of the line segment
  o::Real max_x = (lineEnd1[0] > lineEnd2[0]) ? lineEnd1[0] : lineEnd2[0];
  o::Real min_x = (lineEnd1[0] < lineEnd2[0]) ? lineEnd1[0] : lineEnd2[0];
  o::Real max_y = (lineEnd1[1] > lineEnd2[1]) ? lineEnd1[1] : lineEnd2[1];
  o::Real min_y = (lineEnd1[1] < lineEnd2[1]) ? lineEnd1[1] : lineEnd2[1];

  bool are_in_range = (point[0] <= max_x && point[0] >= min_x &&
                       point[1] <= max_y && point[1] >= min_y);
  return are_coollinear && are_in_range;
}

bool search_adjacency_with_bcc(
    o::Mesh& mesh, PS* ptcls, p::Segment<double[3], Kokkos::CudaSpace> x,
    p::Segment<double[3], Kokkos::CudaSpace> xtgt,
    p::Segment<int, Kokkos::CudaSpace> pid, o::Write<o::LO> elem_ids,
    o::Write<o::Real> xpoints_d, o::Write<o::LO> xface_id,
    o::Read<o::I8> zone_boundary_sides, int looplimit = 10) {
  OMEGA_H_CHECK(mesh.dim() == 2);  // only for pseudo3D now
  const auto elemArea = o::measure_elements_real(&mesh);
  o::Real tol = p::compute_tolerance_from_area(elemArea);
  const auto side_is_exposed = o::mark_exposed_sides(&mesh);
  const auto coords = mesh.coords();
  const auto faces2nodes = mesh.ask_verts_of(o::FACE);
  const auto face2faceFace = mesh.ask_dual().ab2b;
  const auto face2faceOffsets = mesh.ask_dual().a2ab;
  const auto node2faceFace = mesh.ask_up(o::VERT, o::FACE).ab2b;
  const auto node2faceOffsets = mesh.ask_up(o::VERT, o::FACE).a2ab;
  const auto face2edgeEdge = mesh.ask_down(o::FACE, o::EDGE).ab2b;
  const auto edge2nodeNode =
      mesh.ask_down(o::EDGE, o::VERT).ab2b;  // always 2; offset not needed
  const auto edge2faceFace = mesh.ask_up(o::EDGE, o::FACE).ab2b;
  const auto edge2faceOffsets = mesh.ask_up(o::EDGE, o::FACE).a2ab;
  const auto psCapacity = ptcls->capacity();

  const auto exposed_edges = o::mark_exposed_sides(&mesh);

  o::Write<o::LO> ptcl_done(psCapacity, 0, "search_done");
  o::Write<o::LO> elem_ids_next(psCapacity, -1, "elem_ids_next");

  auto fill = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      elem_ids[pid] = e;
      // printf("INFO: Particle %d is in element %d\n", pid, e);
    } else {
      ptcl_done[pid] = 1;
    }
  };
  parallel_for(ptcls, fill, "searchMesh_fill_elem_ids");
  printf("INFO: Starting search for the particles\n");

  bool found = false;

  {  // original search in pumipush does it in a loop but here it is searched
    auto check_initial_position =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        auto elmId = elem_ids[pid];
        OMEGA_H_CHECK(elmId >= 0);
        // get the corners of the element
        o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1), xtgt(pid, 2)};
        const o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        // printf("INFO: Particle %d in element %d: origin: %.16f %.16f; dest: "
        //        "%.16f %.16f\n",
        //        pid, e, origin_rThz[0], origin_rThz[2], dest_rThz[0],
        //        dest_rThz[2]);
        //  printf("INFO: Checking particle %d in element %d\n", pid, elmId);
        //  printf("INFO: Origin position: %.16f %.16f %.16f\n", origin_rThz[0],
        //         origin_rThz[1], origin_rThz[2]);
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        // check if the particle is in the element
        o::Vector<3> bcc = o::barycentric_from_global<2, 2>(
            {origin_rThz[0], origin_rThz[2]}, current_el_vert_coords);
        if (!all_positive(bcc, EPSILON)) {
          printf(
              "Error: Particle not in this "
              "element"
              "\tpid %d elem %d\n",
              pid, elmId);
          printf("bcc %.16f %.16f %.16f\n", bcc[0], bcc[1], bcc[2]);
          printf("Position of the particle: %.16f %.16f\n", origin_rThz[0],
                 origin_rThz[2]);
          printf("Face vertex ids: %d %d %d\n", current_el_verts[0],
                 current_el_verts[1], current_el_verts[2]);
          // grab the location and find out the element
          // search_through_mesh(mesh, origin_rThz);
          OMEGA_H_CHECK(false);
        }
      }
    };
    parallel_for(ptcls, check_initial_position, "check_initial_postion");

    auto search_and_update_destination =
        PS_LAMBDA(const int& e, const int& pid, const int& mask) {
      if (mask > 0 && ptcl_done[pid] == 0) {
        const auto current_el_verts = o::gather_verts<3>(faces2nodes, e);
        const o::Few<o::LO, 3> current_el_edges = {face2edgeEdge[3 * e],
                                                   face2edgeEdge[3 * e + 1],
                                                   face2edgeEdge[3 * e + 2]};
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            o::gather_vectors<3, 2>(coords, current_el_verts);
        const o::Vector<3> dest_rThz = {xtgt(pid, 0), xtgt(pid, 1),
                                        xtgt(pid, 2)};
        const o::Vector<3> origin_rThz = {x(pid, 0), x(pid, 1), x(pid, 2)};
        const o::Vector<2> origin_rz = {origin_rThz[0], origin_rThz[2]};
        const o::Vector<2> dest_rz = {dest_rThz[0], dest_rThz[2]};
        const o::Vector<2> direction_rz =
            (dest_rz - origin_rz) / norm(dest_rz - origin_rz);

        auto bcc_dest =
            o::barycentric_from_global<2, 2>(dest_rz, current_el_vert_coords);
        auto bcc_origin =
            o::barycentric_from_global<2, 2>(origin_rz, current_el_vert_coords);
        // printf("INFO: BCC origin: %.16f %.16f %.16f\n", bcc_origin[0],
        //        bcc_origin[1], bcc_origin[2]);
#ifdef DEBUG
        check_origin_bcc_validity(bcc_origin);
#endif
        o::LO origin_on_edge =
            get_edge_holding_point(bcc_origin, current_el_edges);
        o::LO other_face = get_the_other_adj_face_of_edge(
            origin_on_edge, e, edge2faceOffsets, edge2faceFace, exposed_edges);
        // printf("INFO: Other face: %d\n", other_face);
        // NOTE: most of the time particle won't be on the edge

        // todo: modify this portion
        // it is important when the particle somehow ended up on the edge
        // but should be vary rare
        o::LO intersecting_face = e;
        if (origin_on_edge != -1) {  // dest in on the same side of the edge
                                     // with the other node of the face
          o::LO edge = origin_on_edge;
          o::Few<o::LO, 2> edge_verts = {edge2nodeNode[2 * edge],
                                         edge2nodeNode[2 * edge + 1]};
          o::LO third_vert = -1;
          for (int i = 0; i < 3; i++) {
            third_vert = (current_el_verts[i] != edge_verts[0] &&
                          current_el_verts[i] != edge_verts[1])
                             ? current_el_verts[i]
                             : third_vert;
          }
          OMEGA_H_CHECK(third_vert != -1);
          o::Vector<2> third_vert_coords = o::get_vector<2>(coords, third_vert);
          auto edge_coords = o::gather_vectors<2, 2>(coords, edge_verts);
          // ref: https://math.stackexchange.com/a/162733
          bool is_on_same_side_of_line =
              ((edge_coords[0][1] - edge_coords[1][1]) *
                   (third_vert_coords[0] - edge_coords[0][0]) +
               (edge_coords[1][0] - edge_coords[0][0]) *
                   (third_vert_coords[1] - edge_coords[0][1])) *
                  ((edge_coords[0][1] - edge_coords[1][1]) *
                       (dest_rz[0] - edge_coords[0][0]) +
                   (edge_coords[1][0] - edge_coords[0][0]) *
                       (dest_rz[1] - edge_coords[0][1])) >
              0;
          intersecting_face =
              (!is_on_same_side_of_line) ? other_face : intersecting_face;
        }
        // printf("INFO: Intersecting face: %d\n", intersecting_face);

        o::Real optical_distance = o::norm(dest_rz - origin_rz);
        bool leaked = false;
        o::Vector<2> current_origin = origin_rz;
        // o::LO max_iteration = 100;
        o::LO iteration = 0;
        // if (pid == 2){
        //   printf("Particle %d in element %d: Origin: %.16f, %.16f Dest:
        //   %.16f, %.16f\n", pid, intersecting_face, current_origin[0],
        //   current_origin[1], dest_rz[0], dest_rz[1]);
        // }
        while (optical_distance > 0.0) {
          // pseudoCode: 1. find the intersecting edge
          // 2. if the edge is not exposed, find the other face
          // 3. do this until the d is zero or edge is exposed
          // printf("Search: Particle %d in element %d: Current origin: %f,
          // %f\n", pid, intersecting_face, current_origin[0],
          // current_origin[1]);
          const auto intersecting_face_verts =
              o::gather_verts<3>(faces2nodes, intersecting_face);
          const auto intersecting_face_vert_coords =
              o::gather_vectors<3, 2>(coords, intersecting_face_verts);
          const o::Few<o::LO, 3> intersecting_face_edges = {
              face2edgeEdge[3 * intersecting_face],
              face2edgeEdge[3 * intersecting_face + 1],
              face2edgeEdge[3 * intersecting_face + 2]};
          const auto bcc_origin_intFace = o::barycentric_from_global<2, 2>(
              current_origin, intersecting_face_vert_coords);
          const auto bcc_dest_intFace = o::barycentric_from_global<2, 2>(
              dest_rz, intersecting_face_vert_coords);

          o::LO bcc_intersected_edge = -1;
          o::Vector<3> bcc_intersected_point = {0.0, 0.0, 0.0};
          bool intersected = false;
          for (int i = 0; i < 3; i++) {
            auto intersection_result = find_intersection_with_bcc(
                bcc_origin_intFace, bcc_dest_intFace, i);
            o::LO temp_edge = intersecting_face_edges[(i + 1) % 3];
            bool not_previous_edge = temp_edge != origin_on_edge;
            bool valid_intersection_found =
                intersection_result.exists && not_previous_edge;
            bcc_intersected_edge =
                (valid_intersection_found) ? temp_edge : bcc_intersected_edge;
            bcc_intersected_point = (valid_intersection_found)
                                        ? intersection_result.bcc
                                        : bcc_intersected_point;
            intersected = (valid_intersection_found) ? true : intersected;
            // if (pid == 9834){
            //  printf("Iter: %d :Particle %d in element %d: Intersected edge:
            //  %d\n",i, pid, intersecting_face, bcc_intersected_edge);
            //  printf("Iter: %d :Current origin: %.16f, %.16f\n current dest:
            //  %.16f, %.16f\n",i, current_origin[0], current_origin[1],
            //  dest_rz[0], dest_rz[1]); printf("Iter: %d :Origin on edge %d ,
            //  bcc_intersected_edge %d\n",i, origin_on_edge,
            //  bcc_intersected_edge);
            // }
          }
          origin_on_edge = (bcc_intersected_edge != -1) ? bcc_intersected_edge
                                                        : origin_on_edge;
#ifdef DEBUG
          bool edge_found = (bcc_intersected_edge != -1);
          OMEGA_H_CHECK(intersected == edge_found);
          // dest not in the element but didn't intersect any edge
          // if (!all_positive(bcc_dest_intFace, 0.0) && !intersected){
          //  printf("Search Error!!: particle %d dest not in the element %d but
          //  didn't intersect any edge\n", pid, intersecting_face);
          //}
          // OMEGA_H_CHECK(!all_positive(bcc_dest_intFace, 0.0) &&
          // !intersected);
#endif
          auto intersected_point = barycentric2real(
              intersecting_face_vert_coords, bcc_intersected_point);
          o::Vector<2> cur_it_dest =
              (intersected) ? intersected_point : dest_rz;
          o::Real parsed_distance = o::norm(cur_it_dest - current_origin);
          optical_distance = optical_distance - parsed_distance;

          o::LO next_face = get_the_other_adj_face_of_edge(
              bcc_intersected_edge, intersecting_face, edge2faceOffsets,
              edge2faceFace, exposed_edges);

          bool is_exposed = (bcc_intersected_edge != -1)
                                ? exposed_edges[bcc_intersected_edge]
                                : false;
          leaked = is_exposed;
          // if (leaked){ printf("INFO: Particle %d leaked from element %d edge
          // %d\n", pid, intersecting_face, bcc_intersected_edge);}
          // intersecting_face =
          //    (leaked || next_face == -1) ? intersecting_face : next_face;
          intersecting_face = (intersected) ? next_face : intersecting_face;
          // update origin position for next iteration in while loop
          current_origin = cur_it_dest;
          iteration++;
#ifdef dDEBUG
          // check if the current position is in the current element
          if (!leaked) {
            auto next_face_verts =
                o::gather_verts<3>(faces2nodes, intersecting_face);
            auto next_face_coords =
                o::gather_vectors<3, 2>(coords, next_face_verts);
            auto bcc_next_origin =
                o::barycentric_from_global<2, 2>(cur_it_dest, next_face_coords);
            bool invalid_bcc = !all_positive(bcc_next_origin, EPSILON);
            if (invalid_bcc) {
              // print all the information
              printf("Search Error!!: BCC origin: %.16f, %.16f, %.16f\n",
                     bcc_next_origin[0], bcc_next_origin[1],
                     bcc_next_origin[2]);
              printf(
                  "Search Error!!: Particle %d in element %d (original "
                  "element: %d): Origin: %.16f, %.16f "
                  "Dest: %.16f, %.16f\n",
                  pid, intersecting_face, e, origin_rThz[0], origin_rThz[2],
                  dest_rz[0], dest_rz[1]);
              printf("Search Error!!: Iteration: %d\n", iteration);
              printf(
                  "Search Error!!: Particle %d in element %d: Intersected "
                  "edge: %d\n",
                  pid, intersecting_face, bcc_intersected_edge);
              printf(
                  "Search Error!!: Particle %d in element %d: Intersected "
                  "point: %.16f, "
                  "%.16f\n",
                  pid, intersecting_face, intersected_point[0],
                  intersected_point[1]);
              printf(
                  "Search Error!!: Particle %d in element %d: Next face: %d\n",
                  pid, intersecting_face, next_face);
              printf(
                  "Search Error!!: Element vert coords: (%.16f, %.16f), "
                  "(%.16f, %.16f), "
                  "(%.16f, %.16f)\n",
                  next_face_coords[0][0], next_face_coords[0][1],
                  next_face_coords[1][0], next_face_coords[1][1],
                  next_face_coords[2][0], next_face_coords[2][1]);
            }
            OMEGA_H_CHECK(all_positive(bcc_next_origin, EPSILON));
          }

#endif

          // printf("Intersected edge for particle %d in element %d: %d\n", pid,
          // intersecting_face, bcc_intersected_edge);

          if (leaked || !intersected) {
            // if (leaked){
            // printf("Search ends for particle %d in element %d as it
            // leaked\n", pid, intersecting_face);
            // }
            // if (!intersected){
            // printf("Search ends for particle %d in element %d as it didn't
            // intersect\n", pid, intersecting_face);
            // }
            // if (optical_distance <= 0.0){
            //   printf("Search ends for particle %d in element %d as it reached
            //   the destination\n", pid, intersecting_face);
            // }
            break;
          }
        }
        elem_ids_next[pid] = (leaked) ? -1 : intersecting_face;

        // printf("INFO: Particle %d intersects at %.16f %.16f\n", pid,
        //        intersection_point[0], intersection_point[1]);
        // printf("INFO: Intersected edge: %d\n", intersected_edge);
      }
    };
    parallel_for(ptcls, search_and_update_destination,
                 "search_and_update_particle_destination");

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
      search_adjacency_with_bcc(*mesh, ptcls, x, xtgt, pid, elem_ids, xpoints_d,
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
  // if comm_size is 1, then all elements are owned by rank 0
  if (comm_size == 1) {
    o::parallel_for(mesh.nelems(), OMEGA_H_LAMBDA(o::LO i) { owners[i] = 0; });
    return;
  }
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
    o::Vector<2> center2d;
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
      OMEGA_H_CHECK(false);
    }
    // get rank based on the location of the center2d starting from the
    // lower-left corner and moving up and right
    int rank = int((center2d[0] - startx) / dx) +
               ncuts_x * int((center2d[1] - starty) / dy);
    if (rank >= comm_size || rank < 0) {
      printf("Error: rank is out of bounds\n");
      printf("Rank: %d comm_size: %d\n", rank, comm_size);
      OMEGA_H_CHECK(false);
    }
    owners[i] = rank;
  };
  o::parallel_for(mesh.nelems(), lamb);
  if (comm_rank == 0) {
    // varify_balance_of_partitions(owners, comm_size);
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

void computeAvgPtclDensity(p::Mesh& picparts, PS* ptcls) {
  o::Mesh* mesh = picparts.mesh();
  // create an array to store the number of particles in each element
  o::Write<o::LO> elmPtclCnt_w(mesh->nelems(), 0);
  // parallel loop over elements and particles
  auto lamb = PS_LAMBDA(const int& e, const int& pid, const int& mask) {
    if (mask > 0) {
      Kokkos::atomic_fetch_add(&(elmPtclCnt_w[e]), 1);
    }
  };
  ps::parallel_for(ptcls, lamb);
  o::Write<o::Real> epc_w(mesh->nelems(), 0);
  const auto convert = OMEGA_H_LAMBDA(o::LO i) {
    epc_w[i] = static_cast<o::Real>(elmPtclCnt_w[i]);
  };
  o::parallel_for(mesh->nelems(), convert, "convert_to_real");
  o::Reals epc(epc_w);
  mesh->add_tag(o::FACE, "element_particle_count", 1, o::Reals(epc));
  // get the list of elements adjacent to each vertex
  auto verts2elems = mesh->ask_up(o::VERT, picparts.dim());
  // create a device writeable array to store the computed density
  o::Write<o::Real> ad_w(mesh->nverts(), 0);
  const auto accumulate = OMEGA_H_LAMBDA(o::LO i) {
    const auto deg = verts2elems.a2ab[i + 1] - verts2elems.a2ab[i];
    const auto firstElm = verts2elems.a2ab[i];
    o::Real vertVal = 0.00;
    for (int j = 0; j < deg; j++) {
      const auto elm = verts2elems.ab2b[firstElm + j];
      vertVal += epc[elm];
    }
    ad_w[i] = vertVal / deg;
  };
  o::parallel_for(mesh->nverts(), accumulate, "calculate_avg_density");
  o::Read<o::Real> ad_r(ad_w);
  mesh->set_tag(o::VERT, "avg_density", ad_r);

  // compute area_weighted_avg_density
  o::Write<o::Real> particle_density(mesh->nelems(), 0.0);
  o::Real total_area = area_of_2d_mesh(*mesh);
  auto face2verts = mesh->ask_down(o::FACE, o::VERT).ab2b;
  auto coords = mesh->coords();

  auto compute_density = OMEGA_H_LAMBDA(o::LO i) {
    auto face_nodes = o::gather_verts<3>(face2verts, i);
    o::Few<o::Vector<2>, 3> face_coords =
        o::gather_vectors<3, 2>(coords, face_nodes);
    o::Real face_area = area_tri(face_coords);
    o::Real face_density = epc[i] / face_area;
    particle_density[i] = face_density;
  };
  o::parallel_for(mesh->nfaces(), compute_density, "compute_density");

  o::Reals particle_density_r(particle_density);
  mesh->add_tag(o::FACE, "particle_density", 1, o::Reals(particle_density_r));
}

o::Real area_of_2d_mesh(o::Mesh& mesh) {
  const auto coords = mesh.coords();
  const auto faces2nodes = mesh.ask_down(o::FACE, o::VERT).ab2b;
  const auto n_faces = mesh.nfaces();
  o::Real total_area = 0.0;

  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<>(0, n_faces),
      KOKKOS_LAMBDA(const int i, o::Real& local_area) {
        auto face_nodes = o::gather_verts<3>(faces2nodes, i);
        o::Few<o::Vector<2>, 3> face_coords;
        face_coords = o::gather_vectors<3, 2>(coords, face_nodes);
        o::Real face_area = area_tri(face_coords);
        local_area += face_area;
      },
      Kokkos::Sum<o::Real>(total_area));

  return total_area;
}