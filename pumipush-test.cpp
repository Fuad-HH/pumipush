#include "pumipush.h"

#include <Omega_h_bbox.hpp>
#include <Omega_h_defines.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_tag.hpp>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

bool is_close(double a, double b, double tol = 1e-5) {
  return std::abs(a - b) < tol;
}
template <typename T>
bool match_with_expected(T min, T max, T expected_min, T expected_max);
void varify7k3d(o::Library* lib);
void verifyCorseWestcase(o::Library* lib);
void checkCPNFileReading(o::Library* lib);
void check2dBox(o::Library* lib);
void check_cyl2cart();

int main(int argc, char** argv) {
  auto lib = o::Library(&argc, &argv);
  OMEGA_H_CHECK(std::string(lib.version()) == OMEGA_H_SEMVER);
  varify7k3d(&lib);
  verifyCorseWestcase(&lib);
  checkCPNFileReading(&lib);
  check2dBox(&lib);
  check_cyl2cart();
}

void varify7k3d(o::Library* lib) {
  o::Vector<2> min, max;
  Omega_h::Mesh mesh7kcube(lib);
  Omega_h::binary::read("assets/7k.osh", lib->world(), &mesh7kcube);
  std::cout << "Mesh loaded successfully with " << mesh7kcube.nelems()
            << " elements\n\t\t\t"
            << "and " << mesh7kcube.nverts() << " vertices\n";
  get_bounding_box_in_xy_plane(mesh7kcube, min, max);
  o::Vector<2> expected_min_7k = {-0.5, -0.5};
  o::Vector<2> expected_max_7k = {65, 65};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_7k, expected_max_7k));
}

void verifyCorseWestcase(o::Library* lib) {
  o::Vector<2> min, max;
  Omega_h::Mesh meshwestcoarse(lib);
  Omega_h::binary::read("assets/coarseWestLCPP.osh", lib->world(),
                        &meshwestcoarse);
  std::cout << "Mesh loaded successfully with " << meshwestcoarse.nelems()
            << " elements\n\t\t\t"
            << "and " << meshwestcoarse.nverts() << " vertices"
            << "dim = " << meshwestcoarse.dim() << "\n";
  get_bounding_box_in_xy_plane(meshwestcoarse, min, max);
  o::Vector<2> expected_min_west = {1.83431, -0.94};
  o::Vector<2> expected_max_west = {3.19847, 0.7986};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_west, expected_max_west));
}

void checkCPNFileReading(o::Library* lib) {
  std::string cpn_file = "test.cpn";
  o::Write<o::LO> owners(7, 0);
  ownerFromCPN(cpn_file, owners);

  std::vector<o::LO> expected_owner = {0, 0, 1, 2, 3, 3, 4};
  OMEGA_H_CHECK(owners.size() == expected_owner.size());
  for (int i = 0; i < owners.size(); i++) {
    OMEGA_H_CHECK(owners[i] == expected_owner[i]);
  }
}

void check2dBox(o::Library* lib) {
  o::Vector<2> min, max;
  // 2d case
  Omega_h::Mesh mesh2d(lib);
  Omega_h::binary::read("assets/2d_box.osh", lib->world(), &mesh2d);
  std::cout << "Mesh loaded successfully with " << mesh2d.nelems()
            << " elements\n\t\t\t"
            << "and " << mesh2d.nverts() << " vertices\n";
  get_bounding_box_in_xy_plane(mesh2d, min, max);
  o::Vector<2> expected_min_2d = {0, 0};
  o::Vector<2> expected_max_2d = {1, 0.8};
  OMEGA_H_CHECK(
      match_with_expected(min, max, expected_min_2d, expected_max_2d));
}

template <typename T>
bool match_with_expected(T min, T max, T expected_min, T expected_max) {
  if (!is_close(min[0], expected_min[0]) ||
      !is_close(min[1], expected_min[1]) ||
      !is_close(max[0], expected_max[0]) ||
      !is_close(max[1], expected_max[1])) {
    return false;
  }
  return true;
}

void check_cyl2cart() {
  o::Vector<3> cyl = {1, 0, 0};
  o::Vector<3> cart;
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 1));
  OMEGA_H_CHECK(is_close(cart[1], 0));
  OMEGA_H_CHECK(is_close(cart[2], 0));

  cyl = {1, o::PI / 2, 3};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 0));
  OMEGA_H_CHECK(is_close(cart[1], 1));
  OMEGA_H_CHECK(is_close(cart[2], 3));

  cyl = {1, o::PI, 0};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], -1));
  OMEGA_H_CHECK(is_close(cart[1], 0));
  OMEGA_H_CHECK(is_close(cart[2], 0));

  cyl = {2, -o::PI / 2, 0};
  cylindrical2cartesian(cyl, cart);
  OMEGA_H_CHECK(is_close(cart[0], 0));
  OMEGA_H_CHECK(is_close(cart[1], -2));
  OMEGA_H_CHECK(is_close(cart[2], 0));
}

void cart2cyl() {
  o::Vector<3> cart = {1, 0, 0};
  o::Vector<3> cyl;
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], 0));
  OMEGA_H_CHECK(is_close(cyl[2], 0));

  cart = {0, 1, 3};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], o::PI / 2));
  OMEGA_H_CHECK(is_close(cyl[2], 3));

  cart = {-1, 0, 0};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 1));
  OMEGA_H_CHECK(is_close(cyl[1], o::PI));
  OMEGA_H_CHECK(is_close(cyl[2], 0));

  // 4th quadrant
  cart = {0, -2, 5};
  cartesian2cylindrical(cart, cyl);
  OMEGA_H_CHECK(is_close(cyl[0], 2));
  OMEGA_H_CHECK(is_close(cyl[1], -o::PI / 2));
  OMEGA_H_CHECK(is_close(cyl[2], 5));
}