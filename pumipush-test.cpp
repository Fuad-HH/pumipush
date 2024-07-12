#include "pumipush.h"

#include <Omega_h_tag.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
  // ****** owner from cpn file ****** //
  std::string cpn_file = "test.cpn";
  o::Write<o::LO> owners(7, 0);
  ownerFromCPN(cpn_file, owners);

  std::vector<o::LO> expected_owner = {0, 0, 1, 2, 3, 3, 4};
  if (owners.size() != expected_owner.size()) {
    std::cerr << "Error: ownerFromCPN failed: size mismatch: " << owners.size()
              << " != " << expected_owner.size() << "\n";
    return 1;
  }
  for (int i = 0; i < owners.size(); i++) {
    if (owners[i] != expected_owner[i]) {
      std::cerr << "Error: ownerFromCPN failed: value mismatch\n";
      return 1;
    }
  }

  return 0;
}