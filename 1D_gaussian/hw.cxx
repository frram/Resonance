#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D chi_T, chi;

  // Model parameters
  BoutReal chi_not;
  BoutReal g;
  BoutReal tau_ac;
  BoutReal gamma;
  BoutReal Kc;
  BoutReal beta;
  BoutReal sigma;

  inline BoutReal gaussian(BoutReal dx) const {
    return exp(-(dx * dx) / (2.0 * sigma * sigma));
  }

protected:
  int init(bool restart) override {
    Options* options = Options::getRoot()->getSection("hw");

    OPTION(options, chi_not, 1e-5);
    OPTION(options, g, 0.5);
    OPTION(options, tau_ac, 0.125);
    OPTION(options, gamma, 0.06);
    OPTION(options, Kc, 0.1);
    OPTION(options, beta, 0.1);
    OPTION(options, sigma, 1e-2);

    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T);
    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K);
    chi_T = 0.0;

    // --- Case 1: sigma = 0 → purely local diffusivity
    if (sigma == 0.0) {
      chi_T = tau_ac * K;

    // --- Case 2: sigma > 0 → Gaussian-weighted averaging
    } else {
      for (int ix = 0; ix < mesh->LocalNx; ++ix) {
        const BoutReal xi = mesh->GlobalX(ix);
        BoutReal wsum = 0.0;
        BoutReal acc  = 0.0;

        for (int jx = 0; jx < mesh->LocalNx; ++jx) {
          const BoutReal xj = mesh->GlobalX(jx);
          const BoutReal dx = xi - xj;
          const BoutReal w  = gaussian(dx);

          wsum += w;
          acc  += w * K(jx, 0, 0);
        }

        const BoutReal Kavg = (wsum > 0.0) ? (acc / wsum) : 0.0;
        chi_T(ix, 0, 0) = tau_ac * Kavg;
      }

      chi_T = smooth_x(chi_T);
    }

    // Add background diffusivity
    chi = chi_T + chi_not;

    // --- Evolve temperature and kinetic energy ---
    ddt(T) = FDDX(chi, DDX(T));
    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K, 1.5);

    return 0;
  }
};

BOUTMAIN(HW);