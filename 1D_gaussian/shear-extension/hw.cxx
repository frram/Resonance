#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D U, kx;              // NEW: mean flow and radial wavenumber
  Field3D chi_T, chi;
  Field3D S, Rxy;             // NEW: diagnostic fields

  // Parameters (existing + new)
  BoutReal chi_not, g, gamma, Kc, beta;

  // NEW parameters for shear model
  BoutReal chi1, S_star;
  BoutReal C_R, ky_const;
  BoutReal mu_U, nu_U;

protected:
  int init(bool restarting) override {
    auto *opt = Options::getRoot()->getSection("hw");

    // Existing parameters
    OPTION(opt, chi_not, 1e-5);
    OPTION(opt, g, 1.0);
    OPTION(opt, gamma, 0.0);
    OPTION(opt, Kc, 0.1);
    OPTION(opt, beta, 0.0);

    // NEW: shear-suppressed diffusivity parameters
    OPTION(opt, chi1, 10.0);
    OPTION(opt, S_star, 1.0);

    // NEW: Reynolds stress and eikonal parameters
    OPTION(opt, C_R, 0.1);
    OPTION(opt, ky_const, 1.0);

    // NEW: mean-flow damping and viscosity
    OPTION(opt, mu_U, 0.1);
    OPTION(opt, nu_U, 1e-5);

    // Evolve fields
    SOLVE_FOR(T, K, U, kx);

    // Save diagnostics
    SAVE_REPEAT(chi_T);
    SAVE_REPEAT(chi);
    SAVE_REPEAT(S);
    SAVE_REPEAT(Rxy);

    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K, U, kx);

    // 1) Compute shear S = dU/dx
    S = DDX(U);
    mesh->communicate(S);

    // 2) Shear-suppressed local turbulent diffusivity:
    //    chi_T = chi1*K / (1 + (S/S_star)^2)
    chi_T = chi1 * K / (1.0 + SQ(S / S_star));

    // Enforce positivity for chi_T
    for (int ix = 0; ix < mesh->LocalNx; ++ix) {
      BoutReal val = chi_T(ix, 0, 0);
      if (val < 0.0) val = 0.0;
      if (val < 1e-16) val = 1e-16;
      chi_T(ix, 0, 0) = val;
    }
    mesh->communicate(chi_T);

    // Total diffusivity
    chi = chi_T + chi_not;
    mesh->communicate(chi);

    // 3) Reynolds stress closure: Rxy = C_R * kx * ky * K
    //    ky is constant (zonal symmetry in y)
    Rxy = C_R * kx * ky_const * K;
    mesh->communicate(Rxy);

    // 4) T equation
    ddt(T) = FDDX(chi, DDX(T));

    // 5) K equation (same structure as yours; with safe power)
    Field3D K_safe = K;
    for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
      BoutReal val = K_safe(ix, 0, 0);
      if (val < 1e-16) val = 1e-16;
      K_safe(ix, 0, 0) = val;
    }

    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K_safe, 1.5);

    // 6) Mean flow U equation: dU/dt = - d/dx(Rxy) - mu_U U + nu_U d2U/dx2
    ddt(U) = -DDX(Rxy) - mu_U * U + nu_U * D2DX2(U);

    // 7) Strict eikonal evolution for kx: dkx/dt = -ky * S = -ky * dU/dx
    ddt(kx) = -ky_const * S;

    return 0;
  }
};

BOUTMAIN(HW);
