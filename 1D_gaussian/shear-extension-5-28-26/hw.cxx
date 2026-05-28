#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>

class HW : public PhysicsModel {

private:

  Field3D T, K;
  Field3D U, kx;

  Field3D chi_T, chi;

  Field3D S;
  Field3D Rxy;

  // Parameters
  BoutReal chi_not;
  BoutReal g;
  BoutReal tau_ac;
  BoutReal gamma;
  BoutReal Kc;
  BoutReal beta;

  // Reynolds stress parameters
  BoutReal C_R;
  BoutReal ky_const;

  // Mean flow damping + viscosity
  BoutReal mu_U;
  BoutReal nu_U;

  // Energy-consistent transfer coefficient
  BoutReal alpha_RS;

protected:

  int init(bool restarting) override {

    auto* options = Options::getRoot()->getSection("hw");

    OPTION(options, chi_not, 1e-5);

    OPTION(options, g, 5e-1);
    OPTION(options, tau_ac, 1e1);
    OPTION(options, gamma, 0.0);
    OPTION(options, Kc, 1e-1);
    OPTION(options, beta, 1e-1);

    OPTION(options, C_R, 1.0);
    OPTION(options, ky_const, 1.0);

    OPTION(options, mu_U, 0.1);
    OPTION(options, nu_U, 1e-5);

    OPTION(options, alpha_RS, 1.0);

    SOLVE_FOR(T, K, U, kx);

    SAVE_REPEAT(chi_T);
    SAVE_REPEAT(chi);

    SAVE_REPEAT(U);
    SAVE_REPEAT(kx);

    SAVE_REPEAT(S);
    SAVE_REPEAT(Rxy);

    return 0;
  }

  int rhs(BoutReal time) override {

    mesh->communicate(T, K, U, kx);

    // -----------------------------------------
    // Mean shear
    // -----------------------------------------

    S = DDX(U);

    mesh->communicate(S);

    // -----------------------------------------
    // Turbulent diffusivity
    // -----------------------------------------

    chi_T = tau_ac * K;

    // Positivity floor
    for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {

      BoutReal val = chi_T(ix,0,0);

      if (val < 1e-16)
        val = 1e-16;

      chi_T(ix,0,0) = val;
    }

    mesh->communicate(chi_T);

    chi = chi_T + chi_not;

    mesh->communicate(chi);

    // -----------------------------------------
    // Reynolds stress
    // -----------------------------------------

    Rxy = C_R * kx * ky_const * K;

    mesh->communicate(Rxy);

    // -----------------------------------------
    // Temperature equation
    // -----------------------------------------

    ddt(T) = FDDX(chi, DDX(T));

    // -----------------------------------------
    // Safe K for nonlinear damping
    // -----------------------------------------

    Field3D K_safe = K;

    for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {

      BoutReal val = K_safe(ix,0,0);

      if (val < 1e-16)
        val = 1e-16;

      K_safe(ix,0,0) = val;
    }

    // -----------------------------------------
    // Reynolds-stress energy transfer
    // -----------------------------------------

    Field3D transfer_RS = alpha_RS * Rxy * S;

    // -----------------------------------------
    // Kinetic energy equation
    // -----------------------------------------

    ddt(K)
      =
        beta * FDDX(chi_T, DDX(K))
        - g * chi_T * DDX(T)
        + gamma * K
        - Kc * pow(K_safe, 1.5)
        - transfer_RS;

    // -----------------------------------------
    // Mean flow equation
    // -----------------------------------------

    ddt(U)
      =
        - DDX(Rxy)
        - mu_U * U
        + nu_U * D2DX2(U);

    // -----------------------------------------
    // Eikonal equation
    // -----------------------------------------

    ddt(kx)
      =
        - ky_const * S;

    return 0;
  }
};

BOUTMAIN(HW);