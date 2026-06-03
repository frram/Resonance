#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>

class HW : public PhysicsModel {

private:

  Field3D T, K;
  Field3D U, theta;

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

  // Mean flow damping + viscosity
  BoutReal mu_U;
  BoutReal nu_U;

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

    OPTION(options, mu_U, 0.1);
    OPTION(options, nu_U, 1e-5);

    SOLVE_FOR(T, K, U, theta);

    SAVE_REPEAT(chi_T);
    SAVE_REPEAT(chi);

    SAVE_REPEAT(U);
    SAVE_REPEAT(theta);

    SAVE_REPEAT(S);
    SAVE_REPEAT(Rxy);

    return 0;
  }

  int rhs(BoutReal time) override {

    mesh->communicate(T, K, U, theta);

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
    // Reynolds stress using bounded angle closure
    //
    // theta = atan(kx / ky)
    //
    // kx ky / (kx^2 + ky^2)
    //     = 0.5 * sin(2 theta)
    //
    // Therefore |Rxy| <= 0.5 * C_R * K
    // -----------------------------------------

    Rxy = 0.5 * C_R * K * sin(2.0 * theta);

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
    //
    // alpha_RS is fixed to 1 for energy consistency.
    // -----------------------------------------

    Field3D transfer_RS = Rxy * S;

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
    // Bounded angle equation
    //
    // kx = ky tan(theta)
    // d_t kx = ky sec^2(theta) d_t theta
    // original: d_t kx = - ky S
    //
    // Therefore:
    // d_t theta = - S cos^2(theta)
    // -----------------------------------------

    ddt(theta)
      =
        - S * pow(cos(theta), 2);

    return 0;
  }
};

BOUTMAIN(HW);