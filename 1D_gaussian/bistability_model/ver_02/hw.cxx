#include <bout/derivs.hxx>
#include <bout/invert_laplace.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D chi_T, chi, roll;

  // Model parameters
  BoutReal chi_not;   // Background diffusivity
  BoutReal Delta;     // Cell width
  BoutReal g;         // Curvature / buoyancy strength
  BoutReal tau_ac;    // Autocorrelation timescale
  BoutReal gamma;     // Linear production / damping of K
  BoutReal Kc;        // Nonlinear damping coefficient
  BoutReal beta;      // K spreading coefficient

  // Rollover parameters
  BoutReal Gc;        // Characteristic rollover gradient
  BoutReal p_roll;    // Rollover sharpness exponent

protected:
  int init(bool restart) {

    Options* options = Options::getRoot()->getSection("hw");
    OPTION(options, chi_not, 1e-5);
    OPTION(options, Delta, 4e-2);
    OPTION(options, g, 5e-1);
    OPTION(options, tau_ac, 1.25e-1);
    OPTION(options, gamma, 6e-2);
    OPTION(options, Kc, 1e-1);
    OPTION(options, beta, 1e-1);

    OPTION(options, Gc, 1.0);
    OPTION(options, p_roll, 2.0);

    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T, chi, roll);

    return 0;
  }

  int rhs(BoutReal time) {

    // Communicate variables
    mesh->communicate(T, K);

    // Rollover depends on gradient magnitude
    roll = 1.0 / (1.0 + pow(abs(DDX(T)) / Gc, p_roll));

    // Turbulent diffusivity with rollover
    chi_T = tau_ac * K * roll;

    // Total diffusivity
    chi = chi_T + chi_not;

    // Evolve temperature using divergence of (chi * grad T)
    ddt(T) = FDDX(chi, DDX(T));

    // Evolve kinetic energy using buoyancy and nonlinear damping
    // No extra suppression in the production term
    ddt(K) = beta * FDDX(chi_T, DDX(K))
             - g * chi_T * DDX(T)
             + gamma * K
             - Kc * pow(K, 1.5);

    return 0;
  }
};

// Define a main() function
BOUTMAIN(HW);