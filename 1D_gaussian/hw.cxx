#include <bout/derivs.hxx>
#include <bout/invert_laplace.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>


class HW : public PhysicsModel {
private:
  Field3D T, K;  // Evolving density
  Field3D chi_T; // chi_T from kernel

  // Model parameters
  BoutReal chi_not; // Background diffusivity
  BoutReal Delta;  // cell width
  BoutReal g;       // curvature/buoyancy strength
  BoutReal tau_ac;  // Autocorrelation timescale
  BoutReal gamma;   // production of K 
  BoutReal Kc;      // Nonlinear damping coefficient


  // Gaussian kernel
  inline BoutReal gaussian(BoutReal dx, Delta) {
    return exp(-pow(dx,2) / (2.0 * Delta * Delta));
  }

protected:
  int init(bool restart) {

	Options *options = Options::getRoot()->getSection("hw");
	OPTION(options, chi_not, 1e-5);
	OPTION(options, Delta, 4e-2);
	OPTION(options, g, 5e-1);
  OPTION(options, tau_ac, 1.25e-1);
  OPTION(options, gamma, 6e-2);
  OPTION(options, Kc, 1e-1);

	SOLVE_FOR(T,K);
  SAVE_REPEAT(chi_T);
	return 0;
  }

  int rhs(BoutReal time) {

	// Communicate variables
	mesh->communicate(T, K);

    // Reset chi_T to zero
  chi_T = 0.0;

    // Compute chi_T(x) via Gaussian convolution of K
    for (int ix = 0; ix < mesh->LocalNx; ++ix) {
        BoutReal x_i = mesh->GlobalX(ix);
        BoutReal sum = 0.0;
  
        for (int jx = 0; jx < mesh->LocalNx; ++jx) {
          BoutReal x_j = mesh->GlobalX(jx);
          BoutReal dx = x_i - x_j;
  
          BoutReal weight = gaussian(dx, Delta);
          sum += K(jx, 0, 0) * weight;
        }
  
        chi_T(ix, 0, 0) = tau_ac * sum;
        chi_T = smooth_x(chi_T);  // This smooths the final chi_T field

      }

    // Evolve temperature using divergence of (chi * grad T)
    ddt(T) = FDDX(chi_T, DDX(T)) + chi_not * D2D2X(T);

    // Evolve kinetic energy using buoyancy and nonlinear damping
    ddt(K) = FDDX(chi_T, DDX(K)) - g* chi_T *DDX(T) + gamma*K - Kc*pow(K,1.5);

	return 0;
  }
};

// Define a main() function
BOUTMAIN(HW);
