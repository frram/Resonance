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


  // 2D Gaussian kernel (isotropic)
  inline BoutReal gaussian(BoutReal dx, BoutReal dy, BoutReal Delta) {
    return exp(-(dx*dx + dy*dy) / (2.0 * Delta * Delta));
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

  // --- build chi_T from K by full 2D convolution (no window) ---
  chi_T = 0.0;

  // If your build has GlobalX/GlobalY:
  for (int ix = 0; ix < mesh->LocalNx; ++ix) {
    BoutReal x_i = mesh->GlobalX(ix);
    for (int iy = 0; iy < mesh->LocalNy; ++iy) {
      BoutReal y_i = mesh->GlobalY(iy);

      BoutReal sum  = 0.0;
      BoutReal sumw = 0.0;

      for (int jx = 0; jx < mesh->LocalNx; ++jx) {
        BoutReal x_j = mesh->GlobalX(jx);
        for (int jy = 0; jy < mesh->LocalNy; ++jy) {
          BoutReal y_j = mesh->GlobalY(jy);
          BoutReal w   = gaussian(x_i - x_j, y_i - y_j, Delta);
          sum  += K(jx, jy, 0) * w;
          sumw += w;
      }
    }

    chi_T(ix, iy, 0) = tau_ac * (sum / std::max(sumw, 1e-30)); // normalized
  }
}

  // Light post-smoothing (optional; do it AFTER the loops)
  chi_T = smooth_x(chi_T);
  chi_T = smooth_y(chi_T);

  chi_T = max(chi_T, 1e-6);
  chi_T = min(chi_T, 5e-2);

    // Evolve temperature using divergence of (chi * grad T)
    ddt(T) = FDDX(chi_T, DDX(T)) + FDDY(chi_T, DDY(T))  + chi_not * ( D2DX2(T) + D2DY2(T) );

    // Evolve kinetic energy using buoyancy and nonlinear damping
    ddt(K) = FDDX(chi_T, DDX(K)) + FDDY(chi_T, DDY(K)) - g* chi_T * ( DDX(T) + DDY(T) ) + gamma*K - Kc*pow(K,1.5);

	return 0;
  }
};

// Define a main() function
BOUTMAIN(HW);
