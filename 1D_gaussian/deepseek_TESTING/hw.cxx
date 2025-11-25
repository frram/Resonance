#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D chi_T, chi;

  // Parameters
  BoutReal chi_not, g, tau_ac, gamma, Kc, beta;
  BoutReal sigma;  // Your σ_χ parameter

protected:
  int init(bool restart) override {
    auto* options = Options::getRoot()->getSection("hw");
    
    OPTION(options, chi_not, 1e-5);
    OPTION(options, g,       0.5);
    OPTION(options, tau_ac,  0.125);
    OPTION(options, gamma,   0.06);
    OPTION(options, Kc,      0.1);
    OPTION(options, beta,    0.1);
    OPTION(options, sigma, 0.01);  // Gaussian width
    
    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T);
    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K);
    
    // Handle the sigma_chi = 0 case separately to avoid numerical issues
    if (sigma < 1e-10) {
      // Purely local transport - matches your working case
      chi_T = tau_ac * K;
    } else {
      // Gaussian kernel convolution: χ_T(x_i) = τ_ac ∑_j G(x_i - x_j) K(x_j)
      chi_T = 0.0;
      
      // Ensure K has valid guard cells
      mesh->communicate(K);
      
      // Calculate the convolution range in grid points (3σ cutoff)
      const int range = static_cast<int>(3.0 * sigma * mesh->LocalNx) + 1;
      
      // Gaussian convolution
      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        const BoutReal xi = mesh->GlobalX(ix);
        BoutReal wsum = 0.0;
        BoutReal acc = 0.0;
        
        // Determine convolution range
        int jmin = std::max(0, ix - range);
        int jmax = std::min(mesh->LocalNx-1, ix + range);
        
        for (int jx = jmin; jx <= jmax; ++jx) {
          const BoutReal xj = mesh->GlobalX(jx);
          const BoutReal dx = xi - xj;
          const BoutReal w = exp(-(dx * dx) / (2.0 * sigma * sigma));
          wsum += w;
          acc += w * K(jx, 0, 0);
        }
        
        // Normalized convolution with safety check
        chi_T(ix, 0, 0) = tau_ac * (acc / (wsum + 1e-16));
      }
      
      // Apply boundary conditions
      mesh->communicate(chi_T);
      mesh->applyBoundary(chi_T, "T");
    }
    
    // Total diffusivity for T equation
    chi = chi_T + chi_not;

    // RHS equations
    ddt(T) = FDDX(chi, DDX(T));
    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K, 1.5);

    return 0;
  }
};