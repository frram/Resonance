#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D chi_T, chi;
  
  // Parameters
  BoutReal chi_not, g, tau_ac, gamma, Kc, beta;
  BoutReal lambda_chi;
  int helm_iters;  // Number of Jacobi iterations

  // Jacobi solver for: (1 - λ²∇²)χ_T = τ_ac K
  Field3D jacobi_helmholtz(const Field3D& K_source) {
    Field3D result = tau_ac * K_source;
    Field3D new_result = result;
    
    BoutReal dx = mesh->GlobalX(1) - mesh->GlobalX(0);
    BoutReal alpha = lambda_chi * lambda_chi / (dx * dx);
    BoutReal denom = 1.0 + 2.0 * alpha;
    
    // Ensure initial positivity
    for (int ix = 0; ix < mesh->LocalNx; ++ix) {
      BoutReal val = result(ix,0,0);
      if (val < 0.0) val = 0.0;
      if (val < 1e-16) val = 1e-16;
      result(ix,0,0) = val;
    }
    
    for (int iter = 0; iter < helm_iters; ++iter) {
      mesh->communicate(result);  // Update boundary cells
      
      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        // RHS: τ_ac K, ensure positivity
        BoutReal rhs = tau_ac * K_source(ix, 0, 0);
        if (rhs < 0.0) rhs = 0.0;
        if (rhs < 1e-16) rhs = 1e-16;
        
        BoutReal left = result(ix-1, 0, 0);
        BoutReal right = result(ix+1, 0, 0);
        
        // Jacobi update: χ_new = (τ_ac K + α(χ_left + χ_right)) / (1 + 2α)
        BoutReal new_val = (rhs + alpha * (left + right)) / denom;
        
        // Enforce positivity
        if (new_val < 0.0) new_val = 0.0;
        if (new_val < 1e-16) new_val = 1e-16;
        
        new_result(ix, 0, 0) = new_val;
      }
      
      // Under-relaxation for stability (ω = 0.8)
      BoutReal omega = 0.8;
      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        result(ix,0,0) = omega * new_result(ix,0,0) + (1.0 - omega) * result(ix,0,0);
      }
    }
    
    // Final positivity enforcement
    for (int ix = 0; ix < mesh->LocalNx; ++ix) {
      BoutReal val = result(ix,0,0);
      if (val < 0.0) val = 0.0;
      if (val < 1e-16) val = 1e-16;
      result(ix,0,0) = val;
    }
    
    mesh->communicate(result);
    return result;
  }

protected:
  int init(bool restarting) override {
    auto *opt = Options::getRoot()->getSection("hw");

    OPTION(opt, chi_not, 1e-5);
    OPTION(opt, g, 1.0);
    OPTION(opt, tau_ac, 10.0);  // Your original value
    OPTION(opt, gamma, 0.0);
    OPTION(opt, Kc, 0.1);
    OPTION(opt, beta, 0.0);
    
    OPTION(opt, lambda_chi, 0.02);
    OPTION(opt, helm_iters, 100);  // Jacobi iterations

    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T);
    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K);

    if (lambda_chi < 1e-10) {
      // Pure local transport
      chi_T = tau_ac * K;
      
      // Ensure positivity
      for (int ix = 0; ix < mesh->LocalNx; ++ix) {
        BoutReal val = chi_T(ix,0,0);
        if (val < 0.0) val = 0.0;
        if (val < 1e-16) val = 1e-16;
        chi_T(ix,0,0) = val;
      }
    } else {
      // Helmholtz smoothing via Jacobi iteration
      chi_T = jacobi_helmholtz(K);
    }
    
    mesh->communicate(chi_T);
    
    // Total diffusivity
    chi = chi_T + chi_not;

    // RHS equations with safety
    ddt(T) = FDDX(chi, DDX(T));
    
    // Safe K for pow() function
    Field3D K_safe = K;
    for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
      BoutReal val = K_safe(ix,0,0);
      if (val < 1e-16) val = 1e-16;
      K_safe(ix,0,0) = val;
    }
    
    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K_safe, 1.5);

    return 0;
  }
};

BOUTMAIN(HW);
