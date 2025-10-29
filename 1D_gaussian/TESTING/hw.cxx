#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  Field3D T, K;
  Field3D chi_T, chi;

  // Parameters
  BoutReal chi_not, g, tau_ac, gamma, Kc, beta;
  // Nonlocality control:
  BoutReal sigma;        // keep your old Gaussian knob (sigma). We'll allow sigma=0 to mean local chi
  BoutReal lambda_chi;   // Helmholtz smoothing length (exponential kernel equivalent)
  int helm_iters;        // Jacobi iterations per RHS
  BoutReal helm_tol;     // (optional) stop criterion on max update

  inline BoutReal gaussian(BoutReal dx) const {
    return exp(-(dx * dx) / (2.0 * sigma * sigma));
  }

  // ---- Solve (I - lambda^2 d2/dx2) chi_T = tau_ac * K  with Jacobi sweeps ----
  void solve_helmholtz_chi(Field3D &chi_T_out, const Field3D &K_in) {
    // grid spacing (assume uniform)
    const BoutReal dx = mesh->GlobalX(1) - mesh->GlobalX(0);
    const BoutReal alpha = (lambda_chi * lambda_chi) / (dx * dx);
    const BoutReal denom = 1.0 + 2.0 * alpha;

    // Initial guess: local scaling
    chi_T_out = tau_ac * K_in;

    Field3D chi_new; chi_new = chi_T_out;

    for (int it = 0; it < helm_iters; ++it) {
      // update halos so neighbors are valid across ranks
      mesh->communicate(chi_T_out);

      BoutReal max_update = 0.0;

      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        // 1D in x; y=z=0
        const BoutReal rhs = tau_ac * K_in(ix,0,0);
        const BoutReal left  = chi_T_out(ix-1,0,0); // guard cells available
        const BoutReal right = chi_T_out(ix+1,0,0);

        // Jacobi update from discretization of (I - λ^2 d2/dx2) χ = rhs
        const BoutReal newval = (rhs + alpha * (left + right)) / denom;
        max_update = std::max(max_update, fabs(newval - chi_T_out(ix,0,0)));
        chi_new(ix,0,0) = newval;
      }

      // Apply natural Neumann BC via mesh boundaries (copies into guards)
      chi_T_out = chi_new;
      mesh->applyBoundary(chi_T_out);

      if (max_update < helm_tol) break; // optional early exit
    }
  }

protected:
  int init(bool restart) override {
    auto* options = Options::getRoot()->getSection("hw");

    OPTION(options, chi_not, 1e-5);
    OPTION(options, g,       0.5);
    OPTION(options, tau_ac,  0.125);
    OPTION(options, gamma,   0.06);
    OPTION(options, Kc,      0.1);
    OPTION(options, beta,    0.1);

    // Nonlocal knobs
    OPTION(options, sigma,       1e-2);  // if you still want the Gaussian path
    OPTION(options, lambda_chi,  1e-2);  // smoothing length for PDE smoother
    OPTION(options, helm_iters,  50);    // 20–100 is usually plenty
    OPTION(options, helm_tol,    1e-10); // small tolerance

    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T);
    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K);

    // Choose the path:
    //  - If sigma < tiny: purely local (chi_T = tau_ac K)
    //  - Else if lambda_chi > 0: PDE smoother (recommended for MPI)
    //  - Else: Gaussian average (legacy)
    const bool use_local     = (sigma < 1e-16) && (lambda_chi < 1e-16);
    const bool use_pde       = (lambda_chi >= 1e-16);

    if (use_local) {
      chi_T = tau_ac * K;

    } else if (use_pde) {
      solve_helmholtz_chi(chi_T, K);

    } else {
      // Gaussian-weighted average (older path; not MPI-global unless you widen halos)
      chi_T = 0.0;
      for (int ix = mesh->LocalNx - 1; ix >= 0; --ix) {
        const BoutReal xi = mesh->GlobalX(ix);
        BoutReal wsum = 0.0, acc = 0.0;
        for (int jx = 0; jx < mesh->LocalNx; ++jx) {
          const BoutReal xj = mesh->GlobalX(jx);
          const BoutReal dx = xi - xj;
          const BoutReal w  = gaussian(dx);
          wsum += w;
          acc  += w * K(jx,0,0);
        }
        chi_T(ix,0,0) = tau_ac * (wsum > 0.0 ? acc/wsum : 0.0);
      }
      chi_T = smooth_x(chi_T);
    }

    // Total diffusivity for T equation
    chi = chi_T + chi_not;

    // RHS
    ddt(T) = FDDX(chi, DDX(T));
    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K, 1.5);

    return 0;
  }
};

BOUTMAIN(HW);
