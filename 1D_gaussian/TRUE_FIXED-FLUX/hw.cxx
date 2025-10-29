#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <bout/smoothing.hxx>
#include <cmath>

class HW : public PhysicsModel {
private:
  // Fields
  Field3D T, K;
  Field3D chi_T, chi;

  // Parameters
  BoutReal chi_not, g, tau_ac, gamma, Kc, beta;

  // Nonlocality control
  BoutReal sigma;        // Gaussian kernel width; sigma=0 disables Gaussian path
  BoutReal lambda_chi;   // Helmholtz smoothing length; lambda_chi=0 disables PDE path
  int      helm_iters;   // Jacobi iterations
  BoutReal helm_tol;     // Jacobi tol

  // True fixed-flux BC controls
  bool     use_true_fixed_flux; // if true, enforce Q-in/out each timestep
  BoutReal Qin;                 // heat flux INTO domain at xmin (>0 means inflow)
  BoutReal Qout;                // heat flux OUT of domain at xmax (>0 means outflow)
  BoutReal chit_floor = 1e-12;  // avoid division blow-ups if chi_tot ~ 0

  inline BoutReal gaussian(BoutReal dx) const {
    return exp(-(dx * dx) / (2.0 * sigma * sigma));
  }

  // Solve (I - lambda^2 d2/dx2) chi_T = tau_ac * K with simple Jacobi
  void solve_helmholtz_chi(Field3D &chi_T_out, const Field3D &K_in) {
    const BoutReal dx = mesh->GlobalX(1) - mesh->GlobalX(0);
    const BoutReal alpha = (lambda_chi * lambda_chi) / (dx * dx);
    const BoutReal denom = 1.0 + 2.0 * alpha;

    chi_T_out = tau_ac * K_in;
    Field3D chi_new = chi_T_out;

    for (int it = 0; it < helm_iters; ++it) {
      mesh->communicate(chi_T_out);
      BoutReal max_update = 0.0;

      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        // 1D in x; y=z single index
        const BoutReal rhs   = tau_ac * K_in(ix,0,0);
        const BoutReal left  = chi_T_out(ix-1,0,0);
        const BoutReal right = chi_T_out(ix+1,0,0);

        const BoutReal newval = (rhs + alpha * (left + right)) / denom;
        max_update = std::max(max_update, fabs(newval - chi_T_out(ix,0,0)));
        chi_new(ix,0,0) = newval;
      }

      chi_T_out = chi_new;
      mesh->applyBoundary(chi_T_out); // natural Neumann on chi_T
      if (max_update < helm_tol) break;
    }
  }

  // --- Enforce TRUE fixed-flux BC on T by setting ghost cells each step
  //     Q = - (chi0 + chi_T) * dT/dx  =>  dT/dx = -Q / chi_tot
  void apply_fixed_flux_bc_T() {
    Mesh *m = mesh;
    Coordinates *c = m->getCoordinates();
    const int xs = m->xstart; // first interior
    const int xe = m->xend;   // last interior

    for (int z = m->zstart; z <= m->zend; ++z) {
      for (int y = m->ystart; y <= m->yend; ++y) {

        // Left boundary (xmin): Qin > 0 means inflow (into domain)
        BoutReal chi_tot_L = chi_not + chi_T(xs, y, z);
        if (fabs(chi_tot_L) < chit_floor)
          chi_tot_L = (chi_tot_L >= 0 ? chit_floor : -chit_floor);

        BoutReal dTdx_L = - Qin / chi_tot_L;           // desired gradient at boundary
        BoutReal dxL    = c->dx(xs, y, z);
        // set left ghost (xs-1) so that (T(xs)-T(xs-1))/dxL = dTdx_L
        T(xs - 1, y, z) = T(xs, y, z) - dTdx_L * dxL;

        // Right boundary (xmax): Qout > 0 means outflow (leaving domain)
        BoutReal chi_tot_R = chi_not + chi_T(xe, y, z);
        if (fabs(chi_tot_R) < chit_floor)
          chi_tot_R = (chi_tot_R >= 0 ? chit_floor : -chit_floor);

        BoutReal dTdx_R = - Qout / chi_tot_R;
        BoutReal dxR    = c->dx(xe, y, z);
        // set right ghost (xe+1) so that (T(xe+1)-T(xe))/dxR = dTdx_R
        T(xe + 1, y, z) = T(xe, y, z) + dTdx_R * dxR;
      }
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
    OPTION(options, sigma,       1e-2);
    OPTION(options, lambda_chi,  1e-2);
    OPTION(options, helm_iters,  50);
    OPTION(options, helm_tol,    1e-10);

    // True fixed-flux controls
    OPTION(options, use_true_fixed_flux, false);
    OPTION(options, Qin,  1.0); // read from [hw]
    OPTION(options, Qout, 1.0);

    SOLVE_FOR(T, K);
    SAVE_REPEAT(chi_T);
    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, K);

    // --- Build chi_T using selected mode ---
    const bool use_local = (sigma < 1e-16) && (lambda_chi < 1e-16);
    const bool use_pde   = (lambda_chi >= 1e-16);

    if (use_local) {
      chi_T = tau_ac * K;
    } else if (use_pde) {
      solve_helmholtz_chi(chi_T, K);
    } else {
      // Gaussian smoothing (legacy; local domain unless halos widened)
      chi_T = 0.0;
      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        const BoutReal xi = mesh->GlobalX(ix);
        BoutReal wsum = 0.0, acc = 0.0;
        for (int jx = mesh->xstart; jx <= mesh->xend; ++jx) {
          const BoutReal xj = mesh->GlobalX(jx);
          const BoutReal dx = xi - xj;
          const BoutReal w  = (sigma > 0.0 ? gaussian(dx) : 0.0);
          wsum += w;
          acc  += w * K(jx,0,0);
        }
        chi_T(ix,0,0) = tau_ac * (wsum > 0.0 ? acc/wsum : 0.0);
      }
      chi_T = smooth_x(chi_T);
    }

    // Total diffusivity for T equation
    chi = chi_T + chi_not;

    // --- Enforce boundary conditions on T ---
    // If using TRUE fixed-flux, set ghost cells dynamically from current chi_tot
    if (use_true_fixed_flux) {
      apply_fixed_flux_bc_T();
    } else {
      // else BOUT++ will apply whatever you set in [T] (e.g., constgradient(A))
      mesh->applyBoundary(T);
    }

    // Now safe to take derivatives/operators of T
    ddt(T) = FDDX(chi, DDX(T));

    // K equation (uses updated chi_T and DDX(T))
    ddt(K) = beta * FDDX(chi_T, DDX(K))
           - g * chi_T * DDX(T)
           + gamma * K
           - Kc * pow(K, 1.5);

    return 0;
  }
};

BOUTMAIN(HW);
