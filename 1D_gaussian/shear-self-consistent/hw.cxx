// PhiZonalTK_shear.cxx
#include <bout/derivs.hxx>
#include <bout/physicsmodel.hxx>
#include <cmath>

class PhiZonalTK : public PhysicsModel {
private:
  // Evolved fields
  Field3D T;      // mean profile (stored on mesh)
  Field3D U;      // zonal flow (stored on mesh)
  Field3D phi;    // fluctuations (2D in x,y)

  // Diagnostics
  Field3D ux, uy;     // ExB velocities
  Field3D Rxy;        // y-averaged Reynolds stress
  Field3D Kdiag;      // y-averaged turbulence intensity
  Field3D S;          // shear rate = dU/dx
  Field3D chiT;       // turbulent diffusivity
  Field3D chi;        // total diffusivity

  // Parameters
  BoutReal chi0;                 // baseline diffusivity
  BoutReal tau_ac;               // chiT = tau_ac * K (times shear suppression)
  BoutReal S_star;               // shear-suppression scale
  bool shear_suppress;           // apply Lorentzian suppression?
  BoutReal alpha;                // gradient-drive coefficient
  BoutReal nu_phi, D_phi;        // phi damping + diffusion
  BoutReal mu_U, nu_U;           // U damping + viscosity
  BoutReal C_R;                  // Reynolds-stress efficiency
  BoutReal eps_noise;            // optional tiny symmetry-breaking forcing

  // y-average: returns a field constant in y (and z) but stored as Field3D
  Field3D yavg(const Field3D& f) {
    Field3D out = 0.0;

    for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
      for (int iz = mesh->zstart; iz <= mesh->zend; ++iz) {
        BoutReal sum = 0.0;
        int count = 0;

        for (int iy = mesh->ystart; iy <= mesh->yend; ++iy) {
          sum += f(ix, iy, iz);
          count++;
        }

        const BoutReal mean =
            (count > 0) ? (sum / static_cast<BoutReal>(count)) : 0.0;

        for (int iy = mesh->ystart; iy <= mesh->yend; ++iy) {
          out(ix, iy, iz) = mean;
        }
      }
    }
    return out;
  }

protected:
  int init(bool /*restarting*/) override {
    auto* opt = Options::getRoot()->getSection("hw");

    // Transport closure
    OPTION(opt, chi0, 1e-5);
    OPTION(opt, tau_ac, 1.0);
    OPTION(opt, S_star, 1.0);
    OPTION(opt, shear_suppress, true);  // user requested default true

    // phi dynamics
    OPTION(opt, alpha,  1.0);
    OPTION(opt, nu_phi, 0.5);
    OPTION(opt, D_phi,  1e-3);

    // mean flow dynamics
    OPTION(opt, mu_U, 0.1);
    OPTION(opt, nu_U, 1e-5);

    // RS efficiency
    OPTION(opt, C_R, 1.0);

    // Optional deterministic symmetry-breaking forcing (0 disables)
    OPTION(opt, eps_noise, 0.0);

    SOLVE_FOR(T, U, phi);

    // Save diagnostics
    SAVE_REPEAT(chi);
    SAVE_REPEAT(chiT);
    SAVE_REPEAT(Kdiag);
    SAVE_REPEAT(Rxy);
    SAVE_REPEAT(S);
    SAVE_REPEAT(U);
    SAVE_REPEAT(T);
    SAVE_REPEAT(phi);

    return 0;
  }

  int rhs(BoutReal /*time*/) override {
    mesh->communicate(T, U, phi);

    // ------------------------------------------------------------
    // 1) Fluctuation velocities from phi
    // ------------------------------------------------------------
    ux = -DDY(phi);
    uy =  DDX(phi);

    // Reynolds stress (y-avg), with efficiency factor
    Rxy = C_R * yavg(ux * uy);

    // ------------------------------------------------------------
    // 2) Turbulence intensity: K = <|grad phi|^2>_y
    // ------------------------------------------------------------
    Field3D dphix = DDX(phi);
    Field3D dphiy = DDY(phi);
    Kdiag = yavg(dphix*dphix + dphiy*dphiy);

    // ------------------------------------------------------------
    // 3) Shear rate S = dU/dx
    // ------------------------------------------------------------
    S = DDX(U);

    // ------------------------------------------------------------
    // 4) Turbulent diffusivity with optional Lorentzian shear suppression
    //     chiT = tau_ac * K * 1/(1 + (S/S_star)^2)
    // ------------------------------------------------------------
    Field3D suppress = 1.0;
    if (shear_suppress) {
      // avoid division by zero
      const BoutReal Sstar_safe = (std::abs(S_star) < 1e-16) ? 1e-16 : S_star;
      suppress = 1.0 / (1.0 + SQ(S / Sstar_safe));
    }
    chiT = tau_ac * Kdiag * suppress;
    chi  = chi0 + chiT;

    mesh->communicate(chi);

    // ------------------------------------------------------------
    // 5) phi equation:
    //     dphi/dt + U dphi/dy = alpha (dT/dx) phi - nu_phi phi + D_phi Laplacian(phi)
    // ------------------------------------------------------------
    Field3D Tx = DDX(T);

    ddt(phi) =
        - U * DDY(phi)
        + alpha * Tx * phi
        - nu_phi * phi
        + D_phi * (D2DX2(phi) + D2DY2(phi));

    // optional tiny deterministic seed (keeps symmetry from pinning RS to ~0)
    if (eps_noise > 0.0) {
      for (int ix = mesh->xstart; ix <= mesh->xend; ++ix) {
        for (int iy = mesh->ystart; iy <= mesh->yend; ++iy) {
          const BoutReal s = std::sin(0.013 * ix + 0.021 * iy);
          ddt(phi)(ix, iy, 0) += eps_noise * s;
        }
      }
    }

    // ------------------------------------------------------------
    // 6) U equation:
    //     dU/dt = -d/dx(Rxy) - mu_U U + nu_U d2U/dx2
    // ------------------------------------------------------------
    ddt(U) = -DDX(Rxy) - mu_U * U + nu_U * D2DX2(U);

    // ------------------------------------------------------------
    // 7) T equation:
    //     dT/dt = d/dx( chi dT/dx )
    // ------------------------------------------------------------
    ddt(T) = FDDX(chi, DDX(T));

    return 0;
  }
};

BOUTMAIN(PhiZonalTK);