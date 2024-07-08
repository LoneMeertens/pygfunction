from matplotlib import pyplot as plt
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 1.0             # Borehole buried depth (m)
    H = 100.0           # Borehole length (m)
    r_b = 0.05          # Borehole radius (m)
    B = 5.0             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # g-Function calculation options
    options = {'nSegments': 1,
               'segment_ratios': None,
               'disp': True,
               'kClusters': 0}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Borefield A
    N_1 = 2
    N_2 = 1
    field_A = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    # Move field 5m north
    for b in field_A: b.y = b.y + B

    # Borefield B
    N_1 = 2
    N_2 = 1
    field_B = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)
    # Slightly change the radius of the boreholes. This tricks pygfunction
    # into classifying the boreholes of fields A and B into different groups
    for b in field_B: b.r_b = b.r_b * 1.0001

    # Borefield C
    N_1 = 2
    N_2 = 2
    field_C = gt.boreholes.rectangle_field(N_1, N_2, B, B, H, D, r_b)

    # -------------------------------------------------------------------------
    # Evaluate g-functions for field C
    # -------------------------------------------------------------------------
    gfunc_C = gt.gfunction.gFunction(
        field_C, alpha, time=time, options=options, method='equivalent', boundary_condition='UHTR')

    # -------------------------------------------------------------------------
    # Evaluate self- and cross- g-functions for fields A and B
    # -------------------------------------------------------------------------
    field_AB = field_A + field_B
    gfunc_AB = gt.gfunction.gFunction(
        field_AB, alpha, options=options, method='equivalent')
    # Number of equivalent boreholes in fields A and B
    nEqBoreholes_A = np.max(gfunc_AB.solver.clusters[:len(field_A)]) + 1
    nEqBoreholes_B = gfunc_AB.solver.nEqBoreholes - nEqBoreholes_A
    # Number of boreholes represented by each equivalent borehole in each group
    nBoreholes_A = np.array([b.nBoreholes for b in gfunc_AB.solver.boreholes[:nEqBoreholes_A]])
    nBoreholes_B = np.array([b.nBoreholes for b in gfunc_AB.solver.boreholes[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B]])
    # Matrix of thermal response factors (eq. borehole to eq. borehole)
    h_AB = gfunc_AB.solver.thermal_response_factors(time, alpha).y[:,:,1:]
    # Weighted sum of the borehole-to-borehole thermal response factors
    g_AA = np.zeros(Nt)
    g_AB = np.zeros(Nt)
    g_BA = np.zeros(Nt)
    g_BB = np.zeros(Nt)
    for k in range(Nt):
        g_AA[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,:nEqBoreholes_A,k]) / np.sum(nBoreholes_A)
        g_AB[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_A)
        g_BA[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,:nEqBoreholes_A,k]) / np.sum(nBoreholes_B)
        g_BB[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_B)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------
    ax = gfunc_C.visualize_g_function().axes[0]
    ax.plot(lntts, g_AA)
    ax.plot(lntts, g_AB)
    ax.plot(lntts, g_AA + g_AB, 'ks')
    ax.legend(['C->C (Ref.)', 'A->A', 'B->A', 'A->A + B->A'])
    plt.tight_layout()

if __name__ == '__main__':
    main()