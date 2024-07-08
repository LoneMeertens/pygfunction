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
    Nt_geo = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time_geo = gt.utilities.time_geometric(dt, tmax, Nt_geo)
    lntts_geo = np.log(time_geo/ts)

    # ClaessonJaved expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt_cla = 79                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time_cla = gt.utilities.time_ClaessonJaved(dt, tmax, 5)
    lntts_cla = np.log(time_cla/ts)


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
    gfunc_C_geo = gt.gfunction.gFunction(
        field_C, alpha, time=time_geo, options=options, method='equivalent', boundary_condition='UHTR')
    
    gfunc_C_cla = gt.gfunction.gFunction(
        field_C, alpha, time=time_cla, options=options, method='equivalent', boundary_condition='UHTR')

    # -------------------------------------------------------------------------
    # Evaluate self- and cross- g-functions for fields A and B
    # -------------------------------------------------------------------------
    # geometrical time vector
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
    h_AB = gfunc_AB.solver.thermal_response_factors(time_geo, alpha).y[:,:,1:]
    # Weighted sum of the borehole-to-borehole thermal response factors
    g_AA_geo = np.zeros(Nt_geo)
    g_AB_geo = np.zeros(Nt_geo)
    g_BA_geo = np.zeros(Nt_geo)
    g_BB_geo = np.zeros(Nt_geo)
    for k in range(Nt_geo):
        g_AA_geo[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,:nEqBoreholes_A,k]) / np.sum(nBoreholes_A)
        g_AB_geo[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_A)
        g_BA_geo[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,:nEqBoreholes_A,k]) / np.sum(nBoreholes_B)
        g_BB_geo[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_B)

    #Claesson time vector
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
    h_AB = gfunc_AB.solver.thermal_response_factors(time_cla, alpha).y[:,:,1:]
    # Weighted sum of the borehole-to-borehole thermal response factors
    g_AA_cla = np.zeros(Nt_cla)
    g_AB_cla = np.zeros(Nt_cla)
    g_BA_cla = np.zeros(Nt_cla)
    g_BB_cla = np.zeros(Nt_cla)
    for k in range(Nt_cla):
        g_AA_cla[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,:nEqBoreholes_A,k]) / np.sum(nBoreholes_A)
        g_AB_cla[k] = np.sum(nBoreholes_A @ h_AB[:nEqBoreholes_A,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_A)
        g_BA_cla[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,:nEqBoreholes_A,k]) / np.sum(nBoreholes_B)
        g_BB_cla[k] = np.sum(nBoreholes_B @ h_AB[nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,nEqBoreholes_A:nEqBoreholes_A+nEqBoreholes_B,k]) / np.sum(nBoreholes_B)



    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------
    ax = gfunc_C_geo.visualize_g_function().axes[0]
    ax.plot(lntts_geo, g_AA_geo)
    ax.plot(lntts_geo, g_AB_geo)
    ax.plot(lntts_geo, g_AA_geo + g_AB_geo, 'ks')
    ax.legend(['C->C (Ref.)', 'A->A', 'B->A', 'A->A + B->A'])
    plt.tight_layout()
    plt.show

if __name__ == '__main__':
    main()

