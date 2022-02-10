# Asphalt molecules aggregation in Toluene + Heptane solvent

import pandas as pd
import numpy as np
from numpy import linalg as la
import matplotlib
from sklearn.linear_model import LinearRegression
from scipy import stats

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['axes.linewidth'] = 1.2
matplotlib.rcParams['figure.max_open_warning'] = False

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------

analysis = 'rdf_analysis'  # ['fractal_dimension', 'clusters_analysis', 'rdf_analysis']

ref_epsilon = 10  # unit in ang
minimum_distance = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]  # unit in ang

asphalt_type = 'A2'
asphalt_with_chains = False
asphalt_mass_fraction = 'M20'
fd_analysis = '3D'
rdf_dimension = '3D'

n_frames = dict(nvt=11, npt=11, md=1001)
n_particles = dict(A0=[12, {'M1': 8, 'M5': 40, 'M10': 80, 'M20': 160, 'M30': 240, 'M40': 320, 'M50': 400}, 8],
                   A1=[16, {'M1': 6, 'M5': 33, 'M10': 66, 'M20': 133, 'M30': 200, 'M40': 266, 'M50': 333}, 12],
                   A2=[21, {'M1': 4, 'M5': 21, 'M10': 43, 'M20': 87, 'M30': 131, 'M40': 175, 'M50': 219}, 12])


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def read_data(name):
    n_mols = n_particles[asphalt_type][1][asphalt_mass_fraction]
    n_atoms = n_particles[asphalt_type][0] if asphalt_with_chains else n_particles[asphalt_type][2]
    n_atoms_ring = n_particles[asphalt_type][2]
    line = ''
    for i in range(3):
        line = name.readline()
    line = line.split()
    box = float(line[1])
    line = name.readline()
    coord, cmass_ring, cmass_chain = [], [], []
    for i in range(n_mols):
        xyz, cmx, cmy, cmz = [], 0, 0, 0
        cmx_a, cmy_a, cmz_a, cmx_b, cmy_b, cmz_b, cmx_c, cmy_c, cmz_c = 0, 0, 0, 0, 0, 0, 0, 0, 0
        xx_ref, yy_ref, zz_ref = 0, 0, 0
        for j in range(n_atoms):
            line = name.readline()
            line = line.split()
            xx, yy, zz = float(line[5]), float(line[6]), float(line[7])
            if j == 0:
                xx_ref, yy_ref, zz_ref = xx, yy, zz
            else:
                if (xx - xx_ref) > (box / 2):
                    xx = xx - box
                elif (xx_ref - xx) > (box / 2):
                    xx = xx + box
                if (yy - yy_ref) > (box / 2):
                    yy = yy - box
                elif (yy_ref - yy) > (box / 2):
                    yy = yy + box
                if (zz - zz_ref) > (box / 2):
                    zz = zz - box
                elif (zz_ref - zz) > (box / 2):
                    zz = zz + box
            xyz.append([xx, yy, zz])
            if j < n_atoms_ring:
                cmx, cmy, cmz = cmx + xx, cmy + yy, cmz + zz
            elif j < n_atoms_ring + 3:
                cmx_a, cmy_a, cmz_a = cmx_a + xx, cmy_a + yy, cmz_a + zz
            elif j < n_atoms_ring + 6:
                cmx_b, cmy_b, cmz_b = cmx_b + xx, cmy_b + yy, cmz_b + zz
            else:
                cmx_c, cmy_c, cmz_c = cmx_c + xx, cmy_c + yy, cmz_c + zz
        coord.append(xyz)
        cmass_ring.append([cmx / n_atoms_ring, cmy / n_atoms_ring, cmz / n_atoms_ring])
        if asphalt_with_chains:
            cmass_chain.append([[cmx_a / 3, cmy_a / 3, cmz_a / 3],
                                [cmx_b / 3, cmy_b / 3, cmz_b / 3],
                                [cmx_c / 3, cmy_c / 3, cmz_c / 3]])
    for i in range(2):
        line = name.readline()
    return box, coord, cmass_ring, cmass_chain


def find_clusters(box, cmass, eps):
    n_mols = n_particles[asphalt_type][1][asphalt_mass_fraction]
    clusters = []
    mols_clustered = []
    mol = 0
    while len(mols_clustered) < n_mols:
        is_cluster_done = False
        temporary_cluster, n = [mol], 0
        while not is_cluster_done:
            i = temporary_cluster[n]
            for j in range(n_mols):
                already_counted = False
                if i == j or j in temporary_cluster:
                    continue
                for cluster in clusters:
                    if j in cluster:
                        already_counted = True
                        break
                if already_counted:
                    continue
                rdx = np.abs(cmass[i][0] - cmass[j][0])
                rdy = np.abs(cmass[i][1] - cmass[j][1])
                rdz = np.abs(cmass[i][2] - cmass[j][2])
                rdx = rdx - np.int(rdx / (box / 2)) * box / 2
                rdy = rdy - np.int(rdy / (box / 2)) * box / 2
                rdz = rdz - np.int(rdz / (box / 2)) * box / 2
                dist = np.sqrt(rdx ** 2 + rdy ** 2 + rdz ** 2)
                if dist <= eps:
                    temporary_cluster.append(j)
            n += 1
            if n == len(temporary_cluster):
                clusters.append(temporary_cluster)
                is_cluster_done = True
        # ---------------------------------------------
        for j in clusters[-1]:
            mols_clustered.append(j)
        while mol in mols_clustered:
            mol += 1
    return clusters


def gyration_tensor(cluster, coord):
    n, n_atoms = len(cluster), n_particles[asphalt_type][0]
    cmx, cmy, cmz, atoms = 0, 0, 0, []
    for mol in cluster:
        k = 0
        for atom in coord[mol]:
            cmx, cmy, cmz = cmx + atom[0], cmy + atom[1], cmz + atom[2]
            atoms.append([atom[0], atom[1], atom[2]])
            k += 1
    cmass = [cmx / (n * n_atoms), cmy / (n * n_atoms), cmz / (n * n_atoms)]
    tensor = []
    for i in range(3):
        s_i = []
        for j in range(3):
            s_ij = 0
            for k in range(n * n_atoms):
                s_ij += (atoms[k][i] - cmass[i]) * (atoms[k][j] - cmass[j])
            s_i.append(s_ij / n)
        tensor.append(s_i)
    tensor = np.array(tensor)
    return tensor


def shape_metrics(tensor):
    w, v = la.eig(tensor)
    w = np.real_if_close(w, 100)
    l1, l2, l3 = min(w[0], w[1], w[2]), 0, max(w[0], w[1], w[2])
    for i in w:
        l2 = i if i not in (l1, l3) else l2
    # rg: radius of gyration
    rg2 = l1 + l2 + l3
    rg = np.sqrt(rg2)
    # b: asphericity
    b = l3 - 0.5 * (l2 + l1)
    # c: acylindricity
    c = l2 - l1
    # k: anisothropy
    k2 = (b ** 2 + 0.75 * (c ** 2)) / (rg2 ** 2)
    return rg2, rg, b / rg2, c / rg2, k2


def clusters_shape(t, clusters, coord):
    n_mols = n_particles[asphalt_type][1][asphalt_mass_fraction]
    n = len(clusters)
    sum_1, sum_2 = 0, 0
    size_largest = 0
    rg_largest, b_largest, c_largest, k2_largest = 0, 0, 0, 0
    clusters_1, clusters_2, clusters_3, clusters_more = 0, 0, 0, 0
    rg_distribution = []
    for cluster in clusters:
        if len(cluster) == 1:
            clusters_1 += 1
        elif len(cluster) == 2:
            clusters_2 += 1
        elif len(cluster) == 3:
            clusters_3 += 1
        else:
            clusters_more += 1
        i, n_i = len(cluster), 1
        s_tensor = gyration_tensor(cluster, coord)
        rg2, rg, norm_b, norm_c, k2 = shape_metrics(s_tensor)
        rg_distribution.append(rg)
        if len(cluster) > size_largest:
            size_largest = len(cluster)
            rg_largest, b_largest, c_largest, k2_largest = rg, norm_b, norm_c, k2
        sum_1 += n_i * np.square(i) * rg2
        sum_2 += n_i * np.square(i)
    mean_rg = np.sqrt(sum_1 / sum_2) if sum_2 != 0 else 0
    mean_gz2 = sum_2 / n if n != 0 else 0
    percentage_largest = (size_largest / n_mols) * 100
    return [t, n, clusters_1, clusters_2, clusters_3, clusters_more, percentage_largest, mean_rg, mean_gz2,
            rg_largest, b_largest, c_largest, k2_largest]


def clusters_histogram(t, clusters):
    hist = [0] * 21
    hist[0] = t
    for cluster in clusters:
        n = len(cluster)
        if n < 10:
            hist[2*n-1] += 1
            hist[2*n] += n
        else:
            hist[19] += 1
            hist[20] += n
    return hist


def time_elapse(sim, frm, t):
    if sim == 'nvt':
        dt = 2.5e-9 if frm < n_frames[sim] - 1 else 50e-9
    elif sim == 'npt':
        dt = 50e-9 if frm < n_frames[sim] - 1 else 0.5e-9
    else:
        dt = 0.5e-9
    t += dt
    return t


def fractal_dimension(box, cmass, eps):
    n_mols, n_atoms = n_particles[asphalt_type][1][asphalt_mass_fraction], n_particles[asphalt_type][0]
    corr_sum = 0
    for i in range(n_mols):
        for j in range(i, n_mols):
            if i == j:
                continue
            rdx = np.abs(cmass[i][0] - cmass[j][0])
            rdy = np.abs(cmass[i][1] - cmass[j][1])
            rdz = np.abs(cmass[i][2] - cmass[j][2])
            rdx = rdx - np.int(rdx / (box / 2)) * box / 2
            rdy = rdy - np.int(rdy / (box / 2)) * box / 2
            rdz = rdz - np.int(rdz / (box / 2)) * box / 2
            if fd_analysis == '3D':
                dist = np.sqrt(rdx ** 2 + rdy ** 2 + rdz ** 2)
            else:
                dist = np.sqrt(rdx ** 2 + rdy ** 2)
            heaviside = 1 if eps >= dist else 0
            corr_sum += heaviside
    corr_sum = corr_sum / (n_mols * (n_mols - 1))
    return corr_sum


def linear_regression(config_df):
    x_list = config_df['Log(eps/ref_eps)'].to_list()
    y_list = config_df['Log(C_eps)'].to_list()
    x_1, x_2 = np.array(x_list[9:18]).reshape(-1, 1), np.array(x_list[27:56]).reshape(-1, 1)
    y_1, y_2 = np.array(y_list[9:18]), np.array(y_list[27:56])
    # -----------------
    slope_1 = slope_calculation(x_1, y_1)
    slope_2 = slope_calculation(x_2, y_2)
    return slope_1, slope_2


def slope_calculation(_x, _y):
    linear_model = LinearRegression()
    linear_model.fit(_x, _y)
    _slope = linear_model.coef_[0]
    return _slope


def rdf_calc(ref, sel, cmass_ring, cmass_chain, rdf_dim):
    cmass_ref = cmass_ring if ref == 'aromatic' else cmass_chain
    cmass_sel = cmass_ring if sel == 'aromatic' else cmass_chain
    rdf_temp = [0] * n_bin
    for i, com_ref in enumerate(cmass_ref):
        for j, com_sel in enumerate(cmass_sel):
            if i != j:
                radius = rdf_distance(ref, sel, com_ref, com_sel, rdf_dim)
                bin_k = np.int((radius - r_min) / del_r)
                if radius < r_max:
                    rdf_temp[bin_k] = rdf_temp[bin_k] + 1
    return rdf_temp


def rdf_distance(ref, sel, com_ref, com_sel, rdf_dim):
    if ref == 'aromatic':
        if sel == 'aromatic':
            del_x = np.abs(com_ref[0] - com_sel[0])
            del_y = np.abs(com_ref[1] - com_sel[1])
            del_z = np.abs(com_ref[2] - com_sel[2])
        elif sel == 'aliphatic_a':
            del_x = np.abs(com_ref[0] - com_sel[0][0])
            del_y = np.abs(com_ref[1] - com_sel[0][1])
            del_z = np.abs(com_ref[2] - com_sel[0][2])
        else:
            del_x = np.abs(com_ref[0] - com_sel[1][0])
            del_y = np.abs(com_ref[1] - com_sel[1][1])
            del_z = np.abs(com_ref[2] - com_sel[1][2])
    else:
        if sel == 'aliphatic_a':
            del_x = np.abs(com_ref[0][0] - com_sel[0][0])
            del_y = np.abs(com_ref[0][1] - com_sel[0][1])
            del_z = np.abs(com_ref[0][2] - com_sel[0][2])
        else:
            del_x = np.abs(com_ref[0][0] - com_sel[1][0])
            del_y = np.abs(com_ref[0][1] - com_sel[1][1])
            del_z = np.abs(com_ref[0][2] - com_sel[1][2])
    del_x = del_x - box_length if del_x > box_length / 2 else del_x
    del_y = del_y - box_length if del_y > box_length / 2 else del_y
    del_z = del_z - box_length if del_z > box_length / 2 else del_z
    radius = np.sqrt(del_x ** 2 + del_y ** 2 + del_z ** 2) if rdf_dim == '3D' else np.sqrt(del_x ** 2 + del_y ** 2)
    return radius


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# analysis 1 (finding fractal dimension of the self assembled networks)
if analysis == 'fractal_dimension':
    configs = []
    file_name = 'asphaltene_md.pdb'
    print('Analyze of md simulations begins!')
    with open(file_name) as data_asphalt:
        nf = 0
        for frame in range(n_frames['md']):
            box_length, coordinates, com_rings, com_chains,  = read_data(data_asphalt)
            if frame >= 901:
                configs.append([])
                print(nf) if (np.mod(nf, 10) == 0) else True
                for epsilon in range(1, 120):
                    c_eps = fractal_dimension(box_length, com_rings, epsilon)
                    configs[nf].append([np.log(epsilon * 1.0 / ref_epsilon), np.log(c_eps)])
                nf += 1

    # output
    d_fractal = []
    writer = pd.ExcelWriter(f'analysisFD_{asphalt_type}_{asphalt_mass_fraction}_{fd_analysis}.xls')
    for frame in range(nf):
        configs_md = pd.DataFrame(configs[frame], columns=['Log(eps/ref_eps)', 'Log(C_eps)'])
        configs_md.to_excel(writer, f'{frame + 1}') if (frame >= 90) else True
        df_1, df_2 = linear_regression(configs_md)
        d_fractal.append([df_1, df_2])
    d_fractal_md = pd.DataFrame(d_fractal, columns=['region 1', 'region 2'])
    d_fractal_md.to_excel(writer, 'fd_slopes')
    writer.save()

# analysis 2 (finding clusters shape metrics)
if analysis == 'clusters_analysis':
    time = 0
    configsH, configsS = [], []
    for simulation in ['nvt', 'npt', 'md']:
        file_name = 'asphaltene_{}.pdb'.format(simulation)
        print('Analyze of {} simulations begins!'.format(simulation))
        with open(file_name) as data_asphalt:
            for frame in range(n_frames[simulation]):
                if (file_name in ['asphaltene_npt.pdb', 'asphaltene_md.pdb']) and (frame == 0):
                    continue
                print(frame) if (file_name == 'asphaltene_md.pdb') and (np.mod(frame, 100) == 0) else True
                box_length, coordinates, com_rings, com_chains = read_data(data_asphalt)
                for md_index, epsilon in enumerate(minimum_distance):
                    configsH.append([]) if frame == 0 else True
                    all_clusters = find_clusters(box_length, com_rings, epsilon)
                    config_hist = clusters_histogram(time, all_clusters)
                    configsH[md_index].append(config_hist)
                    config_shape = clusters_shape(time, all_clusters, coordinates)
                    configsS[md_index].append(clusters_shape)
                time = time_elapse(simulation, frame, time)

    # output
    suffix = 'WithChain' if asphalt_with_chains else 'WithoutChain'
    writer = pd.ExcelWriter(f'analysisHIST_{asphalt_type}_{asphalt_mass_fraction}_{suffix}.xls')
    for md_index, epsilon in enumerate(minimum_distance):
        configs_mdH = pd.DataFrame(configsH[md_index],
                                   columns=['time',
                                            'length 1', '# mol 1', 'length 2', '# mol 2', 'length 3', '# mol 3',
                                            'length 4', '# mol 4', 'length 5', '# mol 5', 'length 6', '# mol 6',
                                            'length 7', '# mol 7', 'length 8', '# mol 8', 'length 9', '# mol 9',
                                            'length 10', '# mol 10'])
        configs_mdS = pd.DataFrame(configsS[md_index],
                                   columns=['time', '# clusters',
                                            'singlets', 'doublets', 'triplets', 'larger clusters',
                                            '% of molecules in the largest cluster',
                                            'average gyration radius', 'average size of clusters',
                                            'largest cluster gyration radius',
                                            'largest cluster normalized asphericity',
                                            'largest cluster normalized acylindericity',
                                            'largest cluster anisothropy'])
        configs_mdS.to_excel(writer, f'{epsilon}')
    writer.save()

# analysis 3 (rdf calculations)
if analysis == 'rdf_analysis':
    r_min, r_max, del_r = 0.0, 58.0, 0.1
    n_bin = np.int((r_max - r_min) / del_r)
    bins_list = []
    nf = n_frames['md']
    file_name = 'asphaltene_md.pdb'
    print('Analyze of md simulations begins!')
    with open(file_name) as data_asphalt:
        box_size = 0
        for frame in range(nf):
            print(frame) if np.mod(frame, 20) == 0 else True
            box_length, coordinates, com_rings, com_chains = read_data(data_asphalt)
            box_size += box_length
            if frame == 0:
                rdf_r_r = [0] * n_bin
                if asphalt_with_chains:
                    rdf_r_a = [0] * n_bin
                    rdf_r_b = [0] * n_bin
                    rdf_a_a = [0] * n_bin
                    rdf_a_b = [0] * n_bin
            # ----------------
            rdf_r_r_temp = rdf_calc('aromatic', 'aromatic', com_rings, com_chains, rdf_dimension)
            if asphalt_with_chains:
                rdf_r_a_temp = rdf_calc('aromatic', 'aliphatic_a', com_rings, com_chains, rdf_dimension)
                rdf_r_b_temp = rdf_calc('aromatic', 'aliphatic_b', com_rings, com_chains, rdf_dimension)
                rdf_a_a_temp = rdf_calc('aliphatic_a', 'aliphatic_a', com_rings, com_chains, rdf_dimension)
                rdf_a_b_temp = rdf_calc('aliphatic_a', 'aliphatic_b', com_rings, com_chains, rdf_dimension)
            # ----------------
            rdf_r_r = [x + y for (x, y) in zip(rdf_r_r, rdf_r_r_temp)]
            if asphalt_with_chains:
                rdf_r_a = [x + y for (x, y) in zip(rdf_r_a, rdf_r_a_temp)]
                rdf_r_b = [x + y for (x, y) in zip(rdf_r_b, rdf_r_b_temp)]
                rdf_a_a = [x + y for (x, y) in zip(rdf_a_a, rdf_a_a_temp)]
                rdf_a_b = [x + y for (x, y) in zip(rdf_a_b, rdf_a_b_temp)]
        box_size = box_size / nf
        for k in range(n_bin):
            bin_r = r_min + (k + 1) * del_r
            bins_list.append(k * del_r)
            bin_size = (4/3) * np.pi * ((bin_r**3)-((bin_r-del_r)**3)) if rdf_dimension == '3D' else \
                2 * np.pi * bin_r * del_r
            rdf_r_r[k] = ((rdf_r_r[k] / bin_size) / (len(com_rings) * len(com_rings) / (box_size**3))) / nf
            if asphalt_with_chains:
                rdf_r_a[k] = ((rdf_r_a[k] / bin_size) / (len(com_rings) * len(com_chains) / (box_size**3))) / nf
                rdf_r_b[k] = ((rdf_r_b[k] / bin_size) / (len(com_rings) * len(com_chains) / (box_size**3))) / nf
                rdf_a_a[k] = ((rdf_a_a[k] / bin_size) / (len(com_chains) * len(com_chains) / (box_size**3))) / nf
                rdf_a_b[k] = ((rdf_a_b[k] / bin_size) / (len(com_chains) * len(com_chains) / (box_size**3))) / nf

    # output
    writer = pd.ExcelWriter(f'analysisRDF_{asphalt_type}_{asphalt_mass_fraction}_{rdf_dimension}.xls')
    if asphalt_with_chains:
        rdf_names, rdf_lists = ['r_r', 'r_a', 'r_b', 'a_a', 'a_b'], [rdf_r_r, rdf_r_a, rdf_r_b, rdf_a_a, rdf_a_b]
    else:
        rdf_names, rdf_lists = ['r_r'], [rdf_r_r]
    for rdf_name, rdf_list in zip(rdf_names, rdf_lists):
        rdf_df = pd.DataFrame(bins_list, columns=['radius'])
        rdf_df[f'rdf{rdf_name}'] = rdf_list
        rdf_df.to_excel(writer, f'{rdf_name}')
    writer.save()


# --------------------------------------------------------------------------------------------------------------------
# THE END
# --------------------------------------------------------------------------------------------------------------------
print('=> DONE!')
