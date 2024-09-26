import open3d as o3d
import numpy as np


"""--------------------
RANSAC plane detection
-----------------------"""

def detect_planes_from_pcd(df, distance_threshold=30, ransac_n=3, min_points=200, num_iterations=1000, max_angle=80):
    """
    Cette fonction prend un df au format lidar qui contient des points lidar X,Y,H
    (à priori d'une instance particulière de batiment )
    retourne une liste de plan et le df avec pour chaque point l'identification du plan auquel il est rattaché)
    :param df: lidar dataframe contenant au minimum les points lidar X,Y,H
    :param distance_threshold: min distance point can have with plane
    :param ransac_n: 3 (coeff ransac)
    :param min_points: minimum number of points to build a plane
    :param num_iterations: coeff ransac
    :param max_angle: max angle to build a plane
    :return: df with plane_id column added (copied from original dataframe)
    list of planes : tuples (plan equation coeff list, number of linked points)
       ex: array([ 3.08946962e-01, -4.97514806e-01,  8.10574360e-01,  3.17330189e+08]), 1223)

    """
    # Conversion du DataFrame en pcd Open3D
    df = df.copy()
    df.reset_index(drop=True, inplace=True)  # les indexes doivent être propres

    points = df[['X', 'Y', 'Z']].to_numpy() # pour la détection des plans c'est Z qu'on veut
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Initialiser les id de plan (0 = pas de plan)
    df['plane_id'] = 0

    plane_id = 1
    planes = []

    # Garder la trace des indices
    remaining_indices = df.index.to_numpy()

    while len(pcd.points) > 0:
        # Détecter récursivement les plans dans le nuage de points en commençant par le plus grand
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)

        if len(inliers) < min_points:
            # le plan ne contient pas assez de points
            break

        # ajouter le plan à planes (coeef plan et nombre de points), seulement pour inclinaison < max_angle
        # on ne garde pas les verticales (> max angle)
        if calculate_inclination(plane_model) <= max_angle:
            # Enregistrer les indices des inliers dans le DataFrame
            df.loc[remaining_indices[inliers], 'plane_id'] = plane_id
            planes.append((plane_model, len(remaining_indices[inliers])))
            plane_id += 1

        # On enlève les points du plan détecté au pcd
        pcd = pcd.select_by_index(inliers, invert=True)

        # Important ! Mettre à jour les indices restants
        remaining_indices = remaining_indices[~np.isin(remaining_indices, remaining_indices[inliers])]

        if len(pcd.points) < min_points:
            # il n'y a plus assez de points pour faire un plan
            break


    return df, planes



def calculate_inclination(plane_coefficients):
    """ return degree inclination from [a,b,c,d] plan coefficient
    :param plane_coefficients: [a,b,c,d]
    :return: inclination (degree)
    """
    a, b, c, d = plane_coefficients
    normal_vector = np.array([a, b, c])
    norm_normal_vector = np.linalg.norm(normal_vector)

    # Inclinaison en radians par rapport à l'axe z
    theta = np.arccos(c / norm_normal_vector)

    theta_degrees = np.degrees(theta)
    return theta_degrees
