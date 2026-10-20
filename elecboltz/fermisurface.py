from .symmetry import Symmetry

import numpy as np
import scipy.sparse
from scipy.spatial import KDTree


class IsoEnergySurface:
    """A class to represent an iso-energy surface of a material.

    As an example, the Fermi surface is an iso-energy surface at the
    Fermi energy (i.e., the chemical potential at zero temperature).

    Parameters
    ----------
    kpoints : np.ndarray
        The discretized k-points on the Fermi surface. Each row
        corresponds to a k-point in the form ``[kx, ky, kz]``.
    kfaces : np.ndarray
        The faces of the triangulated surface in k-space. Each row
        corresponds to a face in the form ``[i, j, k]``, where
        ``i``, ``j``, and ``k`` are the indices of the vertices
        of the face in the ``kpoints`` array.
    
    Attributes
    ----------
    periodic_projector : scipy.sparse.csr_array
        Projects quantities into the periodic k-space, where points that
        are periodic images of each other are mapped to the same point.
    symmetry_projector : scipy.sparse.csr_array
        Projects scalar quantities from the symmetrized surface onto the
        irreducible wedge. Each column corresponds to a point on this
        surface and contains one nonzero entry for its equivalent wedge
        point.
    n_irreducible : int
        The number of the irreducible wedge's points. The indices
        of the points on the irreducible wedge range from 0 to this
        value - 1.
    """
    def __init__(self, kpoints: np.ndarray, kfaces: np.ndarray):
        self.kpoints = kpoints
        self.kfaces = kfaces
        self.periodic_projector = None
        self.symmetry_projector = None
        self.n_irreducible = -1
        self._compute_adaptive_tolerance()
        self.clean()

    def clean(self):
        """Remove duplicate points and zero-area faces."""
        self._remove_duplicate_points()
        self._remove_zero_area_faces()
        self._remove_orphan_points()

    def symmetrize(self, symmetry: Symmetry):
        """Symmetrize the surface using the provided point group symmetry.
        
        Parameters
        ----------
        symmetry
            The point group symmetry to use for symmetrization.
        """
        # Cut down to the irreducible wedge
        for plane_normal in symmetry.wedge_planes:
            self._clip(plane_normal)
        wedge_kpoints = self.kpoints.copy()
        # Apply the symmetry operations to generate the full surface
        new_kpoints = []
        new_kfaces = []
        for matrix in symmetry.matrices:
            transformed_kpoints = self.kpoints @ matrix.T
            transformed_kfaces = self.kfaces.copy()
            # Ensure the face orientation is preserved after transformation
            if np.linalg.det(matrix) < 0:
                transformed_kfaces = transformed_kfaces[:, [0, 2, 1]]
            new_kpoints.append(transformed_kpoints)
            new_kfaces.append(
                transformed_kfaces + len(self.kpoints) * (len(new_kpoints)-1))
        # Stitch together the transformed k-points and faces
        self.kpoints = np.vstack(new_kpoints)
        self.kfaces = np.vstack(new_kfaces)
        self.clean()

        self.n_irreducible = len(wedge_kpoints)
        self._build_symmetry_projector(wedge_kpoints, symmetry.matrices)

    def sort_and_reindex(self, sort_axis):
        """
        Sort the k-points along a specified axis and reindex the faces
        accordingly.
        """
        new_order = np.argsort(self.kpoints[:, sort_axis])
        old_to_new_map = np.empty(len(new_order), dtype=int)
        old_to_new_map[new_order] = np.arange(len(new_order))
        self.kfaces = old_to_new_map[self.kfaces]
        self.kpoints = self.kpoints[new_order]

    def apply_periodicity(self, gvec, periodic_axes):
        """
        Find duplicate points on the periodic boundaries, then make the
        periodic mesh arrays.

        Parameters
        ----------
        gvec : np.ndarray
            The reciprocal lattice vectors of the material.
        periodic_axes : list[int]
            The axes along which the surface is periodic. For example,
            if the surface is periodic along the x and y axes, then
            ``periodic_axes`` should be ``[0, 1]``.
        """
        duplicates = dict()
        for axis in periodic_axes:
            low_border = np.argwhere(
                self.kpoints[:, axis] + gvec[axis] < self.tol_weld).ravel()
            high_border = np.argwhere(
                self.kpoints[:, axis] - gvec[axis] > -self.tol_weld).ravel()
            if len(low_border) == 0 or len(high_border) == 0:
                continue

            k1 = self.kpoints[low_border][None, :]
            k2 = self.kpoints[high_border][:, None]
            kdiff = k2 - k1
            kdiff[:, :, axis] += gvec[axis]
            kdiff[:, :, axis] %= 2 * gvec[axis]
            kdiff[:, :, axis] -= gvec[axis]
            kdiff = np.linalg.norm(kdiff, axis=-1)

            min_pair = np.argmin(kdiff, axis=1)
            is_duplicate = kdiff[np.arange(len(high_border)), min_pair
                                 ] < self.tol_weld
            duplicates.update(dict(zip(
                high_border[is_duplicate],
                low_border[min_pair[is_duplicate]])))
        self._build_periodic_projector(duplicates)

    def _compute_adaptive_tolerance(self):
        edges = (self.kpoints[np.roll(self.kfaces, -1, axis=1)]
                 - self.kpoints[self.kfaces])
        lengths = np.linalg.norm(edges, axis=-1).ravel()

        valid_lengths = lengths[lengths > 0]
        mean_l = np.median(valid_lengths)
        
        self.tol_snap = 1e-3 * mean_l
        self.tol_weld = 1e-5 * mean_l
        self.tol_area = 1e-6 * mean_l**2

    def _remove_duplicate_points(self):
        tree = KDTree(self.kpoints)
        pairs = tree.query_pairs(self.tol_weld, output_type='ndarray')
        if len(pairs) > 0:
            # Build sparse adjacency graph and find connected components
            adj = scipy.sparse.csr_array(
                (np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])),
                shape=(self.kpoints.shape[0], self.kpoints.shape[0]))
            n_unique, labels = scipy.sparse.csgraph.connected_components(
                adj, directed=False)
            
            # Compute the mean coordinate for each unique vertex cluster
            kpoints_unique = np.zeros((n_unique, 3), dtype=np.float64)
            np.add.at(kpoints_unique, labels, self.kpoints)
            counts = np.bincount(labels, minlength=n_unique)[:, None]
            kpoints_unique /= counts

            self.kpoints = kpoints_unique
            self.kfaces = labels[self.kfaces]

    def _remove_zero_area_faces(self):
        # Drop collapsed faces
        self.kfaces = np.array(
            [face for face in self.kfaces if len(set(face)) == 3])
        # Remove zero-area faces (collinear points)
        areas = np.linalg.norm(np.cross(
            self.kpoints[self.kfaces[:, 1]] - self.kpoints[self.kfaces[:, 0]],
            self.kpoints[self.kfaces[:, 2]] - self.kpoints[self.kfaces[:, 0]]),
            axis=1) / 2
        self.kfaces = self.kfaces[areas > self.tol_area]

    def _remove_orphan_points(self):
        referenced_points = np.unique(self.kfaces)
        # Set all non-referenced points to -1
        # and reindex the referenced points
        reindex = np.full(len(self.kpoints), -1, dtype=int)
        reindex[referenced_points] = np.arange(len(referenced_points))
        
        self.kpoints = self.kpoints[referenced_points]
        self.kfaces = reindex[self.kfaces]

    def _build_periodic_projector(self, duplicates):
        """
        Build the periodic kpoints and kfaces arrays by removing
        duplicate points and reindexing.
        """
        if not duplicates:
            self.periodic_projector = scipy.sparse.eye(
                len(self.kpoints), format='csr')
        else:
            unique_mask = np.full(len(self.kpoints), True)
            unique_mask[list(duplicates.keys())] = False
            reindex_map = np.cumsum(unique_mask) - 1
            reindex_map[list(duplicates.keys())] = reindex_map[
                list(duplicates.values())]
            self.periodic_projector = scipy.sparse.csr_array(
                (np.ones(len(self.kpoints)),
                (reindex_map, np.arange(len(self.kpoints)))),
                shape=(np.count_nonzero(unique_mask), len(self.kpoints)))

    def _build_symmetry_projector(self, wedge_kpoints, matrices):
        """Build the projector from the full mesh to the irreducible wedge."""
        wedge_tree = KDTree(wedge_kpoints)
        nearest_distance = np.full(len(self.kpoints), np.inf)
        wedge_indices = np.full(len(self.kpoints), -1, dtype=int)

        for matrix in matrices:
            distances, indices = wedge_tree.query(self.kpoints @ matrix)
            improve = distances < nearest_distance
            nearest_distance[improve] = distances[improve]
            wedge_indices[improve] = indices[improve]

        if np.any(nearest_distance > self.tol_snap):
            raise RuntimeError(
                "Could not map every symmetrized point to the irreducible "
                "wedge.")

        self.symmetry_projector = scipy.sparse.csr_array(
            (np.ones(len(self.kpoints)),
             (wedge_indices, np.arange(len(self.kpoints)))),
            shape=(len(wedge_kpoints), len(self.kpoints)))

    def _clip(self, plane_normal):
        distance = np.dot(self.kpoints, plane_normal)
        self._snap_to_plane(distance, plane_normal)

        is_vertex_inside = distance[self.kfaces] >= 0
        inside_vertex_count = np.sum(is_vertex_inside, axis=1)

        # No faces are cut
        if (not np.any(inside_vertex_count == 1)
                and not np.any(inside_vertex_count == 2)):
            self.kfaces = self.kfaces[inside_vertex_count == 3]
            self._remove_orphan_points()
            return

        intersections, intersection_points = self._get_intersections(distance)
        new_faces = self._cut_faces(
            intersections, is_vertex_inside, inside_vertex_count)

        self.kpoints = np.vstack((self.kpoints, intersection_points))
        self.kfaces = np.concatenate(
            [face_group for face_group in new_faces if len(face_group)]
            ).astype(int)
        self.clean()

    def _snap_to_plane(self, distance, plane_normal):
        snap_mask = np.abs(distance) < self.tol_snap
        self.kpoints[snap_mask] -= distance[snap_mask][:, None] * plane_normal
        distance[snap_mask] = 0.0

    def _get_intersections(self, distance):
        is_point_inside = distance >= 0
        unique_edges, unique_edge_index = self._get_unique_edges()
        # If one vertex is inside and the other is outside
        # then the edge crosses the plane
        is_edge_vertex_inside = is_point_inside[unique_edges]
        is_edge_crossing = np.logical_xor(
            is_edge_vertex_inside[:, 0],
            is_edge_vertex_inside[:, 1])

        crossing_edges = unique_edges[is_edge_crossing]
        crossing_index = np.full(len(unique_edges), -1, dtype=int)

        if len(crossing_edges) > 0:
            start, end = crossing_edges[:, 0], crossing_edges[:, 1]
            fraction = distance[start] / (distance[start]-distance[end])
            intersection_points = (self.kpoints[start] + fraction[:, None]*(
                                   self.kpoints[end] - self.kpoints[start]))
            crossing_index[is_edge_crossing] = \
                len(self.kpoints) + np.arange(len(crossing_edges))
        else:
            intersection_points = np.empty((0, 3), dtype=float)

        intersections = crossing_index[unique_edge_index].reshape(
            self.kfaces.shape)
        return intersections, intersection_points

    def _get_unique_edges(self):
        edges = np.stack((self.kfaces, np.roll(self.kfaces, -1, axis=1)),
                         axis=-1)
        non_directed_edges = np.sort(edges, axis=-1).reshape(-1, 2)

        unique_edges, unique_edge_index = np.unique(
            non_directed_edges, axis=0, return_inverse=True)
        return unique_edges, unique_edge_index

    def _cut_faces(self, intersections, is_vertex_inside, inside_vertex_count):
        new_faces = []

        # One vertex inside: one output triangle.
        mask = inside_vertex_count == 1
        new_faces.extend([np.stack((
                self.kfaces[mask & is_vertex_inside[:, index], index],
                intersections[mask & is_vertex_inside[:, index], index],
                intersections[mask & is_vertex_inside[:, index], index - 1]),
                axis=1) for index in range(3)])

        # Two vertices inside: one output quadrilateral, triangulated.
        mask = inside_vertex_count == 2
        outside_masks = [mask & ~is_vertex_inside[:, index]
                         for index in range(3)]
        # Triangulate the quadrilateral
        quad_a = np.concatenate([np.stack((
            intersections[outside_masks[index], index],
            self.kfaces[outside_masks[index], (index + 1) % 3],
            self.kfaces[outside_masks[index], (index + 2) % 3]),
            axis=1) for index in range(3)])
        quad_b = np.concatenate([np.stack((
            intersections[outside_masks[index], index],
            self.kfaces[outside_masks[index], (index + 2) % 3],
            intersections[outside_masks[index], index - 1]),
            axis=1) for index in range(3)])
        new_faces.extend((quad_a, quad_b))

        # Fully inside faces are retained unchanged.
        new_faces.append(self.kfaces[inside_vertex_count == 3])
        return new_faces
