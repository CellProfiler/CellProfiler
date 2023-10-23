import numpy as np

import cellprofiler_library.functions.segmentation as lib_seg

class TestSegmentation:
    def test_01_01_2d_dense_empty_to_sparse(self):
        """
        Test conversion of 2D dense with no labels to sparse

        dense input: labels=1, c=1,t=1,z=1,y=3,x=3

            0 0 0
            0 0 0
            0 0 0
        """
        dense = np.zeros((1, 1, 1, 1, 3, 3))
        indices = np.array([])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        expected_ys = np.array([])
        expected_xs = np.array([])
        expected_labels = np.array([])

        expected_ys_dtype = np.uint16
        expected_xs_dtype = np.uint16
        expected_labels_dtype = np.uint8

        # TODO - 4758: change, x,y,z,label to enum
        np.testing.assert_array_equal(sparse['y'], expected_ys)
        np.testing.assert_array_equal(sparse['x'], expected_xs)
        np.testing.assert_array_equal(sparse['label'], expected_labels)
        np.testing.assert_equal(sparse['y'].dtype, expected_ys_dtype)
        np.testing.assert_equal(sparse['x'].dtype, expected_xs_dtype)
        np.testing.assert_equal(sparse['label'].dtype, expected_labels_dtype)

    def test_01_02_2d_dense_nonoverlap_to_sparse(self):
        """
        Test conversion of 2D dense without overlapping labels to sparse
        
        dense input: labels=1, c=1,t=1,z=1,y=3,x=3

            1 1 0
            2 0 0
            2 0 0
        """
        dense = np.array([1,1,0,2,0,0,2,0,0]).reshape((1,1,1,1,3,3))
        indices = np.array([1,2])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        expected_ys = np.array([0, 0, 1, 2])
        expected_xs = np.array([0, 1, 0, 0])
        expected_labels = np.array([1, 1, 2, 2])

        expected_labels_dtype = np.uint8
        expected_ys_dtype = np.uint16
        expected_xs_dtype = np.uint16

        np.testing.assert_array_equal(sparse['y'], expected_ys)
        np.testing.assert_array_equal(sparse['x'], expected_xs)
        np.testing.assert_array_equal(sparse['label'], expected_labels)
        np.testing.assert_equal(sparse['y'].dtype, expected_ys_dtype)
        np.testing.assert_equal(sparse['x'].dtype, expected_xs_dtype)
        np.testing.assert_equal(sparse['label'].dtype, expected_labels_dtype)

    def test_01_03_2d_dense_overlap_to_sparse(self):
        """
        Test conversion of 2D dense with overlapping labels to sparse

        dense input: labels=2, c=1,t=1,z=1,y=3,x=2

            1 0
            1 0
            0 0
            ---
            0 0
            2 2
            0 0
        """
        dense = np.array([[1,0,1,0,0,0],[0,0,2,2,0,0]]).reshape((2,1,1,1,3,2))
        indices = np.array([[1], [2]])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        expected_ys = np.array([0,1,1,1])
        expected_xs = np.array([0,0,0,1])
        expected_labels = np.array([1,1,2,2])

        expected_ys_dtype = np.uint16
        expected_xs_dtype = np.uint16
        expected_labels_dtype = np.uint8

        np.testing.assert_array_equal(sparse['y'], expected_ys)
        np.testing.assert_array_equal(sparse['x'], expected_xs)
        np.testing.assert_array_equal(sparse['label'], expected_labels)
        np.testing.assert_equal(sparse['y'].dtype, expected_ys_dtype)
        np.testing.assert_equal(sparse['x'].dtype, expected_xs_dtype)
        np.testing.assert_equal(sparse['label'].dtype, expected_labels_dtype)

    def test_01_04_2d_random_dense_to_sparse(self):
        """
        Test conversion of 2D dense with some more complex shapes to sparse

        Make 10 circles that might overlap
        """
        r = np.random.RandomState()
        r.seed(201)
        i, j = np.mgrid[0:50, 0:50]
        dense = np.zeros((10, 1, 1, 1, 50, 50), np.uint32)
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = np.min([min(loc - 1, 49 - loc) for loc in (x_loc, y_loc)])
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            dense[idx, 0, 0, 0, mask] = idx + 1

        indices = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        assert np.sum(dense > 0) == len(sparse)

        retrieval = dense[
            :,
            0,
            0,
            0,
            sparse['y'],
            sparse['x'],
        ]
        matches = (retrieval == sparse['label'][None, :])
        assert np.all(np.sum(matches, 0) == 1)

    def test_01_05_3d_dense_nonoverlap_to_sparse(self):
        """
        Test conversion of 3D dense with no overlapping labels to sparse

        dense input: labels=1, c=1,t=1,z=3,y=3,x=2

               +------+ 
              / 1  0 /|
             / 1  0 / +
            +------+ /
            | 1  0 |/
            +------+
               +------+ 
              / 2  2 /|
             / 2  2 / +
            +------+ /
            | 2  2 |/
            +------+
               +------+ 
              / 0  0 /|
             / 0  0 / +
            +------+ /
            | 0  0 |/
            +------+
        """
        dense = np.array(
            [1,0,2,2,0,0,1,0,2,2,0,0,1,0,2,2,0,0]
        ).reshape((1,1,1,3,3,2))
        indices = np.array([[1], [2]])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        expected_zs = np.array([0,0,0,1,1,1,2,2,2])
        expected_ys = np.array([0,1,1,0,1,1,0,1,1])
        expected_xs = np.array([0,0,1,0,0,1,0,0,1])
        expected_labels = np.array([1,2,2,1,2,2,1,2,2])

        expected_zs_dtype = np.uint16
        expected_ys_dtype = np.uint16
        expected_xs_dtype = np.uint16
        expected_labels_dtype = np.uint8

        np.testing.assert_array_equal(sparse['z'], expected_zs)
        np.testing.assert_array_equal(sparse['y'], expected_ys)
        np.testing.assert_array_equal(sparse['x'], expected_xs)
        np.testing.assert_array_equal(sparse['label'], expected_labels)
        np.testing.assert_equal(sparse['z'].dtype, expected_zs_dtype)
        np.testing.assert_equal(sparse['y'].dtype, expected_ys_dtype)
        np.testing.assert_equal(sparse['x'].dtype, expected_xs_dtype)
        np.testing.assert_equal(sparse['label'].dtype, expected_labels_dtype)

    def test_01_06_3d_dense_noverlap_to_sparse(self):
        """
        Test conversion of 3D dense with overlapping labels to sparse

        dense input: labels=2, c=1,t=1,z=3,y=3,x=2

               +------+  |     +------+
              / 1  1 /|  |    / 0  0 /|
             / 1  1 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 1  1 |/    |  | 0  0 |/  
            +------+     |  +------+   
               +------+  |     +------+
              / 1  1 /|  |    / 0  0 /|
             / 1  1 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 1  1 |/    |  | 0  0 |/  
            +------+     |  +------+   
               +------+  |     +------+
              / 0  0 /|  |    / 0  0 /|
             / 0  0 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 0  0 |/    |  | 0  0 |/  
            +------+     |  +------+   
        """
        dense = np.array(
            [1,1,1,1,0,0,1,1,1,1,0,0,1,1,1,1,0,0,
             0,0,0,0,0,0,2,2,2,2,2,2,0,0,0,0,0,0]
        ).reshape((2,1,1,3,3,2))
        indices = np.array([[1], [2]])
        sparse = lib_seg.convert_dense_to_sparse(dense, indices)

        expected_zs = np.array([0,0,0,0,1,1,1,1,2,2,2,2, 1,1,1,1,1,1])
        expected_ys = np.array([0,0,1,1,0,0,1,1,0,0,1,1, 0,0,1,1,2,2])
        expected_xs = np.array([0,1,0,1,0,1,0,1,0,1,0,1, 0,1,0,1,0,1])
        expected_labels = np.array([1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2])

        expected_zs_dtype = np.uint16
        expected_ys_dtype = np.uint16
        expected_xs_dtype = np.uint16
        expected_labels_dtype = np.uint8

        np.testing.assert_array_equal(sparse['z'], expected_zs)
        np.testing.assert_array_equal(sparse['y'], expected_ys)
        np.testing.assert_array_equal(sparse['x'], expected_xs)
        np.testing.assert_array_equal(sparse['label'], expected_labels)
        np.testing.assert_equal(sparse['z'].dtype, expected_zs_dtype)
        np.testing.assert_equal(sparse['y'].dtype, expected_ys_dtype)
        np.testing.assert_equal(sparse['x'].dtype, expected_xs_dtype)
        np.testing.assert_equal(sparse['label'].dtype, expected_labels_dtype)

    def test_02_01_convert_sparse_to_2d_empty_dense(self):
        """
        Test conversion of sparse to 2D dense with no labels

        dense output: labels=1, c=1,t=1,z=1,y=1,x=1

            0 0 0
            0 0 0
            0 0 0
        """
        sparse_plain = np.array([], dtype=[('label', 'u1')])
        sparse_shaped = np.array([], dtype=[('y', '<u2'), ('x', '<u2'), ('label', 'u1')])

        dense_plain, dpi = lib_seg.convert_sparse_to_dense(sparse_plain)
        dense_shaped, dsi = lib_seg.convert_sparse_to_dense(sparse_shaped, dense_shape=(1, 1, 1, 1, 3, 3))

        dense_plain_expected = np.zeros((1, 1, 1, 1, 1, 1), np.uint8)
        dense_shaped_expected = np.zeros((1, 1, 1, 1, 3, 3), np.uint8)
        indices_expected = np.array([[]], np.uint8)

        np.testing.assert_array_equal(dense_plain, dense_plain_expected)
        np.testing.assert_array_equal(dense_shaped, dense_shaped_expected)
        np.testing.assert_array_equal(dpi, indices_expected)
        np.testing.assert_array_equal(dsi, indices_expected)


    def test_02_02_convert_sparse_to_2d_nooverlap_dense(self):
        """
        Test conversion of sparse to 2D dense with no overlapping labels

        dense output: labels=1, c=1,t=1,z=1,y=3,x=3

            1 1 0
            2 0 0
            2 0 0
        """
        sparse = np.array(
            [(0, 0, 1), (0, 1, 1), (1, 0, 2), (2, 0, 2)],
            dtype=[('y', '<u2'), ('x', '<u2'), ('label', 'u1')]
        )

        dense_plain, dpi = lib_seg.convert_sparse_to_dense(sparse)
        dense_shaped, dsi = lib_seg.convert_sparse_to_dense(
            sparse, dense_shape=(1, 1, 1, 3, 3)
        )

        # + 1 padding
        dense_plain_expected = np.array(
            [1,1,0,2,0,0,2,0,0,0,0,0],
            np.uint8
        ).reshape((1,1,1,1,4,3))
        dense_shaped_expected = np.array(
            [1,1,0,2,0,0,2,0,0],
            np.uint8
        ).reshape((1,1,1,1,3,3))
        indices_expected = np.array([[1,2]], np.uint8)

        np.testing.assert_array_equal(dense_plain, dense_plain_expected)
        np.testing.assert_array_equal(dense_shaped, dense_shaped_expected)
        np.testing.assert_array_equal(dpi, indices_expected)
        np.testing.assert_array_equal(dsi, indices_expected)

    def test_02_03_convert_sparse_to_2d_overlap_dense(self):
        """
        Test conversion of sparse to 2D dense with overlapping labels

        dense output: labels=2, c=1,t=1,z=1,y=3,x=2

            1 0
            1 0
            0 0
            ---
            0 0
            2 2
            0 0
        """
        sparse = np.array(
            [(0, 0, 1), (1, 0, 1), (1, 0, 2), (1, 1, 2)],
            dtype=[('y', '<u2'), ('x', '<u2'), ('label', 'u1')]
        )

        dense_plain, dpi = lib_seg.convert_sparse_to_dense(sparse)
        dense_shaped, dsi = lib_seg.convert_sparse_to_dense(
            sparse, dense_shape=(1, 1, 1, 3, 2)
        )

        # + 1 padding
        dense_plain_expected = np.array(
            [[1,0,0,1,0,0,0,0,0],[0,0,0,2,2,0,0,0,0]],
            np.uint8
        ).reshape((2,1,1,1,3,3))
        dense_shaped_expected = np.array(
            [[1,0,1,0,0,0],[0,0,2,2,0,0]],
            np.uint8
        ).reshape((2,1,1,1,3,2))
        indices_expected = np.array([[1], [2]], np.uint8)

        np.testing.assert_array_equal(dense_plain, dense_plain_expected)
        np.testing.assert_array_equal(dense_shaped, dense_shaped_expected)
        np.testing.assert_array_equal(dpi, indices_expected)
        np.testing.assert_array_equal(dsi, indices_expected)


    def test_02_04_convert_random_sparse_to_dense(self):
        """
        Test conversion of 2D sparse with some more complex shapes to dense

        Make 10 circles that might overlap
        """
        r = np.random.RandomState()
        r.seed(201)
        i, j = np.mgrid[0:50, 0:50]
        ii = []
        jj = []
        vv = []
        for idx in range(10):
            x_loc = r.uniform() * 30 + 10
            y_loc = r.uniform() * 30 + 10
            max_radius = np.min(
                [min(loc - 1, 49 - loc) for loc in (x_loc, y_loc)]
            )
            radius = r.uniform() * (max_radius - 5) + 5
            mask = ((i - y_loc) ** 2 + (j - x_loc) ** 2) <= radius ** 2
            ii.append(i[mask])
            jj.append(j[mask])
            vv.append(np.ones(np.sum(mask), np.uint32) * (idx + 1))

        sparse = np.core.records.fromarrays(
            [np.hstack(x) for x in (ii, jj, vv)],
            [('y', np.uint32, 1), ('x', np.uint32, 1), ('label', np.uint32, 1)]
        )

        dense_shape = (1, 1, 1, 50, 50)
        dense, indices = lib_seg.convert_sparse_to_dense(sparse, dense_shape=dense_shape)
        assert tuple(dense.shape[1:]) == dense_shape
        assert np.sum(dense > 0) == len(sparse)

        retrieval = dense[
            :,
            0,
            0,
            0,
            sparse['y'],
            sparse['x'],
        ]
        matches = (retrieval == sparse['label'][None, :])
        assert np.all(np.sum(matches, 0) == 1)

    def test_02_05_convert_sparse_to_dense_3d_nooverlap(self):
        """
        Test conversion of sparse to 3D dense with no overlapping labels

        dense output: labels=1, c=1,t=1,z=3,y=3,x=2

               +------+ 
              / 1  0 /|
             / 1  0 / +
            +------+ /
            | 1  0 |/
            +------+
               +------+ 
              / 2  2 /|
             / 2  2 / +
            +------+ /
            | 2  2 |/
            +------+
               +------+ 
              / 0  0 /|
             / 0  0 / +
            +------+ /
            | 0  0 |/
            +------+
        """
        sparse = np.array(
            [(0, 0, 0, 1), (0, 1, 0, 2), (0, 1, 1, 2),
             (1, 0, 0, 1), (1, 1, 0, 2), (1, 1, 1, 2),
             (2, 0, 0, 1), (2, 1, 0, 2), (2, 1, 1, 2)],
            dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')]
        )

        dense_plain, dpi = lib_seg.convert_sparse_to_dense(sparse)
        dense_shaped, dsi = lib_seg.convert_sparse_to_dense(
            sparse, dense_shape=(1, 1, 3, 3, 2)
        )

        # + 1 padding
        dense_plain_expected = np.array(
             [1,0,0,2,2,0,0,0,0,
              1,0,0,2,2,0,0,0,0,
              1,0,0,2,2,0,0,0,0,
              0,0,0,0,0,0,0,0,0],
              np.uint8
        ).reshape((1,1,1,4,3,3))
        dense_shaped_expected = np.array(
             [1,0,2,2,0,0,
              1,0,2,2,0,0,
              1,0,2,2,0,0],
              np.uint8
        ).reshape((1,1,1,3,3,2))
        indices_expected = np.array([[1, 2]], np.uint8)

        np.testing.assert_array_equal(dense_plain, dense_plain_expected)
        np.testing.assert_array_equal(dense_shaped, dense_shaped_expected)
        np.testing.assert_array_equal(dpi, indices_expected)
        np.testing.assert_array_equal(dsi, indices_expected)

    def test_02_06_convert_sparse_to_dense_3d_overlap(self):
        """
        Test converstion of sparse to 3D dense with overlapping labels

        dense output: labels=2, c=1,t=1,z=3,y=3,x=2

               +------+  |     +------+
              / 1  1 /|  |    / 0  0 /|
             / 1  1 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 1  1 |/    |  | 0  0 |/  
            +------+     |  +------+   
               +------+  |     +------+
              / 1  1 /|  |    / 0  0 /|
             / 1  1 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 1  1 |/    |  | 0  0 |/  
            +------+     |  +------+   
               +------+  |     +------+
              / 0  0 /|  |    / 0  0 /|
             / 0  0 / +  |   / 2  2 / +
            +------+ /   |  +------+ / 
            | 0  0 |/    |  | 0  0 |/  
            +------+     |  +------+   
        """
        sparse = np.array(
            [(0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 1),
             (1, 0, 0, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 1),
             (2, 0, 0, 1), (2, 0, 1, 1), (2, 1, 0, 1), (2, 1, 1, 1),
             
             (1, 0, 0, 2), (1, 0, 1, 2),
             (1, 1, 0, 2), (1, 1, 1, 2),
             (1, 2, 0, 2), (1, 2, 1, 2)],
            dtype=[('z', '<u2'), ('y', '<u2'), ('x', '<u2'), ('label', 'u1')]
        )

        dense_plain, dpi = lib_seg.convert_sparse_to_dense(sparse)
        dense_shaped, dsi = lib_seg.convert_sparse_to_dense(
            sparse, dense_shape=(1, 1, 3, 3, 2)
        )

        # + 1 padding
        dense_plain_expected = np.array(
            [1,1,0,1,1,0,0,0,0,0,0,0,
             1,1,0,1,1,0,0,0,0,0,0,0,
             1,1,0,1,1,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,

             0,0,0,0,0,0,0,0,0,0,0,0,
             2,2,0,2,2,0,2,2,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0,
             0,0,0,0,0,0,0,0,0,0,0,0],
             np.uint8
        ).reshape((2,1,1,4,4,3))
        dense_shaped_expected = np.array(
            [1,1,1,1,0,0,
             1,1,1,1,0,0,
             1,1,1,1,0,0,

             0,0,0,0,0,0,
             2,2,2,2,2,2,
             0,0,0,0,0,0],
             np.uint8
        ).reshape((2,1,1,3,3,2))
        indices_expected = np.array([[1], [2]], np.uint8)

        np.testing.assert_array_equal(dense_plain, dense_plain_expected)
        np.testing.assert_array_equal(dense_shaped, dense_shaped_expected)
        np.testing.assert_array_equal(dpi, indices_expected)
        np.testing.assert_array_equal(dsi, indices_expected)
