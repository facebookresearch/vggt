# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

from training.data.dataset_util import adjust_track_rot90, rotate_90_degrees


class TrackRotationTest(unittest.TestCase):
    def check_tracks_follow_pixels(self, clockwise):
        for height, width in ((3, 5), (5, 3), (4, 4)):
            with self.subTest(height=height, width=width):
                labels = np.arange(height * width).reshape(height, width)
                image = np.repeat(labels[..., None], 3, axis=-1)
                y, x = np.indices((height, width))
                tracks = np.stack((x.ravel(), y.ravel()), axis=-1)
                original_tracks = tracks.copy()
                rotated_image, rotated_depth, _, _, rotated_tracks = rotate_90_degrees(
                    image,
                    labels,
                    np.eye(4)[:3],
                    np.eye(3),
                    clockwise=clockwise,
                    track=tracks,
                )
                k = -1 if clockwise else 1
                np.testing.assert_array_equal(rotated_image, np.rot90(image, k))
                np.testing.assert_array_equal(rotated_depth, np.rot90(labels, k))
                np.testing.assert_array_equal(tracks, original_tracks)
                np.testing.assert_array_equal(
                    rotated_image[rotated_tracks[:, 1], rotated_tracks[:, 0], 0],
                    labels.ravel(),
                )

    def test_clockwise_tracks_follow_rotated_pixels(self):
        self.check_tracks_follow_pixels(clockwise=True)

    def test_counterclockwise_tracks_follow_rotated_pixels(self):
        self.check_tracks_follow_pixels(clockwise=False)

    def test_clockwise_subpixel_tracks(self):
        tracks = np.array([[0.25, 1.5], [3.5, 0.25]])
        np.testing.assert_allclose(
            adjust_track_rot90(tracks, image_width=5, image_height=3, clockwise=True),
            [[0.5, 0.25], [1.75, 3.5]],
        )

    def test_counterclockwise_subpixel_tracks(self):
        tracks = np.array([[0.25, 1.5], [3.5, 0.25]])
        np.testing.assert_allclose(
            adjust_track_rot90(tracks, image_width=5, image_height=3, clockwise=False),
            [[1.5, 3.75], [0.25, 0.5]],
        )


if __name__ == "__main__":
    unittest.main()
