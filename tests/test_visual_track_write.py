import pytest
import torch

from vggt.utils.visual_track import visualize_tracks_on_images


@pytest.mark.parametrize("blocked", ["frame_0000.png", "tracks_grid.png"])
def test_failed_track_image_write_reports_destination(tmp_path, blocked):
    (tmp_path / blocked).mkdir()
    images = torch.zeros((1, 3, 8, 8))
    tracks = torch.empty((1, 0, 2))
    with pytest.raises(OSError, match=blocked):
        visualize_tracks_on_images(images, tracks, out_dir=tmp_path)
