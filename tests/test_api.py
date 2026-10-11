from server.main import _cloud_for_mode, _frame_payload, summary


def test_summary_endpoint_reads_repository_artifacts() -> None:
    payload = summary()
    assert payload["frame_count"] == 1362
    assert payload["map_points"] == 9285
    assert payload["coverage"] > 20


def test_pointcloud_endpoint_supports_dense_and_sensor_modes() -> None:
    dense_points, dense_colors, _ = _cloud_for_mode("dense")
    sensor_points, sensor_colors, _ = _cloud_for_mode("sensor")

    assert len(dense_points) > 0
    assert len(sensor_points) > 0
    assert len(dense_points) == len(dense_colors)
    assert len(sensor_points) == len(sensor_colors)


def test_replay_endpoint_keeps_lost_pose_missing() -> None:
    payload = _frame_payload(1357)

    assert payload["tracking_status"] == "LOST"
    assert payload["camera_position"] is None
