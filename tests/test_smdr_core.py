import ezdxf
import pytest

from smdr_core import (
    build_session_from_doc,
    dxf_points_to_scene,
    extract_template_from_scene_polygon,
    load_template,
    scan_session,
    scene_points_to_dxf,
)


def _make_cluster_doc():
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    pattern = [
        ((0.0, 0.0), 2.0),
        ((5.0, 0.0), 0.75),
        ((0.0, 8.0), 0.75),
    ]
    translations = [
        (10.0, 10.0),
        (30.0, -5.0),
        (-5.0, 35.0),
    ]
    for dx, dy in translations:
        for (px, py), radius in pattern:
            msp.add_circle((px + dx, py + dy), radius=radius)

    msp.add_circle((80.0, 80.0), radius=1.25)
    msp.add_line((0.0, 0.0), (10.0, 0.0))
    return doc


def test_scene_round_trip_preserves_dxf_coordinates():
    dxf_points = [
        (-10.5, 4.25),
        (0.0, 0.0),
        (15.75, -9.5),
        (2.125, 3.875),
    ]

    scene_points = dxf_points_to_scene(dxf_points)
    round_trip = scene_points_to_dxf(scene_points)

    assert round_trip == pytest.approx(dxf_points)


def test_extract_template_from_scene_polygon_selects_expected_dxf_features():
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    msp.add_circle((10.0, 10.0), radius=1.0)
    msp.add_circle((16.0, 10.0), radius=1.0)
    msp.add_circle((13.0, 15.0), radius=2.0)
    msp.add_circle((40.0, 40.0), radius=3.0)

    session, stats = build_session_from_doc(doc)
    assert stats["coord_basis"] == "dxf_scene"

    dxf_polygon = [
        (8.0, 8.0),
        (18.0, 8.0),
        (18.0, 18.0),
        (8.0, 18.0),
    ]
    result = extract_template_from_scene_polygon(session, dxf_points_to_scene(dxf_polygon))
    template = load_template(session, result["template_id"])

    selected = sorted((entity["x"], entity["y"], entity["size"]) for entity in template["entities"])
    assert selected == pytest.approx(
        [
            (10.0, 10.0, 1.0),
            (13.0, 15.0, 2.0),
            (16.0, 10.0, 1.0),
        ]
    )


def test_standard_and_fast_scan_return_same_match_centers():
    session, _ = build_session_from_doc(_make_cluster_doc())

    select_polygon = [
        (7.0, 7.0),
        (17.0, 7.0),
        (17.0, 20.0),
        (7.0, 20.0),
    ]
    extract_result = extract_template_from_scene_polygon(session, dxf_points_to_scene(select_polygon))
    template_id = extract_result["template_id"]

    standard = scan_session(session, template_id=template_id, fast=False)
    fast = scan_session(session, template_id=template_id, fast=True)

    standard_points = sorted((match["dxf_x"], match["dxf_y"]) for match in standard["matches"])
    fast_points = sorted((match["dxf_x"], match["dxf_y"]) for match in fast["matches"])

    assert standard["match_count"] == 3
    assert fast["match_count"] == 3
    assert fast_points == pytest.approx(standard_points)
