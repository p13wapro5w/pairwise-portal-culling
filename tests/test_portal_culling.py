import pytest
import render.portal_culling as mod

CASES = [
    ("portal_frustum", (1, 2)),
    ("recursion_depth", (2, 3)),
    ("near_plane", (0.1, 0.2)),
    ("dynamic_rebuild", (1, 2)),
    ("mirror_flip", (1, 2)),
    ("runtime_update", (1, 2)),
    ("overlap_dedup", (1, 2)),
    ("crossing_transition", (1, 2)),
    ("frame_occlusion", (1, 2)),
    ("per_camera", (1, 2)),
    ("portal_poly", (4, 3)),
    ("visibility_set", (1, 2)),
    ("clip_plane", ([0.1, 0.2],)),
    ("portal_graph", (1, 2)),
    ("cull_frustum", (1, 2)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
