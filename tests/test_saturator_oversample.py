import pytest
import media.saturator_oversample as mod

CASES = [
    ("oversample_by_rate", (1,)),
    ("band_limit", (1,)),
    ("curve_by_longterm", (1,)),
    ("multichannel_phase", (1,)),
    ("bypass_crossfade", (1,)),
    ("output_limit", (1,)),
    ("process", (1,)),
    ("saturate", (1,)),
    ("bypass", (1,)),
    ("render", (1,)),
    ("validate_distortion", (1,)),
    ("metrics", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
