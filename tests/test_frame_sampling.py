from preprocessing.frame_sampling import uniform_indices

def test_single_frame_repeats(): assert uniform_indices(1,4).tolist()==[0,0,0,0]
def test_bounds():
    x=uniform_indices(7,32); assert len(x)==32 and x.min()>=0 and x.max()<7


def test_adaptive_motion_sampling_is_deterministic_and_fixed_count():
    import numpy as np
    from preprocessing.frame_sampling import adaptive_indices
    scores=np.ones(100,dtype=np.float32)
    scores[60:80]=20.0
    a=adaptive_indices(100,8,scores)
    assert len(a)==8
    assert np.array_equal(a,adaptive_indices(100,8,scores))
    assert a.min()>=0 and a.max()<100
