from preprocessing.frame_sampling import uniform_indices

def test_single_frame_repeats(): assert uniform_indices(1,4).tolist()==[0,0,0,0]
def test_bounds():
    x=uniform_indices(7,32); assert len(x)==32 and x.min()>=0 and x.max()<7
