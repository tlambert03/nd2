import nd2
from nd2 import _nd2decode

f = nd2.ND2File("/Users/talley/Downloads/001-overview_crop.nd2")

m = f._rdr._image_metadata()
e = _nd2decode.Experiment(**m)
