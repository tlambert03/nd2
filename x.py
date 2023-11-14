
import nd2

with nd2.ND2File("/Users/talley/Downloads/2023-09-11_MLY003_Cpf1.nd2") as f:
    print(f.experiment)
    print(f.attributes)
    # print(f._rdr._raw_experiment)
    print(f.read_frame(1000))
    breakpoint()