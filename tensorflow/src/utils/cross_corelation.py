import numpy as np
def corelation2d(in1 = np.array((3,3)), in2 = np.array((3,3)), mode = "valid"):
    current_element = (0,0)
    y = 0
    stride = 1
    output = np.array(in2.shape)
    for iter in range(len(in2)**2):#n * m output matrix
        submatrix = in1[0:in2.shape[0], 0:in2.shape[0]]
        print(submatrix)
        y += np.sum(np.multiply(submatrix, in2))
        print(f"y = {y}")

def main():
    in1 = [[1,6,2],
           [5,3,1],
           [7,0,4]]
    in2 = [[1,2],
           [-1,0]]
    corelation2d(in1, in2)
if __name__ == "main":
    main()




