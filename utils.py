def Eigenvalue(scaler,low_bond,up_bond):
    #計算upbond、lowbond 間距
    return scaler * (up_bond - low_bond)

class MatrixHandler():

    def _minus(self, vec1, vec2):
        for l1, l2 in zip(vec1, vec2):
            yield l1 - l2

    def _add(self, vec1, vec2):
        for l1, l2 in zip(vec1, vec2):
            yield l1 + l2

    def minus_vec(self, vec1, vec2):
        yield from self._minus(vec1, vec2)

    def add_vec(self, vec1, vec2):
        yield from self._add(vec1, vec2)

    def sub_vec(self, arr1, arr2):
        for vec1, vec2 in zip(arr1, arr2):
            yield list(self.minus_vec(vec1, vec2))

    def sum_vec(self, arr1, arr2):
        for vec1, vec2 in zip(arr1, arr2):
            yield list(self.add_vec(vec1, vec2))

    def threeDimOperation(self, arrs1, arrs2, method):
        for arr1, arr2 in zip(arrs1, arrs2):
            if method == 'add' or method == '+':
                yield list(self.sum_vec(arr1, arr2))
            elif method == 'minus' or method == '-':
                yield list(self.sub_vec(arr1, arr2))

    def twoDimOperation(self, arr1, arr2, method):
        for vec1, vec2 in zip(arr1, arr2):
            if method == 'add' or method == '+':
                yield list(self.add_vec(vec1, vec2))
            elif method == 'minus' or method == '-':
                yield list(self.minus_vec(vec1, vec2))

if __name__ == '__main__':
    weights_arrs = [
                    [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
                    [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
                    [[2.1,2.2,2.3,2.4]]
                    ]
    weights_arrs2 = [
                    [[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]],
                    [[1.1,1.2,1.3],[1.4,1.5,1.6],[1.7,1.8,1.9],[1.3,1.2,1.1]],
                    [[2.1,2.2,2.3,2.4]]
                    ]
    bias_arr = [
                [0.1,0.2,0.3],
                [1.1,1.2,1.3,1.4],
                [2.1]
                ]
    bias_arr2 = [
                [0.1,0.2,0.3],
                [1.1,1.2,1.3,1.4],
                [2.1]
                ]
    # print(list(math_weightsArrs(weights_arrs, weights_arrs2, '-')))
    # print(list(math_biasArr(bias_arr, bias_arr2, '-')))
    print(list(add_vec([1,2], [])))
