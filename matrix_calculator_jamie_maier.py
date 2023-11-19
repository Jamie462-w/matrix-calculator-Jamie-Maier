#Python Code Linear Algebra take-home assignment Jamie Maier
#In this file is all the code for the whole assignment
#To be able to solve bigger matrices (as big as 30x30 or more), I chose to work with cofactors


#Gets the Matrix and Vector b from the User and makes a list out of it
def matrix_input(input_text):
    print(input_text)
    rows = int(input("Enter the number of rows: "))
    cols = int(input("Enter the number of columns: "))

    matrix = []
    for i in range(rows):
        row = list(map(float, input(f"Enter row {i+1} elements separated by space: ").strip().split()))
        if len(row) != cols:
            raise ValueError("Row does not have the correct number of elements")
        matrix.append(row)

    return matrix

def main():
    if input("Do you want to invert a matrix? (yes/no) ").lower() == 'yes':
        matrix = matrix_input("Enter the matrix to invert:")
        print("Inverse of the matrix:")
        print(inverse(matrix))

    if input("Do you want to solve a linear system Ax = b? (yes/no) ").lower() == 'yes':
        A = matrix_input("Enter the matrix A:")
        b = list(map(float, input("Enter the vector b separated by space: ").strip().split()))
        print("Solving Ax = b")
        solution = solve_linear_system(A, b)
        print("Solution x:", solution)

#This function calculates the determinants for matrices from all sizes
#For matrices larger than 2x2, the function calculates the solution with recursion and the minor function, e.g. a 3x3 matrixes determinant is calculated by three 2x2 submatrices
def determinant(matrix):
    size = len(matrix)
    #In this case the determiant is just the value of the matrix
    if size == 1:
        return matrix[0][0]
    #Formula used = a11*a22 - a12*a21
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    det = 0
    for c in range(size):
        det += ((-1)**c) * matrix[0][c] * determinant(minor(matrix, 0, c))
    return det

#This function calls the determinant function, which calculates the determinant and makes sure, that the result is not 0
def check_inveritbility(matrix):
    return determinant(matrix) != 0

#This function calculates the minor of a matrix with matrix slicing, which means that it returns a submatrix of the original matrix with leaving out the i-row and j-column
def minor(matrix, i, j):
    #matrix[:i] takes all the rows until the row i and matrix[i+1] takes all the rows after the row i. Same for j just for columns
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]

#This function returns the transposed matrix
def transpose(matrix):
    return list(map(list, zip(*matrix)))

#This function creates an empty list and fills it with the calculated cofactor for every cell in the matrix( The cofactor is calculated by taking the determinant of the minor and multiply it by (-1)^(r+c) for every cell with the position (r,c))
def cofactor(matrix):
    cofactors = []
    for r in range(len(matrix)):
        cofactorRow = []
        for c in range(len(matrix)):
            minor_determinant = determinant(minor(matrix, r, c))
            cofactorRow.append(((-1)**(r+c)) * minor_determinant)
        cofactors.append(cofactorRow)
    return cofactors

#This combines all the functions to calculate the inverse matrix with the formula of A^-1 = 1/det(A) * adj(A)
def inverse(matrix):
    if not check_inveritbility(matrix):
        return "Error: The determinant equals 0, which means that the matrix is not invertible and singular."

    det = determinant(matrix)
    cofactors = cofactor(matrix)
    adj = transpose(cofactors)
    return [[element / det for element in row] for row in adj]

#This function calculates the product of two matrices
def matrix_multiplication(A, B):
    return [[sum(a * b for a, b in zip(A_row, B_col)) for B_col in zip(*B)] for A_row in A]

def solve_linear_system(A, b):
    A_inv = inverse(A)
    if isinstance(A_inv, str):  
        return A_inv
    
    #This transforms b into a column vector (it's necessary for the matrix multiplication)
    b_column = [[b_i] for b_i in b]

    #This multiplies A_inv with b_column
    x_column = matrix_multiplication(A_inv, b_column)

    #This flattens the result to get a 1x3 vector and also rounds the numbers because we are not allowed to use numpy
    x = [round(x_i[0]) for x_i in x_column]
    return x


# Provide 10 sample matrices for Task 1 (first matrix is the solution for Task3)
print("First matrix is the solution for Task 3")
sample_matrices = [
    # Task 3
    [
        [1, 2, 3],
        [0, 1, 4],
        [5, 6, 0]
    ],
    [
        [1, 2], 
        [3, 4]
    ],  
    [
        [1, 2], 
        [3, 6]
    ],
    [
        [2, 5], 
        [1, 3]
    ],
    [
        [4, 7, 2], 
        [3, 5, 1], 
        [6, 2, 9]
    ],
    [
        [1, 0, 2, -1], 
        [3, 0, 0, 5], 
        [2, 1, 4, -3], 
        [1, 0, 5, 0]
    ],
    [
        [4, 2], 
        [2, 1]
    ],
    [
        [1, 2, 3], 
        [4, 5, 6], 
        [7, 8, 9]
    ],
    [
        [-1, -2], 
        [-3, -4]
    ],
    [
        [-2, 3, 1], 
        [4, -5, 6], 
        [-7, 8, 9]
    ]
]


# Task 1 
for i, matrix in enumerate(sample_matrices):
    print(f"Matrix {i+1}:")
    inv = inverse(matrix)
    print("Inverse:" if not isinstance(inv, str) else "", inv)






#Task 2 (b):
print("Solution for Task b")
A = [
    [1, -3, -7],
    [-1, 5, 6],
    [-1, 3, 10]
]

b = [10, -21, -7]
print("Solving Ax = b")
print("A:", A)
print("b:", b)
x = solve_linear_system(A, b)
print("Solution x:", x)


if __name__ == "__main__":
    main()