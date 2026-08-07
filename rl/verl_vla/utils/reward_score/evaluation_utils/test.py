from math_util import evaluate_math
from code_util import evaluate_code
import json
# Parse the gold and answer
# If you know that gold will only contain latex or expr (no latex env), use

# Order here is important!


"""
print(evaluate_math("(1,4.5)","(1,\\frac{9}{2})"))
print(evaluate_math("\\boxed{\\frac{x}{7}+\\frac{2}{7}}","\\frac{x+2}{7}"))
print(evaluate_math("\\boxed{2022!}","\\frac{2023!}{2023}"))
print(evaluate_math("\\begin{bmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19 \n \\end{bmatrix}",
                    "\\begin{pmatrix}\n -7 & 6 & -8 \\\\\n 11 & -9 & 12 \\\\\n 15 & -16 & 19\n \\end{pmatrix}"))
print(evaluate_math("\\frac{\\sqrt{505}}{7}", "\\frac{\\sqrt{505}}{7}"))
print(evaluate_math("x^2 + y^2 + 4x - 6y + 13", "(x + 2)^2 + (y - 3)^2"))
print(evaluate_math("\\boxed{1 -frac{1950!}{(1950-m)! \\cdot 1950^m}}","1 - \\frac{\\binom{1950}{m}m!}{1950^m}"))
"""



test_cases = """{"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]], [[{"age": 28, "bmi": 24.0, "child
ren": 2}, {"age": 32, "bmi": 26.0, "children": 1}, {"age": 36, "bmi": 28.0, "children": 3}]], [[{"age": 29, "bmi": 23.5, "children": 1}, {"age": 31, "bmi": 27.5, "children": 2}, {"age": 33, "bmi": 31.5, "children": 3}]], [[{"age": 45
, "bmi": 30.0, "children": 3}, {"age": 55, "bmi": 35.0, "children": 4}, {"age": 65, "bmi": 40.0, "children": 5}]], [[{"age": 27, "bmi": 22.5, "children": 0}, {"age": 29, "bmi": 26.5, "children": 1}, {"age": 31, "bmi": 30.5, "children
": 2}]], [[{"age": 39, "bmi": 22.0, "children": 2}, {"age": 41, "bmi": 24.0, "children": 3}, {"age": 43, "bmi": 26.0, "children": 4}]]], "outputs": [[[{"age": -1.0, "bmi": 1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": 1
.0}, {"age": 1.0, "bmi": -1.0, "children": -1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": -1.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0
}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"ag
e": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0, "bmi": 1.0, "children": 1.0}]], [[{"age": -1.0, "bmi": -1.0, "children": -1.0}, {"age": 0.0, "bmi": 0.0, "children": 0.0}, {"age": 1.0,
 "bmi": 1.0, "children": 1.0}]]]}"""        

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]],], "outputs": [[[{"age": -1.0, "bmi": 1.0, "children": 0.0}, {"age": 0.0, "bmi": 0.0, "children": 1.0}, {"age": 1.0, "bmi": -1.0, "children": -1.0}]],]} , ensure_ascii=False)                                                                                                                                                                                           



completion = """      
```python                                                                                                                                                                                                            
from typing import List, Dict                                                                                                                                                                                      
from statistics import mean, stdev                                                                                                                                                                                 
                                                                                                                                                                                                                    
def standardize_medical_data(data: List[Dict[str, float]]) -> List[Dict[str, float]]:                                                                                                                              
    standardized_data = []                                                                                                                                                                                         
                                                                                                                                                                                                                
    for patient in data:                                                                                                                                                                                           
        standardized_patient = {}                                                                                                                                                                                  
        patient_mean = mean(patient[key] for key in patient if key.isdigit())                                                                                                                                      
        patient_std = stdev(patient[key] for key in patient if key.isdigit())                                                                                                                                      
                                                                                                                                                                                                                
        for key, value in patient.items():                                                                                                                                                                         
            if key.isdigit():                                                                                                                                                                                      
                standardized_patient[key] = (value - patient_mean) / patient_std                                                                                                                                   
            else:                                                                                                                                                                                                  
                standardized_patient[key] = value                                                                                                                                                                  
                                                                                                                                                                                                                
        standardized_data.append(standardized_patient)                                                                                                                                                             
                                                                                                                                                                                                                
    return standardized_data      
```
"""

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]]], "outputs": [[[{"age": 40, "bmi": 30.0, "children": 2}, {"age": 50, "bmi": 25.0, "children": 3}, {"age": 60, "bmi": 20.0, "children": 1}]]]} , ensure_ascii=False)                                                                                                                                                                                           

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [["abc"]], "outputs": [["abc"]]} , ensure_ascii=False)     
                                                                                                                                                                                      
test_cases = "{\"fn_name\": \"standardize_medical_data\", \"inputs\": [[\"abc\",[3,4],[{3:5},4]]], \"outputs\": [[\"abc\",[3,4],[{3:5},4]]]}"    

test_cases = json.dumps({"fn_name": "standardize_medical_data", "inputs": [["abc",[3,4],[{True:5},4]]], "outputs": [["abc",[3,4],[{True:5},4]]]})  
print("mttestcase",test_cases)
print(json.loads(test_cases))

completion = """      
```python           
import sys            
from statistics import mean, stdev           
from typing import List                                                                                                                                                                                                                                                                                                                                                                                           
def standardize_medical_data(a,y,z):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    return a,y  ,z    
```
"""


# completion = """import sys

# def main():
    
#     # Read start and target positions
    
#     # The total number of allowed cells is at most 10^5.
#     for _ in range(n):
#         for c in range(a, b + 1):
    
#     # Directions for the king (8 neighboring cells)
#     directions = [(-1, -1), (-1, 0), (-1, 1),
#                   (1, -1),  (1, 0),  (1, 1)]
    
    
#     # BFS initialization
#     # Mark the starting cell as visited by removing it from allowed set.
    
#     while queue:
#         if (x, y) == target:
#         for dx, dy in directions:
#             if (nx, ny) in allowed:
    

# if __name__ == '__main__':
# """
 


success, metadata_list, format_correctness = evaluate_code(completion, test_cases)
print(success)
print(metadata_list)
print(format_correctness)
