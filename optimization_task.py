import pulp

# 1. Business Data
profit_A = 40
profit_B = 30

labour_A = 2
labour_B = 1

machine_A = 1
machine_B = 1

total_labour = 40
total_machine = 30

# 2. Create LP Model
model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

# Decision Variables
A = pulp.LpVariable('Product_A', lowBound=0)
B = pulp.LpVariable('Product_B', lowBound=0)

# Objective Function
model += profit_A * A + profit_B * B

# 3. Constraints
model += labour_A * A + labour_B * B <= total_labour
model += machine_A * A + machine_B * B <= total_machine

# 4. Solve
model.solve()

# 5. Results
print("Status:", pulp.LpStatus[model.status])
print("Units of Product A to produce:", A.varValue)
print("Units of Product B to produce:", B.varValue)
print("Maximum Profit:", pulp.value(model.objective))
