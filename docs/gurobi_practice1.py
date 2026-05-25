import gurobipy as gp

if __name__ == '__main__':
    m = gp.Model()

    # Create variables
    n = 12
    vars = []
    for i in range(n):
        vars.append(m.addVar(lb=0.0, ub=float('inf'), vtype=gp.GRB.CONTINUOUS, name=f'x{i+1}'))

    # Set objective function

    m.setObjective(320*(vars[0]+vars[1]+vars[2])
                    + 400*(vars[3]+vars[4]+vars[6])
                    + 360*(vars[6]+vars[7]+vars[8])
                        + 290*(vars[9]+vars[10]+vars[11]), gp.GRB.MAXIMIZE)

    # Add constraints
    m.addConstr(500*vars[0] + 700*vars[3] + 600*vars[6] + 400*vars[9] == 7000)
    m.addConstr(500*vars[1] + 700*vars[4] + 600*vars[7] + 400*vars[10] == 9000)
    m.addConstr(500*vars[2] + 700*vars[5] + 600*vars[8] + 400*vars[11] == 5000)

    # Solve it!
    m.optimize()

    print(f"Optimal objective value: {m.objVal}")
    print(f"Solution values: ", end="")
    for i in range(len(vars)):
        print(f"x{i+1}= {vars[i].x}", end=" ")
    print()