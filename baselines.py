"""Baselines from Section 5.1: MILP, Single-ACO and PSO.

All methods use the same A* cost matrices as DualACO (main.py):
    dist_robot_to_task[i, j] = d(s_i, p_j)
    dist_task_to_drop[j]     = d(p_j, q_j)
    dist_drop_to_task[j, k]  = d(q_j, p_k)
Routes use the format of main.py: {robot: [t0, t0 + m, t1, t1 + m, ...]}.
"""
import numpy as np
import pulp as pl


def route_cost(routes, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task):
    # total fleet travel distance, Eqs. (8)-(9)
    total = 0.0
    for i, route in routes.items():
        seq = route[0::2]
        for k, t in enumerate(seq):
            total += dist_robot_to_task[i, t] if k == 0 else dist_drop_to_task[seq[k - 1], t]
            total += dist_task_to_drop[t]
    return float(total)

def build_routes(robot_tasks, m_tasks):
    return {i: [x for t in seq for x in (t, t + m_tasks)] for i, seq in robot_tasks.items()}

def assignment_cost_matrix(dist_robot_to_task, dist_task_to_drop):
    # c_rj = d(s_r, p_j) + d(p_j, q_j), shape (n_robots, m_tasks)
    return dist_robot_to_task + dist_task_to_drop[None, :]

def nearest_neighbor_routes(assignment, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task):
    """Greedy nearest-neighbor sequencing: nearest pickup from the depot, then
    nearest pickup from the current drop-off."""
    n_robots, m_tasks = dist_robot_to_task.shape
    robot_tasks = {}
    for i in range(n_robots):
        remaining = sorted(t for t in range(m_tasks) if assignment[t] == i)
        seq = []
        while remaining:
            if not seq:
                nxt = min(remaining, key=lambda t: dist_robot_to_task[i, t])
            else:
                nxt = min(remaining, key=lambda t: dist_drop_to_task[seq[-1], t])
            seq.append(nxt)
            remaining.remove(nxt)
        robot_tasks[i] = seq
    routes = build_routes(robot_tasks, m_tasks)
    return routes, route_cost(routes, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task)


def run_milp(dist_robot_to_task, dist_task_to_drop, dist_drop_to_task):
    """MILP assignment (CBC) minimizing sum c_rj x_rj, then nearest-neighbor sequencing."""
    n_robots, m_tasks = dist_robot_to_task.shape
    c = assignment_cost_matrix(dist_robot_to_task, dist_task_to_drop)

    prob = pl.LpProblem("assign", pl.LpMinimize)
    x = [[pl.LpVariable(f"x_{i}_{j}", cat=pl.LpBinary) for j in range(m_tasks)] for i in range(n_robots)]
    prob += pl.lpSum(float(c[i, j]) * x[i][j] for i in range(n_robots) for j in range(m_tasks))
    for j in range(m_tasks):
        prob += pl.lpSum(x[i][j] for i in range(n_robots)) == 1
    prob.solve(pl.PULP_CBC_CMD(msg=False))
    if pl.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"MILP not solved: {pl.LpStatus[prob.status]}")

    assignment = np.array([max(range(n_robots), key=lambda i: pl.value(x[i][j])) for j in range(m_tasks)])
    return nearest_neighbor_routes(assignment, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task)


def run_single_aco(run_seed, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task,
                   num_ants=30, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1, q=100.0):
    """Conventional single-pheromone ACO (Eqs. (13)-(14), all ants deposit) on the
    assignment cost c_rj, then nearest-neighbor sequencing."""
    rng = np.random.RandomState(int(run_seed))
    n_robots, m_tasks = dist_robot_to_task.shape
    c = assignment_cost_matrix(dist_robot_to_task, dist_task_to_drop).T  # (m_tasks, n_robots)

    pher = np.ones((m_tasks, n_robots))
    heuristic = 1.0 / (c + 1e-6)

    best_cost = float('inf')
    best_assignment = None
    for iteration in range(num_iterations):
        solutions = []
        for ant_idx in range(num_ants):
            assignment = np.zeros(m_tasks, dtype=int)
            for t in range(m_tasks):
                weights = (pher[t] ** alpha) * (heuristic[t] ** beta)
                assignment[t] = rng.choice(n_robots, p=weights / weights.sum())
            cost = c[np.arange(m_tasks), assignment].sum()
            solutions.append((assignment, cost))
            if cost < best_cost:
                best_cost = cost
                best_assignment = assignment.copy()

        pher *= (1.0 - rho)
        for assignment, cost in solutions:
            pher[np.arange(m_tasks), assignment] += q / (cost + 1e-9)

    return nearest_neighbor_routes(best_assignment, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task)


def run_pso(run_seed, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task,
            num_particles=30, num_iterations=100, w=0.7, c1=1.5, c2=1.5):
    """Standard PSO with random-key encoding, minimizing Eq. (8) directly.

    A particle has 2m keys: keys[:m] order the tasks (ascending), keys[m:] in
    [0, n_robots) pick each task's robot (floor). Each robot visits its tasks
    in key order.
    """
    rng = np.random.RandomState(int(run_seed))
    n_robots, m_tasks = dist_robot_to_task.shape
    lower = np.zeros(2 * m_tasks)
    upper = np.concatenate([np.ones(m_tasks), np.full(m_tasks, float(n_robots))])
    v_max = 0.2 * (upper - lower)

    def decode(keys):
        robot = np.minimum(np.floor(keys[m_tasks:]).astype(int), n_robots - 1)
        order = np.argsort(keys[:m_tasks], kind="stable")
        robot_tasks = {i: [int(t) for t in order if robot[t] == i] for i in range(n_robots)}
        return build_routes(robot_tasks, m_tasks)

    def evaluate(keys):
        return route_cost(decode(keys), dist_robot_to_task, dist_task_to_drop, dist_drop_to_task)

    pos = rng.uniform(lower, upper, size=(num_particles, 2 * m_tasks))
    vel = rng.uniform(-v_max, v_max, size=(num_particles, 2 * m_tasks))
    pbest = pos.copy()
    pbest_cost = np.array([evaluate(p) for p in pos])
    g = int(np.argmin(pbest_cost))
    gbest, gbest_cost = pbest[g].copy(), pbest_cost[g]

    for iteration in range(num_iterations):
        r1 = rng.rand(num_particles, 2 * m_tasks)
        r2 = rng.rand(num_particles, 2 * m_tasks)
        vel = w * vel + c1 * r1 * (pbest - pos) + c2 * r2 * (gbest - pos)
        vel = np.clip(vel, -v_max, v_max)
        pos = np.clip(pos + vel, lower, upper - 1e-9)
        for p in range(num_particles):
            cost = evaluate(pos[p])
            if cost < pbest_cost[p]:
                pbest[p], pbest_cost[p] = pos[p].copy(), cost
                if cost < gbest_cost:
                    gbest, gbest_cost = pos[p].copy(), cost

    routes = decode(gbest)
    return routes, route_cost(routes, dist_robot_to_task, dist_task_to_drop, dist_drop_to_task)
