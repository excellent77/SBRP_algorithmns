from typing import Dict, List, Set, Tuple

import gurobipy as gp
from gurobipy import GRB

from utils.data_models import Route, School, Station, Solution
from utils.instance_processer import load_instance_from_csv
from utils.solution_utils import build_route, build_solution, route_max_load, print_solution_pretty

MAX_ROUTE_MIN = 60.0
BUS_CAPACITY = 40
MAX_TOTAL_BUSES = 6


def build_groups(schools: List[School], stations: List[Station]) -> Tuple[List[Dict], Dict[Tuple[int, int], int]]:
    groups = []
    group_map = {}
    gid = 0
    for st in stations:
        for sch_idx, count in st.demands.items():
            groups.append({
                'id': gid,
                'st_idx': st.idx,
                'sch_idx': sch_idx,
                'count': count
            })
            group_map[(st.idx, sch_idx)] = gid
            gid += 1
    return groups, group_map


def build_route_events(path: List[Tuple[str, int]], groups: List[Dict], stations_by_idx: Dict[int, Station], schools_by_idx: Dict[int, School]) -> List:
    events = []
    for event_type, idx in path:
        if event_type == 'pickup':
            g = groups[idx]
            st = stations_by_idx[g['st_idx']]
            events.append(Station(st.idx, st.name, {g['sch_idx']: g['count']}, st.orig_idx))
        else:
            events.append(schools_by_idx[idx])
    return events


def build_route_groups(route: Route, group_map: Dict[Tuple[int, int], int]) -> Set[int]:
    return {
        group_map[(st_idx, sch_idx)]
        for st_idx, sch_idx, _, _ in route.pickup_detail
    }


def generate_all_feasible_routes(
        schools: List[School],
        stations: List[Station],
        time_matrix: List[List[float]]
) -> Tuple[List[Route], List[Dict], Dict[Tuple[int, int], int]]:
    groups, group_map = build_groups(schools, stations)
    stations_by_idx = {s.idx: s for s in stations}
    schools_by_idx = {s.idx: s for s in schools}
    routes: List[Route] = []
    seen_paths: Set[Tuple[Tuple[str, int], ...]] = set()

    def dfs(curr_orig: int,
            time: float,
            load: int,
            onboard: List[int],
            visited: Set[int],
            path: List[Tuple[str, int]]):

        if time > MAX_ROUTE_MIN or load > BUS_CAPACITY:
            return

        if not onboard and path:
            route_key = tuple(path)
            if route_key not in seen_paths:
                seen_paths.add(route_key)
                events = build_route_events(path, groups, stations_by_idx, schools_by_idx)
                route = build_route(events, schools, stations_by_idx, time_matrix, auto_fill=False)
                if route.minutes <= MAX_ROUTE_MIN + 1e-6 and route_max_load(route) <= BUS_CAPACITY:
                    routes.append(route)

        if len(path) > 2 * len(groups) + 2:
            return

        # Drop any onboard school
        onboard_schools = sorted({groups[gid]['sch_idx'] for gid in onboard})
        for sch_idx in onboard_schools:
            drop_coord = schools_by_idx[sch_idx].orig_idx
            travel = 0.0 if curr_orig is None else time_matrix[curr_orig][drop_coord]
            new_time = time + travel
            if new_time > MAX_ROUTE_MIN:
                continue
            dropped = [gid for gid in onboard if groups[gid]['sch_idx'] == sch_idx]
            new_load = load - sum(groups[gid]['count'] for gid in dropped)
            path.append(('drop', sch_idx))
            dfs(drop_coord, new_time, new_load, [gid for gid in onboard if gid not in dropped], visited, path)
            path.pop()

        # Pickup a new group
        for g in groups:
            if g['id'] in visited:
                continue
            st_coord = stations_by_idx[g['st_idx']].orig_idx
            travel = 0.0 if curr_orig is None else time_matrix[curr_orig][st_coord]
            new_time = time + travel
            if new_time > MAX_ROUTE_MIN:
                continue
            if load + g['count'] > BUS_CAPACITY:
                continue
            path.append(('pickup', g['id']))
            visited.add(g['id'])
            dfs(st_coord, new_time, load + g['count'], onboard + [g['id']], visited, path)
            visited.remove(g['id'])
            path.pop()

    dfs(None, 0.0, 0, [], set(), [])
    return routes, groups, group_map


def solve_set_partitioning(routes: List[Route], groups: List[Dict], group_map: Dict[Tuple[int, int], int], stations: List[Station]) -> Solution:
    model = gp.Model('set_partitioning')
    model.setParam('OutputFlag', 0)

    x = model.addVars(len(routes), vtype=GRB.BINARY, name='x')
    model.setObjective(gp.quicksum(x[i] * routes[i].route_cost() for i in range(len(routes))), GRB.MINIMIZE)

    route_group_sets = [build_route_groups(r, group_map) for r in routes]

    for g in groups:
        g_id = g['id']
        cover_idxs = [i for i, rset in enumerate(route_group_sets) if g_id in rset]
        if not cover_idxs:
            raise RuntimeError(f'No route covers group {g_id} (station {g["st_idx"]}, school {g["sch_idx"]})')
        model.addConstr(gp.quicksum(x[i] for i in cover_idxs) == 1, name=f'cover_{g_id}')

    model.addConstr(gp.quicksum(x[i] for i in range(len(routes))) <= MAX_TOTAL_BUSES, name='max_buses')
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f'Gurobi did not solve optimally, status={model.status}')

    selected_routes = [routes[i] for i in range(len(routes)) if x[i].X > 0.5]
    return build_solution(selected_routes, stations)


def solve_exact_with_gurobi(schools: List[School], stations: List[Station], time_matrix: List[List[float]]) -> Solution:
    routes, groups, group_map = generate_all_feasible_routes(schools, stations, time_matrix)
    print(f'Generated {len(routes)} feasible routes covering {len(groups)} groups.')
    sol = solve_set_partitioning(routes, groups, group_map, stations)
    return sol


if __name__ == '__main__':
    schools, stations, time_matrix = load_instance_from_csv(stops_csv='./data/stops-b_7.csv', time_csv='./data/time-b_7.csv')
    solution = solve_exact_with_gurobi(schools, stations, time_matrix)
    print_solution_pretty(solution, stations, schools)
