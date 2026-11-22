from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import time
from evaluation_custom_adr import AdrOverrideSchedule, HistoricalMovingAverageADR
from estimation_model import EnergyModel, EnergyParameters

# DR -> (SF, BW_Hz, max_payload_bytes) — EU868, repeater-compatible caps
DATA_RATES = {
    0: (12, 125_000, 51),
    1: (11, 125_000, 51),
    2: (10, 125_000, 51),
    3: (9,  125_000, 115),
    4: (8,  125_000, 222),
    5: (7,  125_000, 222),
}

# Per-fragment header/framing overhead subtracted from a DR's raw payload
# cap to get the usable firmware bytes per chunk. Must match whatever
# cmd_send_file actually subtracts (currently 2 bytes there).
FRAGMENT_OVERHEAD_BYTES = 2


def fragment_capacity(dr: int) -> int:
    """Usable firmware bytes per fragment at a given DR (raw cap minus framing overhead)."""
    return DATA_RATES[dr][2] - FRAGMENT_OVERHEAD_BYTES

def simulate_download(
    params: EnergyParameters,
    adr: "AdrOverrideSchedule"
) -> tuple:

    t = 0
    e = 0
    remaining_bytes = params.remaining_bytes
    current_dr = 5
    t_since_last_uplink = 0
    current_class = "A"

    while remaining_bytes > 0:
        current_dr = adr.update(now=t)
        params.ul_dr = current_dr
        params.dl_dr_a = current_dr

        model = EnergyModel(params)
        best, a, c, details = model.select_class()

        if best == "A":
            t += (model.p.t_tx_overhead + a['uplink_airtime'] + model.p.rx1_delay + a['downlink_airtime'] + model.p.t_rx_overhead + model.p.report_interval)
            e += (model.n_a * model.energy(model.p.p_idle, model.p.t_tx_overhead) + a['tx_energy'] + a['idle_energy'] + a['rx_energy'] + model.n_a * model.energy(model.p.p_idle, model.p.t_rx_overhead) + a['sleep_energy']) / model.n_a
            remaining_bytes -= fragment_capacity(current_dr)
        else:
            if current_class != "C" or t_since_last_uplink > model.p.report_interval:
                t += (model.p.t_tx_overhead + a['uplink_airtime'] + model.p.rx1_delay + a['downlink_airtime'] + model.p.t_rx_overhead)
                t_since_last_uplink = 0
            else:
                time_increment = (model.p.class_c_fragment_interval + c['downlink_airtime'])
                t += time_increment
                e += model.energy(model.p.p_rx, time_increment)
                t_since_last_uplink += time_increment
                remaining_bytes -= fragment_capacity(0)

        current_class = best

        params.remaining_bytes = remaining_bytes

    return t, e


def lambda_sweep(
    param_factory: Callable[[], "EnergyParameters"],
    adr_schedule_factory: Callable[[], "AdrOverrideSchedule"],
    n: int = 21
) -> list[dict]:
    """
    Evaluate simulate_mixed() at `n` evenly spaced lambdas in [0, 1] and
    return the (lambda, latency, energy) points. `find_best_lambda`
    assumes latency is non-increasing and energy is non-decreasing in
    lambda; use this to sanity-check that assumption for a given
    scenario (plot it, or just eyeball whether the two columns move
    monotonically) before trusting its bisection result.
    """
    points = []
    for i in range(n):
        lam = i / (n - 1) if n > 1 else 0.0
        params = param_factory()
        params.lambda_ = lam
        t, e = simulate_download(params, adr_schedule_factory())
        points.append({"lambda": lam, "latency": t, "energy": e})
    return points


def find_best_lambda(
    params_factory: Callable[[], "EnergyParameters"],
    adr_schedule_factory: Callable[[], "AdrOverrideSchedule"],
    deadline: Optional[float] = None,
    energy_budget: Optional[float] = None,
    extra_fragments: int = 2,
    tol: float = 1e-4,
) -> dict:
    """
    Find the lambda that drives simulate_mixed()'s greedy A/C scheduler
    to just barely satisfy the given constraint(s), pushing as much of
    the resulting slack as possible into minimizing whichever metric
    isn't hard-constrained -- because raising lambda favors low-latency
    picks fragment-by-fragment (and lowering it favors low-energy
    picks), this is a genuine trade-off dial, not just a class selector.

    `adr_schedule_factory` must return a *fresh* AdrOverrideSchedule each
    call (it carries mutable state), e.g.:
        find_best_lambda(params, lambda: make_sim_adr(period=3600.0), deadline=1800)

    Assumes (and relies on) latency(lambda) being non-increasing and
    energy(lambda) being non-decreasing over [0, 1] -- check this for
    your scenario with lambda_sweep() if results look off; it need not
    hold exactly for every possible ADR/parameter combination.

      - deadline only:       find the SMALLEST lambda with
                              latency(lambda) <= deadline -- the most
                              energy-favoring lambda that still makes the
                              deadline, so latency ends up as close to
                              (just under) the deadline as possible and
                              energy is minimized as a side effect.
      - energy_budget only:  find the LARGEST lambda with
                              energy(lambda) <= energy_budget -- the most
                              latency-favoring lambda that still fits the
                              budget, so energy ends up as close to (just
                              under) the budget as possible and latency
                              is minimized as a side effect.
      - both given:          intersect the two feasible lambda ranges;
                              if non-empty, return its lower end (the
                              deadline-driven threshold), which also
                              satisfies the energy budget and is the most
                              energy-conservative point that does so. If
                              the ranges don't overlap, no lambda
                              satisfies both simultaneously; the smallest
                              deadline-only violation is reported instead
                              and `infeasible=True` is set.
      - neither given:       raises ValueError.
    """
    if deadline is None and energy_budget is None:
        raise ValueError("Provide at least one of deadline or energy_budget")

    def evaluate(lam: float) -> tuple:
        params = params_factory()
        params.lambda_ = lam
        return simulate_download(params, adr_schedule_factory())

    t0, e0 = evaluate(0.0)
    t1, e1 = evaluate(1.0)
    infeasible = False

    lam_deadline = None  # smallest feasible lambda for the deadline
    if deadline is not None:
        if t0 <= deadline:
            lam_deadline = 0.0
        elif t1 > deadline:
            lam_deadline = None  # infeasible even at lam=1.0
        else:
            lo, hi = 0.0, 1.0  # latency(lo) > deadline, latency(hi) <= deadline
            for _ in range(64):
                if hi - lo < tol:
                    break
                mid = (lo + hi) / 2
                t, _ = evaluate(mid)
                if t <= deadline:
                    hi = mid
                else:
                    lo = mid
            lam_deadline = hi

    lam_energy = None  # largest feasible lambda for the energy budget
    if energy_budget is not None:
        if e1 <= energy_budget:
            lam_energy = 1.0
        elif e0 > energy_budget:
            lam_energy = None  # infeasible even at lam=0.0
        else:
            lo, hi = 0.0, 1.0  # energy(lo) <= budget, energy(hi) > budget
            for _ in range(64):
                if hi - lo < tol:
                    break
                mid = (lo + hi) / 2
                _, e = evaluate(mid)
                if e <= energy_budget:
                    lo = mid
                else:
                    hi = mid
            lam_energy = lo

    if deadline is not None and energy_budget is None:
        if lam_deadline is None:
            lam_result, infeasible = 1.0, True
        else:
            lam_result = lam_deadline
    elif energy_budget is not None and deadline is None:
        if lam_energy is None:
            lam_result, infeasible = 0.0, True
        else:
            lam_result = lam_energy
    else:
        if lam_deadline is None or lam_energy is None or lam_deadline > lam_energy:
            # Empty (or undetermined) intersection: no lambda satisfies both.
            lam_result = lam_deadline if lam_deadline is not None else 0.0
            infeasible = True
        else:
            lam_result = lam_deadline  # lower end of the feasible band

    final_time, final_energy = evaluate(lam_result)

    return {
        "lambda": lam_result,
        "infeasible": infeasible,
        "latency": final_time,
        "energy": final_energy,
        "deadline": deadline,
        "energy_budget": energy_budget,
        "lambda_deadline_threshold": lam_deadline,
        "lambda_energy_threshold": lam_energy,
        "latency_at_lambda0": t0,
        "energy_at_lambda0": e0,
        "latency_at_lambda1": t1,
        "energy_at_lambda1": e1,
    }


def main():

    params_factory = lambda: EnergyParameters(
        remaining_bytes=51200,
        report_interval=60.0,
        class_c_fragment_interval=5,
        ul_dr=5,
        class_c_dr=0,
        duty_cycle=0.01,
        dl_duty_cycle=0.10,
        lambda_=0.7,
    )
    def noop(_): return
    actual_adr = lambda: AdrOverrideSchedule(
        apply=noop,
        start=0.0,
        period=3600.0,
    )
    adr_factory = lambda: HistoricalMovingAverageADR(
        apply=noop,
        observation=actual_adr().target_dr,
        window=5,
        initial_dr=5,
    )

    # Sweep lambda to eyeball whether latency/energy move monotonically
    # (find_best_lambda's bisection assumes this).
    print("\n=== Lambda sweep ===")
    for point in lambda_sweep(params_factory, adr_factory, n=100):
        print(f"  lambda={point['lambda']:.2f}  "
              f"latency={point['latency']:.1f}s  energy={point['energy']:.2f}J")

    # Constraint-driven searches.
    print(params_factory().__str__())

    print("\n=== find_best_lambda: deadline only ===")
    deadline = 8000
    r = find_best_lambda(params_factory, adr_factory, deadline=deadline)
    print(f"lambda={r['lambda']:.4f}  latency={r['latency']:.1f}s  "
          f"energy={r['energy']:.2f}J  deadline={deadline}s infeasible={r['infeasible']}")

    print("\n=== find_best_lambda: energy budget only ===")
    energy_budget = 380
    r = find_best_lambda(params_factory, adr_factory, energy_budget=energy_budget)
    print(f"lambda={r['lambda']:.4f}  latency={r['latency']:.1f}s  "
          f"energy={r['energy']:.2f}J energy_budget={energy_budget}J infeasible={r['infeasible']}")

    print("\n=== find_best_lambda: impossible deadline (infeasible case) ===")
    r = find_best_lambda(params_factory, adr_factory, deadline=deadline, energy_budget=energy_budget)
    print(f"lambda={r['lambda']:.4f}  latency={r['latency']:.1f}s  "
          f"energy={r['energy']:.2f}J deadline={deadline}s energy_budget={energy_budget}J infeasible={r['infeasible']}")

    print("\n=== find_best_lambda: both constraints (feasible case) ===")
    deadline = 13000
    energy_budget = 2000
    r = find_best_lambda(params_factory, adr_factory, deadline=deadline, energy_budget=energy_budget)
    print(f"lambda={r['lambda']:.4f}  latency={r['latency']:.1f}s  "
          f"energy={r['energy']:.2f}J deadline={deadline}s energy_budget={energy_budget}J infeasible={r['infeasible']}")


if __name__ == "__main__":
    main()