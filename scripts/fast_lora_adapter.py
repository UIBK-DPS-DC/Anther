import numpy as np
from scipy.special import erf
from anther_model import DR_PARAMS, DEFAULT_POWER, lora_toa

DEFAULT_SENSITIVITY_DBM = {
    7: -123.0,
    8: -126.0,
    9: -129.0,
    10: -132.0,
    11: -134.5,
    12: -137.0,
}


def compute_rss_matrix(ed_positions, gw_positions, tx_power_dbm, pl_d0=105.0, d0=100.0, gamma=2.7):
    ed_pos = np.asarray(ed_positions, dtype=float)
    gw_pos = np.asarray(gw_positions, dtype=float)
    diff = ed_pos[:, None, :] - gw_pos[None, :, :]
    dist = np.maximum(np.linalg.norm(diff, axis=-1), 1.0)
    p_tx = np.asarray(tx_power_dbm, dtype=float).reshape(-1, 1)
    return p_tx - pl_d0 - 10.0 * gamma * np.log10(dist / d0)


def shadow_fading_prob(rss_matrix, sf_vector, sigma_db=6.0, sensitivity_map=None):
    if sensitivity_map is None:
        sensitivity_map = DEFAULT_SENSITIVITY_DBM
    sens = np.array([sensitivity_map[int(sf)] for sf in sf_vector], dtype=float).reshape(-1, 1)
    if sigma_db <= 1e-9:
        return (rss_matrix >= sens).astype(float)
    z = (sens - rss_matrix) / (sigma_db * np.sqrt(2.0))
    return 0.5 * (1.0 - erf(z))


def evaluate_fast_lora_network(
    n_eds=50,
    n_gws=1,
    area_radius_m=1500.0,
    sf_init=8,
    tx_power_dbm=14.0,
    payload_bytes=51,
    packet_interval_s=60.0,
    sigma_db=6.0,
    seed=42,
):
    rng = np.random.default_rng(seed)
    radii = area_radius_m * np.sqrt(rng.uniform(0.01, 1.0, size=n_eds))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_eds)
    ed_pos = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

    if n_gws == 1:
        gw_pos = np.array([[0.0, 0.0]])
    else:
        gw_angles = np.linspace(0.0, 2.0 * np.pi, n_gws, endpoint=False)
        gw_r = 0.5 * area_radius_m
        gw_pos = np.column_stack([gw_r * np.cos(gw_angles), gw_r * np.sin(gw_angles)])

    sf_vec = np.full(n_eds, int(sf_init), dtype=int)
    rss_mat = compute_rss_matrix(ed_pos, gw_pos, tx_power_dbm)
    psi_mat = shadow_fading_prob(rss_mat, sf_vec, sigma_db=sigma_db)

    toa_vec = np.array([lora_toa(payload_bytes, int(s), 125e3) for s in sf_vec])
    duty_cycle = np.clip(toa_vec / max(packet_interval_s, toa_vec.max() + 0.1), 0.0, 0.95)
    overlap_mat = np.outer(duty_cycle, duty_cycle)
    np.fill_diagonal(overlap_mat, 0.0)
    collision_loss = 1.0 - np.clip(np.sum(overlap_mat, axis=1), 0.0, 0.85)

    rx_prob_gw = psi_mat * collision_loss[:, None]
    pdr_vec = 1.0 - np.prod(1.0 - rx_prob_gw, axis=1)

    energy_per_pkt = DEFAULT_POWER["P_tx"] * toa_vec
    bits_delivered = payload_bytes * 8.0 * pdr_vec
    ee_bits_per_j = bits_delivered / np.maximum(energy_per_pkt, 1e-9)

    return {
        "mean_pdr": float(np.mean(pdr_vec)),
        "mean_ee_bits_per_j": float(np.mean(ee_bits_per_j)),
        "mean_rss_dbm": float(np.mean(rss_mat)),
        "pdr_per_ed": pdr_vec,
        "rss_matrix_dbm": rss_mat,
    }


def effective_link_multiplier(dr, n_eds=50, n_gws=1, payload_bytes=51, seed=42):
    sf = DR_PARAMS[dr]["SF"]
    metrics = evaluate_fast_lora_network(
        n_eds=n_eds,
        n_gws=n_gws,
        sf_init=sf,
        payload_bytes=payload_bytes,
        seed=seed,
    )
    pdr = max(metrics["mean_pdr"], 0.05)
    return pdr, 1.0 / pdr
