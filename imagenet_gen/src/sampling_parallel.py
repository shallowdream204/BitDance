import torch

def time_shift_sana(t: torch.Tensor, flow_shift: float = 1., sigma: float = 1.):
    return (1 / flow_shift) / ( (1 / flow_shift) + (1 / t - 1) ** sigma)

def get_score_from_velocity(velocity, x, t):
    alpha_t, d_alpha_t = t, 1
    sigma_t, d_sigma_t = 1 - t, -1
    mean = x
    reverse_alpha_ratio = alpha_t / d_alpha_t
    var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
    score = (reverse_alpha_ratio * velocity - mean) / var
    return score


def get_velocity_from_cfg(velocity, cfg, cfg_mult):
    if cfg_mult == 2:
        cond_v, uncond_v = torch.chunk(velocity, 2, dim=0)
        velocity = uncond_v + cfg * (cond_v - uncond_v)
    return velocity


@torch.compile()
def euler_step(x, v, dt: float, cfg: float, cfg_mult: int):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        x = x + v * dt
    return x


@torch.compile()
def euler_maruyama_step(x, v, t, dt: float, cfg: float, cfg_mult: int):
    with torch.amp.autocast("cuda", enabled=False):
        v = v.to(torch.float32)
        v = get_velocity_from_cfg(v, cfg, cfg_mult)
        score = get_score_from_velocity(v, x, t)
        drift = v + (1 - t) * score
        noise_scale = (2.0 * (1.0 - t) * dt) ** 0.5
        x = x + drift * dt + noise_scale * torch.randn_like(x)
    return x


def euler_maruyama(
    input_dim,
    forward_fn,
    c: torch.Tensor,
    cfg: float = 1.0,
    num_sampling_steps: int = 20,
    last_step_size: float = 0.05,
    time_shift: float = 1.,
):
    cfg_mult = 1
    if cfg > 1.0:
        cfg_mult += 1

    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device)
    # an = (1.0 - last_step_size) / num_sampling_steps
    t_all = torch.linspace(0, 1-last_step_size, num_sampling_steps+1, device=c.device, dtype=torch.float32)
    t_all = time_shift_sana(t_all, time_shift)
    dt = t_all[1:] - t_all[:-1]
    t = torch.tensor(
        0.0, device=c.device, dtype=torch.float32
    )  # use tensor to avoid compile warning
    t_batch = torch.zeros(c.shape[0], device=c.device)
    for i in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        output = forward_fn(
            combined,
            t_batch,
            c,
        )
        v = (output - combined) / (1 - t_batch.view(-1, 1, 1)).clamp_min(0.05)
        x = euler_maruyama_step(x, v, t, dt[i], cfg, cfg_mult)
        t += dt[i]

    combined = torch.cat([x] * cfg_mult, dim=0)
    t_batch[:] = 1 - last_step_size
    output = forward_fn(
        combined,
        t_batch,
        c,
    )
    v = (output - combined) / (1 - t_batch.view(-1, 1, 1)).clamp_min(0.05)
    x = euler_step(x, v, last_step_size, cfg, cfg_mult)

    return torch.cat([x] * cfg_mult, dim=0)


def euler(
    input_dim,
    forward_fn,
    c,
    cfg: float = 1.0,
    num_sampling_steps: int = 50,
):
    cfg_mult = 1
    if cfg > 1.0:
        cfg_mult = 2

    x_shape = list(c.shape)
    x_shape[0] = x_shape[0] // cfg_mult
    x_shape[-1] = input_dim
    x = torch.randn(x_shape, device=c.device)
    dt = 1.0 / num_sampling_steps
    t = 0
    t_batch = torch.zeros(c.shape[0], device=c.device)
    for _ in range(num_sampling_steps):
        t_batch[:] = t
        combined = torch.cat([x] * cfg_mult, dim=0)
        v = forward_fn(combined, t_batch, c)
        x = euler_step(x, v, dt, cfg, cfg_mult)
        t += dt

    return torch.cat([x] * cfg_mult, dim=0)
