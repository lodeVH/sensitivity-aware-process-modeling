clc;

output_root = fullfile(pwd, 'generated_heat_data', 'zero_N1');
if ~exist(output_root, 'dir')
    mkdir(output_root);
end

for noise_mode = 1:3
    [noise_tag, noise_desc] = get_noise_model_info(noise_mode);
    noise_output_dir = fullfile(output_root, noise_tag);
    if ~exist(noise_output_dir, 'dir')
        mkdir(noise_output_dir);
    end

    fprintf('\n=== Noise mode %d (%s) -> %s ===\n', noise_mode, noise_desc, noise_output_dir);

    for H_mode = 1:5
        run_single_case(H_mode, noise_mode, noise_output_dir);
    end
end


function run_single_case(H_mode, noise_mode, output_dir)
rng(42);

%% ==========================
% Simulation settings
%% ==========================
T_end = 100000;      % total simulation time [h]
Ts    = 10;          % sampling time [h]
tspan = [0 T_end];
t_uniform = tspan(1):Ts:tspan(end);

%% ==========================
% Parameters
%% ==========================
p.Tamb = 22.0;

% Temperature node losses
p.a1 = 0.1; p.a2 = 0.1; p.a3 = 0.1; p.a4 = 0.1;

% Input gains for T-nodes
p.b1 = 1; p.b2 = 1; p.b3 = 1; p.b4 = 1;

% Coupling T_i <-> C
cc = 0;
CC = 10;
p.g1C = cc; p.g2C = cc; p.g3C = cc; p.g4C = cc;
p.gC1 = CC; p.gC2 = CC; p.gC3 = CC; p.gC4 = CC;

% Disturbance coupling from H
p.d1 = 0.3; p.d2 = 0.3; p.dC = 0; p.dP = 0;

% Constant drifts / offsets
p.e1 = 0.00; p.e2 = 2.00; p.e3 = 4.00; p.e4 = 6.00; p.eC = 0.00; p.eP = 3.00;

% Power dynamics
p.aP = 0.1;
p.beta1 = 1; p.beta2 = 1; p.beta3 = 1; p.beta4 = 1;

% Sinusoidal H parameters
p.eta0 = 0;
p.eta_amp = 0.5;
p.eta_freq = 1/400;
p.eta_phase = 0.0;

% Decreasing H parameters
p.decay_start = 2.0;
p.decay_end = -1.0;

% Brownian H parameters
p.brown_mean = 0.35;
p.brown_sigma = 0.12;
p.brown_min = 0;
p.brown_max = 0.5;
p.brown_mean_pull = 0.05;

% Midpoint-step H parameters
p.step_time = 0.5 * T_end;
p.step_initial = 0.0;
p.step_level = 1.5;

% PRBS-like H parameters
p.prbs_mean = 0.7;
p.prbs_amp_min = 0;
p.prbs_amp_max = 1;
p.prbs_hold_min = 1000; % hours
p.prbs_hold_max = 5000;

% C dynamics
p.aC = 0.1;

%% ==========================
% Heat-system measurement-noise parameters
%% ==========================
noise_sigma_temp = 0.15;
noise_sigma_power = 0.50;
noise_quant_temp = 0.10;
noise_quant_power = 0.50;
noise_rw_sigma_temp = 0.004;
noise_rw_sigma_power = 0.010;

%% ==========================
% Time constants (tau) around a nominal operating point
%% ==========================
x_op = [22; 22; 22; 22; 22; 0];
u_op = [0; 0; 0; 0];

A = numerical_jacobian_x(@(x) rhs_thermal7_const_u(x, p, u_op), x_op);
lambda = eig(A);

mask = real(lambda) < -1e-9;
tau_h = -1 ./ real(lambda(mask));
tau_s = 3600 * tau_h;

if isempty(tau_h)
    warning('No stable real-part modes found for tau computation at the operating point.');
    tau_h = NaN; tau_s = NaN;
else
    [tau_h, idx_tau] = sort(tau_h, 'ascend');
    tau_s = tau_s(idx_tau);
end

fprintf('\n=== Nominal linearized time constants (before simulation) ===\n');
if all(isnan(tau_h))
    fprintf('Tau values unavailable (no stable modes selected).\n');
else
    for i = 1:numel(tau_h)
        fprintf('Mode %d: tau = %.4f h (%.2f s), 5*tau = %.4f h (%.2f s)\n', ...
            i, tau_h(i), tau_s(i), 5*tau_h(i), 5*tau_s(i));
    end
    fprintf('Fastest tau: %.4f h (%.2f s)\n', min(tau_h), min(tau_s));
    fprintf('Slowest tau: %.4f h (%.2f s)\n\n', max(tau_h), max(tau_s));
end

%% ==========================
% Input constraints and return-to-baseline stepped schedule
%% ==========================
u_nom = [0; 0; 0; 0];
u_min = [0; 0; 0; 0];
u_max = [5; 5; 5; 5];

order = {'u1','u2','u3','u4'};

durations.u1 = 70;   % [h]
durations.u2 = 70;
durations.u3 = 70;
durations.u4 = 70;

% Absolute step-size bounds for u changes from zero baseline.
abs_change_min.u1 = 0.10; abs_change_max.u1 = 0.30;
abs_change_min.u2 = 0.10; abs_change_max.u2 = 0.30;
abs_change_min.u3 = 0.10; abs_change_max.u3 = 0.30;
abs_change_min.u4 = 0.10; abs_change_max.u4 = 0.30;

rest_min = 70;       % [h] time between successive steps
rest_max = 70;

cfgs = make_zero_return_pulses_rest(T_end, order, durations, ...
    abs_change_min, abs_change_max, ...
    u_min, u_max, ...
    rest_min, rest_max);

u1_fun = @(t) eval_step_zero_return_input(t, cfgs.u1, u_nom(1), u_min(1), u_max(1));
u2_fun = @(t) eval_step_zero_return_input(t, cfgs.u2, u_nom(2), u_min(2), u_max(2));
u3_fun = @(t) eval_step_zero_return_input(t, cfgs.u3, u_nom(3), u_min(3), u_max(3));
u4_fun = @(t) eval_step_zero_return_input(t, cfgs.u4, u_nom(4), u_min(4), u_max(4));

%% ==========================
% H profile
%% ==========================
[H_t, H_vals, H_mode_tag, H_mode_desc] = make_H_profile(t_uniform, T_end, H_mode, p);
H_fun = @(t) interp1(H_t, H_vals, t, 'previous', 'extrap');

fprintf('Selected H mode %d: %s\n', H_mode, H_mode_desc);

%% ==========================
% Initial state
%% ==========================
x0 = [22; 22; 22; 22; 22; 0];

%% ==========================
% Simulate
%% ==========================
[t, x] = ode15s(@(t,x) rhs_thermal7(t, x, p, u1_fun, u2_fun, u3_fun, u4_fun, H_fun), ...
    t_uniform, x0);

u1 = arrayfun(u1_fun, t_uniform);
u2 = arrayfun(u2_fun, t_uniform);
u3 = arrayfun(u3_fun, t_uniform);
u4 = arrayfun(u4_fun, t_uniform);
H = arrayfun(H_fun, t_uniform);

T1_true = x(:,1); T2_true = x(:,2); T3_true = x(:,3); T4_true = x(:,4);
C_true  = x(:,5); P_true  = x(:,6);

[noise_tag, noise_desc] = get_noise_model_info(noise_mode);
[Y_meas, noise_meta] = apply_heat_measurement_noise( ...
    [T1_true(:), T2_true(:), T3_true(:), T4_true(:), C_true(:), P_true(:)], ...
    Ts, noise_mode, ...
    noise_sigma_temp, noise_sigma_power, ...
    noise_quant_temp, noise_quant_power, ...
    noise_rw_sigma_temp, noise_rw_sigma_power);

T1 = Y_meas(:,1); T2 = Y_meas(:,2); T3 = Y_meas(:,3);
T4 = Y_meas(:,4); C  = Y_meas(:,5); P  = Y_meas(:,6);

fprintf('Selected noise mode %d: %s\n', noise_mode, noise_desc);

fprintf('=== Time-constant reference (after simulation) ===\n');
if ~all(isnan(tau_h))
    fprintf('Tau range from nominal linearization: [%.4f, %.4f] h  =  [%.2f, %.2f] s\n\n', ...
        min(tau_h), max(tau_h), min(tau_s), max(tau_s));
else
    fprintf('Tau range unavailable (see warning above).\n\n');
end

%% ==========================
% Export CSV
%% ==========================
T = table(t(:), T1(:), T2(:), T3(:), T4(:), C(:), P(:), H(:), ...
    u1(:), u2(:), u3(:), u4(:), ...
    'VariableNames', {'t','T1','T2','T3','T4','C','P','H','u1','u2','u3','u4'});

C_tag = make_C_tag(cc, CC);
duration_tag = make_duration_tag(durations);
filename = fullfile(output_dir, sprintf('heat_zero_%s_%s_%s_%s_Tend%g_Ts%g.csv', ...
    C_tag, duration_tag, H_mode_tag, noise_tag, T_end, Ts));
writetable(T, filename);

fprintf('CSV file written: %s\n', filename);

meta_filename = fullfile(output_dir, sprintf('heat_zero_%s_%s_%s_%s_Tend%g_Ts%g_meta.csv', ...
    C_tag, duration_tag, H_mode_tag, noise_tag, T_end, Ts));
T_meta = make_run_metadata_table(H_mode, H_mode_tag, noise_mode, noise_tag, ...
    noise_sigma_temp, noise_sigma_power, ...
    noise_quant_temp, noise_quant_power, ...
    noise_rw_sigma_temp, noise_rw_sigma_power, Ts);
writetable(T_meta, meta_filename);

fprintf('Metadata CSV file written: %s\n', meta_filename);
disp(noise_meta);
end


function dx = rhs_thermal7(t, x, p, u1_fun, u2_fun, u3_fun, u4_fun, H_fun)
T1 = x(1); T2 = x(2); T3 = x(3); T4 = x(4);
C  = x(5); P  = x(6);

u1 = u1_fun(t);
u2 = u2_fun(t);
u3 = u3_fun(t);
u4 = u4_fun(t);
H = H_fun(t);

dT1 = -p.a1*(T1 - p.Tamb) + p.b1*u1 + p.g1C*(C - T1) + p.d1*H + p.e1;
dT2 = -p.a2*(T2 - p.Tamb) + p.b2*u2 + p.g2C*(C - T2) + p.d2*H + p.e2;
dT3 = -p.a3*(T3 - p.Tamb) + p.b3*u3 + p.g3C*(C - T3) + p.e3;
dT4 = -p.a4*(T4 - p.Tamb) + p.b4*u4 + p.g4C*(C - T4) + p.e4;

dC = -p.aC*(C - p.Tamb) + p.gC1*(T1 - C) + p.gC2*(T2 - C) + p.gC3*(T3 - C) + p.gC4*(T4 - C) + p.dC*H + p.eC;

dP = -p.aP*P + p.beta1*u1 + p.beta2*u2 + p.beta3*u3 + p.beta4*u4 + p.dP*H + p.eP;

dx = [dT1; dT2; dT3; dT4; dC; dP];
end


function cfgs = make_zero_return_pulses_rest(T_end, order, durations, ...
    abs_change_min, abs_change_max, ...
    u_min, u_max, ...
    rest_min, rest_max)
n = numel(order);

for k = 1:n
    nm = order{k};
    cfgs.(nm).start = [];
    cfgs.(nm).stop = [];
    cfgs.(nm).level = [];
end

tcur = 0.0;
iord = 1;

while tcur < T_end
    nm = order{iord};
    idx = iord;
    dur = durations.(nm);

    level = propose_level_absolute(0.0, abs_change_min.(nm), abs_change_max.(nm), u_min(idx), u_max(idx));

    t0 = tcur;
    t1 = min(t0 + dur, T_end);

    cfgs.(nm).start(end+1,1) = t0;
    cfgs.(nm).stop(end+1,1)  = t1;
    cfgs.(nm).level(end+1,1) = level;

    tcur = t1 + rand_between(rest_min, rest_max);
    iord = mod(iord, n) + 1;
end
end


function u = eval_step_zero_return_input(t, cfg, u_nom, u_min, u_max)
u = u_nom;

ne = numel(cfg.start);
for i = 1:ne
    if t >= cfg.start(i) && t < cfg.stop(i)
        u = cfg.level(i);
        break;
    end
end

u = clip_scalar(u, u_min, u_max);
end


function level = propose_level_absolute(base, abs_delta_min, abs_delta_max, lo, hi)
abs_delta_min = max(0, abs_delta_min);
abs_delta_max = max(abs_delta_min, abs_delta_max);

up_room = hi - base;
down_room = base - lo;

can_step_up = up_room >= abs_delta_min;
can_step_down = down_room >= abs_delta_min;

if ~can_step_up && ~can_step_down
    level = clip_scalar(base, lo, hi);
    return;
end

if can_step_up && can_step_down
    direction = 2 * (rand > 0.5) - 1;
elseif can_step_up
    direction = 1;
else
    direction = -1;
end

if direction > 0
    max_feasible = min(abs_delta_max, up_room);
else
    max_feasible = min(abs_delta_max, down_room);
end

delta_mag = rand_between(abs_delta_min, max_feasible);
level = clip_scalar(base + direction * delta_mag, lo, hi);
end


function A = numerical_jacobian_x(fx, x0)
n = numel(x0);
A = zeros(n,n);
epsx = 1e-6;
for i = 1:n
    dx = zeros(n,1);
    dx(i) = epsx;
    fp = fx(x0 + dx);
    fm = fx(x0 - dx);
    A(:,i) = (fp - fm) / (2*epsx);
end
end


function dx = rhs_thermal7_const_u(x, p, u)
T1 = x(1); T2 = x(2); T3 = x(3); T4 = x(4);
C  = x(5); P  = x(6);

u1 = u(1); u2 = u(2); u3 = u(3); u4 = u(4);
H = 0;

dT1 = -p.a1*(T1 - p.Tamb) + p.b1*u1 + p.g1C*(C - T1) + p.d1*H + p.e1;
dT2 = -p.a2*(T2 - p.Tamb) + p.b2*u2 + p.g2C*(C - T2) + p.d2*H + p.e2;
dT3 = -p.a3*(T3 - p.Tamb) + p.b3*u3 + p.g3C*(C - T3) + p.e3;
dT4 = -p.a4*(T4 - p.Tamb) + p.b4*u4 + p.g4C*(C - T4) + p.e4;

dC = -p.aC*(C - p.Tamb) + p.gC1*(T1 - C) + p.gC2*(T2 - C) + p.gC3*(T3 - C) + p.gC4*(T4 - C) + p.dC*H + p.eC;

dP = -p.aP*P + p.beta1*u1 + p.beta2*u2 + p.beta3*u3 + p.beta4*u4 + p.dP*H + p.eP;

dx = [dT1; dT2; dT3; dT4; dC; dP];
end


function [y_meas, meta_tbl] = apply_heat_measurement_noise(y_true, Ts, noise_mode, ...
    sigma_temp, sigma_power, ...
    quant_temp, quant_power, ...
    rw_sigma_temp, rw_sigma_power)

n = size(y_true, 1);
sigmas = [sigma_temp, sigma_temp, sigma_temp, sigma_temp, sigma_temp, sigma_power];
quants = [quant_temp, quant_temp, quant_temp, quant_temp, quant_temp, quant_power];
rw_sigmas = [rw_sigma_temp, rw_sigma_temp, rw_sigma_temp, rw_sigma_temp, rw_sigma_temp, rw_sigma_power];

switch noise_mode
    case 1
        y_meas = y_true + randn(size(y_true)) .* sigmas;
        meta_tbl = table(noise_mode, sigma_temp, sigma_power, ...
            'VariableNames', {'noise_mode','sigma_temp','sigma_power'});

    case 2
        y_noisy = y_true + randn(size(y_true)) .* sigmas;
        y_meas = quantize_columns(y_noisy, quants);
        meta_tbl = table(noise_mode, sigma_temp, sigma_power, quant_temp, quant_power, ...
            'VariableNames', {'noise_mode','sigma_temp','sigma_power','quant_temp','quant_power'});

    case 3
        drift = zeros(size(y_true));
        for j = 1:size(y_true, 2)
            for i = 2:n
                drift(i,j) = drift(i-1,j) + rw_sigmas(j) * Ts * randn;
            end
        end
        y_meas = y_true + drift;
        meta_tbl = table(noise_mode, rw_sigma_temp, rw_sigma_power, Ts, ...
            'VariableNames', {'noise_mode','rw_sigma_temp','rw_sigma_power','Ts'});

    otherwise
        error('Unsupported noise_mode = %d. Choose 1, 2, or 3.', noise_mode);
end
end


function yq = quantize_columns(y, deltas)
yq = y;
for j = 1:size(y, 2)
    yq(:,j) = deltas(j) * round(y(:,j) / deltas(j));
end
end


function [tag, desc] = get_noise_model_info(noise_mode)
switch noise_mode
    case 1
        tag = 'N1';
        desc = 'Gaussian measurement noise';
    case 2
        tag = 'N2';
        desc = 'quantized Gaussian measurement noise';
    case 3
        tag = 'N3';
        desc = 'random-walk measurement noise';
    otherwise
        error('Unsupported noise_mode = %d. Choose 1, 2, or 3.', noise_mode);
end
end


function T_meta = make_run_metadata_table(H_mode, H_tag, noise_mode, noise_tag, ...
    sigma_temp, sigma_power, quant_temp, quant_power, rw_sigma_temp, rw_sigma_power, Ts)
T_meta = table(H_mode, string(H_tag), noise_mode, string(noise_tag), ...
    sigma_temp, sigma_power, quant_temp, quant_power, rw_sigma_temp, rw_sigma_power, Ts, ...
    'VariableNames', {'H_mode','H_tag','noise_mode','noise_tag', ...
    'sigma_temp','sigma_power','quant_temp','quant_power','rw_sigma_temp','rw_sigma_power','Ts'});
end


function v = clip_scalar(v, lo, hi)
v = max(lo, min(hi, v));
end


function r = rand_between(a, b)
r = a + (b - a) * rand;
end


function [t_force, eta_vals, mode_tag, mode_desc] = make_H_profile(t_uniform, T_end, H_mode, p)
t_force = t_uniform(:);

switch H_mode
    case 1
        eta_vals = p.eta0 + p.eta_amp * sin(2*pi*p.eta_freq*t_force + p.eta_phase);
        mode_tag = 'Hsin';
        mode_desc = 'sinusoidal H';

    case 2
        t_half = 0.5 * T_end;
        eta_vals = p.decay_start + (p.decay_end - p.decay_start) * min(t_force / t_half, 1);
        mode_tag = 'Hdec';
        mode_desc = 'decreasing H';

    case 3
        eta_vals = make_bounded_brownian_profile( ...
            t_force, p.brown_mean, p.brown_sigma, p.brown_min, p.brown_max, p.brown_mean_pull);
        mode_tag = 'Hbrn';
        mode_desc = 'Brownian H with non-zero mean';

    case 4
        eta_vals = p.step_initial * ones(size(t_force));
        eta_vals(t_force >= p.step_time) = p.step_level;
        mode_tag = 'Hstep';
        mode_desc = 'midpoint step H';

    case 5
        eta_vals = make_prbs_like_profile(t_force, T_end, p.prbs_mean, p.prbs_amp_min, p.prbs_amp_max, p.prbs_hold_min, p.prbs_hold_max);
        mode_tag = 'Hprbs';
        mode_desc = 'PRBS-like H with non-zero mean';

    otherwise
        error('Unsupported H_mode = %d. Choose 1, 2, 3, 4, or 5.', H_mode);
end
end


function eta_vals = make_prbs_like_profile(t_force, T_end, mean_level, amp_min, amp_max, hold_min, hold_max)
eta_vals = zeros(size(t_force));
tcur = 0.0;
current_level = mean_level;

while tcur <= T_end
    hold_time = rand_between(hold_min, hold_max);
    sign_term = 2 * (rand > 0.5) - 1;
    amp_term = rand_between(amp_min, amp_max);
    next_level = mean_level + sign_term * amp_term;

    mask = t_force >= tcur & t_force < min(tcur + hold_time, T_end + eps);
    eta_vals(mask) = next_level;

    current_level = next_level;
    tcur = tcur + hold_time;
end

if ~any(eta_vals)
    eta_vals(:) = current_level;
end
end


function eta_vals = make_bounded_brownian_profile(t_force, mean_level, sigma, vmin, vmax, mean_pull)
eta_vals = zeros(size(t_force));
eta_vals(1) = clip_scalar(mean_level, vmin, vmax);

for i = 2:numel(t_force)
    dt = max(t_force(i) - t_force(i-1), 0);
    proposal = eta_vals(i-1) + sigma * sqrt(dt) * randn;
    proposal = proposal + mean_pull * (mean_level - eta_vals(i-1));
    eta_vals(i) = clip_scalar(proposal, vmin, vmax);
end
end


function C_tag = make_C_tag(cc, CC)
tol = 1e-12;
has_forward = abs(cc) > tol;
has_backward = abs(CC) > tol;

if has_forward && has_backward
    C_tag = 'Cbi';
elseif ~has_forward && has_backward
    C_tag = 'Cuni';
elseif ~has_forward && ~has_backward
    C_tag = 'Cnone';
else
    C_tag = 'Casym';
end
end


function duration_tag = make_duration_tag(durations)
names = fieldnames(durations);
vals = zeros(numel(names), 1);
for i = 1:numel(names)
    vals(i) = durations.(names{i});
end

if all(abs(vals - vals(1)) < 1e-12)
    duration_tag = sprintf('dur%g', vals(1));
else
    parts = cell(numel(vals), 1);
    for i = 1:numel(vals)
        parts{i} = sprintf('%g', vals(i));
    end
    duration_tag = sprintf('dur%s', strjoin(parts, '-'));
end
end
