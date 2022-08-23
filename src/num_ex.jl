using ML_Hopf
using DiffEqFlux
using LinearAlgebra
using PGFPlotsX
using DifferentialEquations
using LaTeXStrings
using Serialization

tol = 1e-8; #Zero tolerance
N = 100; # collocation points for continuation
sp = 800; # Number of continuation points
sU = 18.0; # Starting point of continuation (wind velocity)
ds = 0.02; # Arclength (note that this remains constant in Numerical_Cont.jl)
st = 1e-3
ind = [1, 3] #index of projecting plane (Not important if we do not use CBC)
BLAS.set_num_threads(1) # This prevents stack overfolowerror
slco = ML_Hopf.continuation_flutter(N, sp, sU, ds) # Perform numerical continuation for the training data
U = slco.V
P = slco.P
amp = ML_Hopf.amp_LCO(U, 1, N)
θ_l = 100

## Generate training data (Data of trajectory)
vel_l = 4
Vel = range(15.0; stop=18.0, length=vel_l)
nh = 10
s_ind = [400, 500, 600, 700] # index of the unstable LCO trainig data

Vel2 = [P[s_ind[i]] for i in 1:length(s_ind)]
##
U₀ = 18.27 # Initial guess of the flutter speed
s = 3.65 # Initial guess of the fold parameter
θ_l = 100; # Number of grid points on the circle

dat = ML_Hopf.generate_data(vel_l, Vel, nh, st, s_ind, U, N, θ_l) # This function generates data
AA = dat.data
Al = dat.data2
tl2 = 1.0
tt = Int(tl2 / st + 1)
t_series = [dat.ts[i][:, 1:tt] for i in 1:length(Vel)]
θ_series = [dat.theta_s[i][1:tt] for i in 1:length(Vel)]
u₀ = dat.d0[1]
v₀ = dat.d0[2]
u0 = [u₀, v₀]
siθ = dat.siθ;
coθ = dat.coθ;
uu0 = dat.d0
## generate unstable time-series
U2 = [U[s_ind[i]] for i in 1:length(s_ind)]
uu0_u = [ML_Hopf.get_sol(U2[j], 6, N, 1, 3).u[1, :] for j = 1:length(Vel2)]
uu_d = [ML_Hopf.get_sol(U2[j], 6, N, 1, 3).u for j = 1:length(Vel2)]
Kp = 0
Kd = 0
t_series2 = []
θ_series2 = []
for ind = 1:4
    g = ML_Hopf.get_sol(U2[ind], 6, 100, 1, 3)
    d = ML_Hopf.LS_harmonics(g.r, g.t, 1, nh)
    p1 = vcat([Vel2[ind], Kp, Kd], d.coeff)
    prob = ODEProblem(ML_Hopf.flutter_eq_CBC, uu0_u[ind], (0, tl2), p1)
    sol = solve(prob, Tsit5(), reltol=1e-7, abstol=1e-7, saveat=st)
    tss = transpose(hcat(ML_Hopf.get_sol2(sol, 1), ML_Hopf.get_sol2(sol, 3)))
    th_s = atan.(ML_Hopf.get_sol2(sol, 3), ML_Hopf.get_sol2(sol, 1))
    t_series2 = vcat(t_series2, [tss])
    θ_series2 = vcat(θ_series2, [th_s])
end

## Train Linear-transformation 

rot = -π * 0.1
R = [cos(rot) -sin(rot); sin(rot) cos(rot)]
θ = vec(1e-2 * R * [2.0 0.0; 0.0 3])
θ = vcat(θ, zeros(2))
scale_f_l = 1e2 # optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.
function loss_lt_(θ_t)
    return ML_Hopf.loss_lt(θ_t, U₀, s, Vel, Vel2, uu0, AA, scale_f_l, nh, θ_l, coθ, siθ)
end
res1 = DiffEqFlux.sciml_train(loss_lt_, θ, ADAM(0.01); maxiters=200)
θ_ = res1.minimizer
Ap = ML_Hopf.lt_pp(θ_, U₀, s, Vel, Vel2, scale_f_l, θ_l, coθ, siθ) # Plot the linear transformation

hidden = 21
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)
θn = initial_params(ann)
scale_f = 1e3
pp = [18.27, 3.67]
θn = vcat(pp, θn)

function loss_nt_(θ_t)
    return ML_Hopf.loss_nt(
        θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, AA, ann, nh, uu0, coθ, siθ
    )
end
res_n = DiffEqFlux.sciml_train(loss_nt_, θn, ADAM(0.01); maxiters=400)
θ_2 = res_n.minimizer

Ap = ML_Hopf.lt_pp_n(θ_2, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ)


#Checking the phase portrait of the model (Unstable LCO)
Ap = ML_Hopf.lt_pp_n_u(θ_2, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ)
ind = 1 # Near the fold
vv = Vel2[ind]
uu = ML_Hopf.get_sol(U[s_ind[ind]], N, 1, 3)



θ_t = [θ_; θ_2]
vel_l_ = 800
ind = 1
bd = ML_Hopf.lt_b_dia(θ_t, ind, vel_l_, ann, θ_l, coθ, siθ, scale_f_l, scale_f)
h = [maximum(t_series[i][1, :]) - minimum(t_series[i][1, :]) for i in 1:length(Vel)]
d_amp = [amp[s_ind[i]] for i in 1:length(s_ind)]
d_P = [P[s_ind[i]] for i in 1:length(s_ind)]
P = vec(P)
amp = vec(amp)
vv = vcat(bd.v, bd.v)
aa = vcat(bd.s, bd.u)


## Train the speed of the oscillation

function Inv_T_u(th0, vel) # This function gives initial conditions of the model
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    np1, np2 = θ_2[1:2]
    pn = θ_2[3:end]
    s_amp = sqrt(np2 / 2 + sqrt(np2^2 + 4 * (vel - np1)) / 2)
    theta = range(-π; stop=π, length=100000)

    uu = [[s_amp * cos(theta[i]), s_amp * sin(theta[i])] for i in 1:length(theta)]
    u = [
        [p5, p6] * norm(uu[i])^2 / scale_f_l +
        T * uu[i] +
        ann([uu[i]; vel - np1], pn) / scale_f for i in 1:length(theta)
    ]
    u2 = dat.d0[2]
    u1 = dat.d0[1]
    t0 = [abs(atan(u[i][2] - u2, u[i][1] - u1) - th0) for i in 1:length(theta)]
    er = minimum(t0)
    return theta[argmin(t0)]
end

function Inv_T_uu(th0, vel) # This function gives initial conditions of the model for unstable branch
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    np1, np2 = θ_2[1:2]
    pn = θ_2[3:end]
    s_amp = sqrt(np2 / 2 - sqrt(np2^2 + 4 * (vel - np1)) / 2)
    theta = range(-π; stop=π, length=100000)

    uu = [[s_amp * cos(theta[i]), s_amp * sin(theta[i])] for i in 1:length(theta)]
    u = [
        [p5, p6] * norm(uu[i])^2 / scale_f_l +
        T * uu[i] +
        ann([uu[i]; vel - np1], pn) / scale_f for i in 1:length(theta)
    ]
    u2 = 0
    u1 = 0
    t0 = [abs(atan(u[i][2] - u2, u[i][1] - u1) - th0) for i in 1:length(theta)]
    er = minimum(t0)
    return theta[argmin(t0)]
end

function dudt_ph(u, p, t)
    θ = u[1]
    r = u[2]
    c = u[3]
    np1, np2 = θ_2[1:2]
    a2 = np2
    δ₀ = np1
    ν = (c - δ₀)
    ω₀ = p[1]
    uu = [r * cos(θ), r * sin(θ), ν]
    du₁ = ω₀ + ann3(uu, p[2:end])[1] / om_scale
    du₂ = 0
    du₃ = 0
    return [du₁, du₂, du₃]
end

function predict_time_T(p) #,uu_t0
    np1, np2 = θ_2[1:2]
    pn = θ_2[3:end]
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, u_t0[i], (0, tl2), p),
                Tsit5(),
                u_t0[i],
                p;
                saveat=st,
                abstol=1e-8,
                reltol=1e-8,
                sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP())
            ),
        ) for i in 1:length(Vel)
    ]
    uu = [
        transpose(
            hcat(
                A1[i][2, :] .* cos.(A1[i][1, :]),
                A1[i][2, :] .* sin.(A1[i][1, :]),
                A1[i][3, :],
            ),
        ) for i in 1:length(Vel)
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i in 1:length(Vel)]
    dis = transpose([p5 * ones(spl) p6 * ones(spl)]) / scale_f_l

    vlT = [
        dis * norm(uu[i][1:2])^2 +
        T * uu[i][1:2, :] +
        ML_Hopf.Array_chain(uu[i], ann, pn) / scale_f for i in 1:length(Vel)
    ]
    Pr = zeros(0, spl)
    for i in 1:length(Vel)
        theta = vlT[i][[1, 2], :]
        Pr = vcat(Pr, theta)
    end

    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, uu_t0[i], (0, tl2), p),
                Tsit5(),
                uu_t0[i],
                p,
                saveat=st,
                abstol=1e-8,
                reltol=1e-8,
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
            ),
        ) for i = 1:length(Vel2)
    ]
    uu = [
        transpose(
            hcat(
                A1[i][2, :] .* cos.(A1[i][1, :]),
                A1[i][2, :] .* sin.(A1[i][1, :]),
                A1[i][3, :],
            ),
        ) for i = 1:length(Vel2)
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i = 1:1:length(Vel2)]

    vl = [uu[i][1:2, :] for i = 1:length(Vel2)]
    vlT = [
        dis * norm(uu[i][1:2])^2 +
        T * uu[i][1:2, :] +
        ML_Hopf.Array_chain(uu[i], ann, pn) / scale_f for i in 1:length(Vel2)]
    Pr2 = zeros(0, spl)
    for i = 1:length(Vel2)
        theta = vlT[i][[1, 2], :]
        Pr2 = vcat(Pr2, theta)
    end

    return [Pr; Pr2]
end

function predict_time_T_ut(p) #,uu_t0
    np1, np2 = θ_2[1:2]
    pn = θ_2[3:end]
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, u_t03[i], (0, tl2), p),
                Tsit5(),
                u_t0[i],
                p;
                saveat=st,
                abstol=1e-8,
                reltol=1e-8,
                sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP())
            ),
        ) for i in 1:length(Vel3)
    ]
    uu = [
        transpose(
            hcat(
                A1[i][2, :] .* cos.(A1[i][1, :]),
                A1[i][2, :] .* sin.(A1[i][1, :]),
                A1[i][3, :],
            ),
        ) for i in 1:length(Vel3)
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i in 1:length(Vel3)]
    dis = transpose([p5 * ones(spl) p6 * ones(spl)]) / scale_f_l

    vlT = [
        dis * norm(uu[i][1:2])^2 +
        T * uu[i][1:2, :] +
        ML_Hopf.Array_chain(uu[i], ann, pn) / scale_f for i in 1:length(Vel3)
    ]
    Pr = zeros(0, spl)
    for i in 1:length(Vel3)
        theta = vlT[i][[1, 2], :]
        Pr = vcat(Pr, theta)
    end
    return Pr
end

function loss_time_T(p)
    pred = predict_time_T(p)
    return sum(abs2, A3 .- pred)
end
## calculate the initial condition of the model that corresponds to the initial training time-series
tl2 = 1.0
spl = Int(tl2 / st + 1)
vl = length(Vel)
hidden = 31
ann3 = FastChain(FastDense(3, hidden, tanh), FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega = 15.3
p = vcat(omega, np)

np1, np2 = θ_2[1:2]
pn = θ_2[3:end]
s_amp = [sqrt(np2 / 2 + sqrt(np2^2 + 4 * (Vel[i] - np1)) / 2) for i in 1:length(Vel)]
theta0 = [θ_series[i][1] for i in 1:length(Vel)]
θ₀ = [Inv_T_u(theta0[i], Vel[i]) for i in 1:length(Vel)]
u_t0 = [[θ₀[i], s_amp[i], Vel[i]] for i in 1:length(Vel)]

u_amp = [sqrt(np2 / 2 - sqrt(np2^2 + 4 * (Vel2[i] - np1)) / 2) for i in 1:length(Vel2)]
theta0u = [θ_series2[i][1] for i in 1:length(Vel2)]
θ₀u = [Inv_T_uu(theta0u[i], Vel2[i]) for i in 1:length(Vel2)]
uu_t0 = [[θ₀u[i], u_amp[i], Vel2[i]] for i in 1:length(Vel2)]

vl2 = length(Vel2)
vl = length(Vel)
spl = length(t_series[1][1, :])
t_s = zeros(vl * 2 * 2, spl)
for i in 1:vl
    t_s[[2 * (i - 1) + 1, 2 * (i - 1) + 2], :] = t_series[i]
end
for i in vl+1:(vl2+vl)
    t_s[[2 * (i - 1) + 1, 2 * (i - 1) + 2], :] = t_series2[i-vl]
end
A3 = t_s

om_scale = 0.3
loss_time_T(p)
res_t = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01); maxiters=300)
res_t = DiffEqFlux.sciml_train(
    loss_time_T, res_t.minimizer, BFGS(; initial_stepnorm=1e-3); maxiters=2000
)
p = res_t.minimizer
res_t.minimum
tv = range(0, 1; length=1001)


## plot untrained time series prediction
s_ind2 = [110, 200, 350, 550]
Vel3 = [P[s_ind2[i]] for i = 1:length(s_ind2)]
U3 = [U[s_ind2[i]] for i in 1:length(s_ind2)]
uu0_3 = [ML_Hopf.get_sol(U3[j], 6, N, 1, 3).u[1, :] for j = 1:length(Vel3)]
uu_d3 = [ML_Hopf.get_sol(U3[j], 6, N, 1, 3).u for j = 1:length(Vel3)]
ind = 1
Kp = 0
Kd = 0
t_series3 = []
θ_series3 = []
for ind = 1:length(s_ind2)
    g = ML_Hopf.get_sol(U3[ind], 6, 100, 1, 3)
    d = ML_Hopf.LS_harmonics(g.r, g.t, 1, nh)
    p1 = vcat([Vel3[ind], Kp, Kd], d.coeff)
    prob = ODEProblem(ML_Hopf.flutter_eq_CBC, uu0_3[ind], (0, tl2), p1)
    sol = solve(prob, Tsit5(), reltol=1e-7, abstol=1e-7, saveat=st)
    tss = transpose(hcat(ML_Hopf.get_sol2(sol, 1), ML_Hopf.get_sol2(sol, 3)))
    t_series3 = vcat(t_series3, [tss])
    if ind < 3
        th_s = atan.(ML_Hopf.get_sol2(sol, 3) - ones(length(sol.t)) * dat.d0[2], ML_Hopf.get_sol2(sol, 1) - ones(length(sol.t)) * dat.d0[1])
    else
        th_s = atan.(ML_Hopf.get_sol2(sol, 3), ML_Hopf.get_sol2(sol, 1))
    end
    θ_series3 = vcat(θ_series3, [th_s])
end

amp1 = [sqrt(np2 / 2 + sqrt(np2^2 + 4 * (Vel3[i] - np1)) / 2) for i in 1:2]
amp2 = [sqrt(np2 / 2 - sqrt(np2^2 + 4 * (Vel3[i] - np1)) / 2) for i in 3:4]
amp3 = vcat(amp1, amp2)
theta03 = [θ_series3[i][1] for i in 1:length(Vel3)]
θ₀31 = [Inv_T_u(theta03[i], Vel3[i]) for i in 1:2]
θ₀32 = [Inv_T_uu(theta03[i], Vel3[i]) for i in 3:4]
θ₀3 = vcat(θ₀31, θ₀32)
u_t03 = [[θ₀3[i], amp3[i], Vel3[i]] for i in 1:length(Vel3)]

t_s3 = zeros(length(Vel3) * 2, spl)
for i in 1:length(Vel3)
    t_s3[[2 * (i - 1) + 1, 2 * (i - 1) + 2], :] = t_series3[i]
end

Ap_u = predict_time_T_ut(p)

newdata = deserialize("num_flutter_pp.jls")
Ap_u = newdata.Ap;
t_s3 = newdata.ts;
tv = newdata.tv;
bd = newdata.bd;
Vel = newdata.Vel;
h = newdata.h;
d_p = newdata.d_P;
d_amp = newdata.d_amp;
P = newdata.P;
amp = newdata.amp;

## Plot figures
ii = 1
d = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$h$ (m)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 4.5e-2,
        ymin = -4e-2,
        ylabel_shift = "-10pt",
        title = "(b-2)"},
    Plot(
        {color = "red", no_marks}, Coordinates(tv, Ap_u[2*(ii-1)+1, :])
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, t_s3[2*(ii-1)+1, :])),
)
@pgf d["every axis title/.style"] = "below right,at={(0,1)}";

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north east",
        height = "6cm",
        width = "6cm",
        xmin = -0.04,
        xmax = 0.04,
        ymax = 0.1,
        ymin = -0.11,
        ytick = [0.1, 0, -0.1],
        ylabel_shift = "-15pt",
        xlabel_shift = "-3pt",
        title = "(b-1)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap_u[2*(ii-1)+1, :], Ap_u[2*(ii-1)+2, :])),
    #  LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(t_s3[2*(ii-1)+1, :], t_s3[2*(ii-1)+2, :]),
    ),
    # LegendEntry("Ground truth"),
)
@pgf a["every x tick scale label/.style"] = "at={(rel axis cs:0.8,-0.15)},anchor=south west,inner sep=-1pt";
@pgf a["every axis title/.style"] = "below right,at={(0,1)}";
d

ii = 3
e = @pgf Axis(
    {
        xlabel = "Time (sec)",
        #  ylabel = L"$h$ (m)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 4.5e-2,
        ymin = -4e-2,
        ylabel_shift = "-10pt",
        title = "(c-2)"
    },
    Plot(
        {color = "red", no_marks}, Coordinates(tv, Ap_u[2*(ii-1)+1, :])
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, t_s3[2*(ii-1)+1, :])),
)
@pgf e["every axis title/.style"] = "below right,at={(0,1)}";


b = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        #   ylabel = L"$\alpha$ (rad)",
        legend_pos = "north east",
        height = "6cm",
        width = "6cm",
        xmin = -0.04,
        xmax = 0.04,
        ymax = 0.1,
        ymin = -0.11,
        ytick = [0.1, 0, -0.1],
        ylabel_shift = "-15pt",
        xlabel_shift = "-3pt",
        title = "(c-1)"},
    Plot({color = "red", no_marks}, Coordinates(Ap_u[2*(ii-1)+1, :], Ap_u[2*(ii-1)+2, :])),
    #  LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(t_s3[2*(ii-1)+1, :], t_s3[2*(ii-1)+2, :]),
    ),
    # LegendEntry("Ground truth"),
)
@pgf b["every x tick scale label/.style"] = "at={(rel axis cs:0.8,-0.15)},anchor=south west,inner sep=-1pt";
@pgf b["every axis title/.style"] = "below right,at={(0,1)}";

ii = 4
f = @pgf Axis(
    {
        xlabel = "Time (sec)",
        # ylabel = L"$h$ (m)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 4.5e-2,
        ymin = -4e-2,
        ylabel_shift = "-10pt",
        title = "(d-2)"},
    Plot(
        {color = "red", no_marks}, Coordinates(tv, Ap_u[2*(ii-1)+1, :])
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, t_s3[2*(ii-1)+1, :])),
)
@pgf f["every axis title/.style"] = "below right,at={(0,1)}";


c = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        #  ylabel = L"$\alpha$ (rad)",
        legend_pos = "north east",
        height = "6cm",
        width = "6cm",
        xmin = -0.04,
        xmax = 0.04,
        ymax = 0.1,
        ymin = -0.11,
        ytick = [0.1, 0, -0.1],
        ylabel_shift = "-15pt",
        xlabel_shift = "-3pt",
        title = "(d-1)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap_u[2*(ii-1)+1, :], Ap_u[2*(ii-1)+2, :])),
    #  LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(t_s3[2*(ii-1)+1, :], t_s3[2*(ii-1)+2, :]),
    ),
    # LegendEntry("Ground truth"),
)
@pgf c["every x tick scale label/.style"] = "at={(rel axis cs:0.8,-0.15)},anchor=south west,inner sep=-1pt";
@pgf c["every axis title/.style"] = "below right,at={(0,1)}";


gp = @pgf GroupPlot(
    {group_style = {group_size = "3 by 3"},
        no_markers,
        legend_pos = "north west",
        xlabel = raw"$x$",
    },
    a, b, c, d, e, f)

gp

##
bd = @pgf Axis(
    {
        xlabel = "Air speed (m/sec)",
        ylabel = "Heave amplitude (m)",
        legend_pos = "north west",
        height = "7cm",
        width = "11cm",
        ymin = 0,
        ymax = 8e-2,
        mark_options = {scale = 1.5},
        title = "(a)"
    },
    Plot({color = "blue", only_marks}, Coordinates(Vel, h)),
    Plot({color = "red", only_marks, mark = "triangle*"}, Coordinates(d_P, d_amp)),
    Plot({color = "blue", no_marks}, Coordinates(vec(P), amp)),
    Plot({color = "red", no_marks}, Coordinates(bd.v, bd.s)),
    Plot({color = "red", no_marks}, Coordinates(bd.v, bd.u)),
)
@pgf bd["every axis title/.style"] = "below right,at={(0,1)}";
bd
##
pgfsave("./Figure/num_flutter/bd_num.pdf", bd)
pgfsave("./Figure/num_flutter/gp_num.pdf", gp)


#data_flutter=(Ap=Ap_u,ts=t_s3,tv=tv,bd=bd,Vel=Vel,h=h,d_P=d_P,d_amp=d_amp,P=P,amp=amp)
#serialize("num_flutter_pp.jls",data_flutter)
#newdata = deserialize("num_flutter_pp.jls")


