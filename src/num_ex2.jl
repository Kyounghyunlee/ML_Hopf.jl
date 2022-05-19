using ML_Hopf
using LinearAlgebra
using Statistics
using DiffEqFlux
using PGFPlotsX
using LaTeXStrings
using DifferentialEquations

tol = 1e-8; #Zero tolerance
N = 100; # collocation points for continuation
sp = 800; # Number of continuation points
sU = 18.0; # Starting point of continuation (wind velocity)
ds = 0.02; # Arclength (note that this remains constant in Numerical_Cont.jl)
st = 1e-3
ind = [1, 3] #index of projecting plane (Not important if we do not use CBC)
BLAS.set_num_threads(1) # This prevents stack overfolowerror
slco = ML_Hopf.continuation_flutter(N, sp, sU, ds)
U = slco.V
P = slco.P
amp = ML_Hopf.amp_LCO(U, 1, N)
θ_l = 100

## Generating training data (Data of trajectory)
vel_l = 5
Vel = range(15.0; stop=18.0, length=vel_l)
nh = 10
s_ind = [600, 700]

Vel2 = [P[s_ind[i]] for i in 1:length(s_ind)]
U₀ = 18.27 #Estimated flutter speed
s = 3.65
θ_l = 100;

dat = ML_Hopf.generate_data(vel_l, Vel, nh, st, s_ind, U, N, θ_l)

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
Ap = ML_Hopf.lt_pp(θ_, U₀, s, Vel, Vel2, scale_f_l, θ_l, coθ, siθ)

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[1][1, :], Ap[1][2, :])),
    #LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(t_series[1][1, :], t_series[1][2, :])),
    #LegendEntry("Ground truth"),
)

pgfsave("./Figures/num_flutter/LTU15_5_2.pdf", a)

## Train nonlinear transformation
hidden = 21
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)
θn = initial_params(ann)
scale_f = 1e3
pp = [18.27, 3.65]
θn = vcat(pp, θn)
function loss_nt_(θ_t)
    return ML_Hopf.loss_nt(
        θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, AA, ann, nh, uu0, coθ, siθ
    )
end
res_n = DiffEqFlux.sciml_train(loss_nt_, θn, ADAM(0.01); maxiters=300)
θ_2 = res_n.minimizer

Ap = ML_Hopf.lt_pp_n(θ_2, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ)
ind = 1
a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3e-2,
        xmax = 3e-2,
        ymax = 6e-2,
        ymin = -9e-2,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :], Ap[ind][2, :])),
    #LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(t_series[ind][1, :], t_series[ind][2, :])),
    #LegendEntry("Ground truth"),
)
pgfsave("./Figures/num_flutter/NNU15_2.pdf", a)

#Checking the phase portrait of the model (Unstable LCO)
Ap = ML_Hopf.lt_pp_n_u(θ_2, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ)
ind = 1 # Near the fold
vv = Vel2[ind]
uu = ML_Hopf.get_sol(U[s_ind[ind]], N, 1, 3)

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3e-2,
        xmax = 3e-2,
        ymax = 6e-2,
        ymin = -9e-2,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[1][1, :], Ap[1][2, :])),
   # LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(uu.u[1, :], uu.u[1, 1]), vcat(uu.u[3, :], uu.u[3, 1])),
    ),
    #LegendEntry("Ground truth"),
)

pgfsave("./Figures/num_flutter/ust_u17_2.pdf", a)

##
ind = 2
vv = Vel2[ind]# Near the equilibrium
uu = ML_Hopf.get_sol(U[s_ind[ind]], 50, 1, 3)

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3e-2,
        xmax = 3e-2,
        ymax = 6e-2,
        ymin = -9e-2,
  #      xtick = -1e-2:4e-3:1e-2,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[2][1, :], Ap[2][2, :])),
    #LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(uu.u[1, :], uu.u[1, 1]), vcat(uu.u[3, :], uu.u[3, 1])),
    ),
    #LegendEntry("Ground truth"),
)
pgfsave("./Figures/num_flutter/ust_u179_2.pdf", a)

##
u_ind = [400, 750]
Vel3 = [P[u_ind[i]] for i in 1:length(u_ind)]
Ap = ML_Hopf.lt_pp_n_u(θ_2, θ_, θ_l, scale_f_l, Vel, Vel3, scale_f, ann, coθ, siθ)

ind = 1
vv = Vel3[ind]
uu = ML_Hopf.get_sol(U[u_ind[ind]], N, 1, 3)

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3e-2,
        xmax = 3e-2,
        ymax = 6e-2,
        ymin = -9e-2,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[1][1, :], Ap[1][2, :])),
    #LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(uu.u[1, :], uu.u[1, 1]), vcat(uu.u[3, :], uu.u[3, 1])),
    ),
    #LegendEntry("Ground truth"),
)
pgfsave("./Figures/num_flutter/ust_u153_2.pdf", a)

##
ind = 2 # Near the equilibrium
vv = Vel3[ind]
uu = ML_Hopf.get_sol(U[u_ind[ind]], 50, 1, 3)

a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north east",
        height = "9cm",
        width = "9cm",
        xmin = -3e-2,
        xmax = 3e-2,
        ymax = 6e-2,
        ymin = -9e-2,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[2][1, :], Ap[2][2, :])),
  #  LegendEntry("Learnt model"),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(uu.u[1, :], uu.u[1, 1]), vcat(uu.u[3, :], uu.u[3, 1])),
    ),
   # LegendEntry("Ground truth"),
)
pgfsave("./Figures/num_flutter/ust_u1813_2.pdf", a)


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

#Plot bifurcation diagram
a = @pgf Axis(
    {
        xlabel = "Air speed (m/sec)",
        ylabel = "Heave amplitude (m)",
        legend_pos = "north west",
        height = "11cm",
        width = "15cm",
        ymin = 0,
        ymax = 8e-2,
        mark_options = {scale = 1.5},
    },
    Plot({color = "blue", only_marks}, Coordinates(Vel, h)),
   # LegendEntry("Training data  (stable LCO)"),
    Plot({color = "red", only_marks, mark = "triangle*"}, Coordinates(d_P, d_amp)),
   # LegendEntry("Training data (unstable LCO)"),
    Plot({color = "blue", no_marks}, Coordinates(vec(P), amp)),
   # LegendEntry("Ground truth"),
    Plot({color = "red", no_marks}, Coordinates(bd.v, bd.s)),
   # LegendEntry("Learnt model"),
    Plot({color = "red", no_marks}, Coordinates(bd.v, bd.u)),
)

pgfsave("./Figures/num_flutter/bd_flutter2.pdf", a)


## Train speed of the phase

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
    u2=dat.d0[2];u1=dat.d0[1]
    t0 = [abs(atan(u[i][2]-u2, u[i][1]-u1) - th0) for i in 1:length(theta)]
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
                sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP()),
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
        ) for i in 1:vl
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i in 1:vl]
    dis = transpose([p5 * ones(spl) p6 * ones(spl)]) / scale_f_l

    vlT = [
        dis * norm(uu[i][1:2])^2 +
        T * uu[i][1:2, :] +
        ML_Hopf.Array_chain(uu[i], ann, pn) / scale_f for i in 1:vl
    ]
    Pr = zeros(0, spl)
    for i in 1:vl
        theta = vlT[i][[1, 2], :]
        Pr = vcat(Pr, theta)
    end
    return Pr
end

function loss_time_T(p)
    pred = predict_time_T(p)
    return sum(abs2, A3 .- pred)
end
##
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

spl = length(t_series[1][1, :])
t_s = zeros(vl * 2, spl)
for i in 1:vl
    t_s[[2 * (i - 1) + 1, 2 * (i - 1) + 2], :] = t_series[i]
end
A3 = t_s

om_scale = 0.3
loss_time_T(p)
res_t = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01); maxiters=100)

res_t = DiffEqFlux.sciml_train(
    loss_time_T, res_t.minimizer, BFGS(; initial_stepnorm=1e-3); maxiters=10000
)

p = res_t.minimizer

tv = range(0, 1; length=1001)

ii=1
a = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$h$ (m)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = 0,
        xmax = 1,
        ymax = 4.5e-2,
        ymin = -4e-2,
        mark_options = {scale = 0.1},
    },
    Plot({color = "red", no_marks}, Coordinates(tv, predict_time_T(p)[2 * (ii - 1) + 1, :])),
   # LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(tv, A3[2 * (ii - 1) + 1, :])),
    #LegendEntry("Ground truth"),
)

pgfsave("./Figures/num_flutter/time_u15_2.pdf", a)

ii=5
a = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$h$ (m)",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = 0,
        xmax = 1,
        ymax = 4.5e-2,
        ymin = -4e-2,
    },
    Plot(
        {color = "red", no_marks}, Coordinates(tv, predict_time_T(p)[2 * (ii - 1) + 1, :])
    ),
    #LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(tv, A3[2 * (ii - 1) + 1, :])),
    #LegendEntry("Ground truth"),
)
pgfsave("./Figures/num_flutter//time_u18_2.pdf", a)
