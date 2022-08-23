using ML_Hopf
using LinearAlgebra
using Statistics
using MAT
using DiffEqFlux
using PGFPlotsX
using LaTeXStrings
using DifferentialEquations
using Serialization

#= Use to format code in BlueStyle
using JuliaFormatter
format("./src/phys_ex.jl", BlueStyle())
format("./src/num_ex.jl", BlueStyle())
format("./src/vdp_ex.jl", BlueStyle())
format("./src/ML_Hopf.jl", BlueStyle())
=#

## Save data
nh = 30
l = 6000
vars = matread("./src/measured_data/CBC_stable_v14_9.mat")
uu = get(vars, "data", 1)
ind1 = 1;
ind2 = 4;
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
mu1 = mean(uu[ind1, 1:l])
mu2 = mean(uu[ind2, 1:l])
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = [[transpose(uu1); transpose(uu2)]]
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = c

vars = matread("./src/measured_data/CBC_stable_v15_6.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)

vars = matread("./src/measured_data/CBC_stable_v16_5.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)

vars = matread("./src/measured_data/CBC_stable_v17_3.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)

vars = matread("./src/measured_data/CBC_unstable_v14_9.mat")
uu = get(vars, "data", 1)
ind1 = 1;
ind2 = 4;
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)

vars = matread("./src/measured_data/CBC_unstable_v15_6.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)

vars = matread("./src/measured_data/CBC_unstable_v16_5.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series = vcat(t_series, [[transpose(uu1); transpose(uu2)]])
rr = sqrt.(uu1 .^ 2 + uu2 .^ 2)
tt = atan.(uu2, uu1)
c = ML_Hopf.LS_harmonics(rr, tt, 1, nh).coeff
AA = hcat(AA, c)
##
vel_l = 4
Vel = [14.9, 15.6, 16.5, 17.3]
Vel2 = [14.9, 15.6, 16.5]
θ_l = 300
θ = range(0, stop = 2π, length = θ_l)
coθ = cos.(θ)
siθ = sin.(θ)
## Linear transformation
rot = π * 0.1
R = [cos(rot) -sin(rot); sin(rot) cos(rot)]
θ = vec(1e-2 * R * [8.0 0.0; 0.0 1.7])
θ = vcat(θ, zeros(2))
scale_f_l = 1e1 # optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.

hidden = 12
ann_l = FastChain(
    FastDense(2, hidden, tanh),
    FastDense(hidden, hidden, tanh),
    FastDense(hidden, 4),
)
θl = initial_params(ann_l)
scale_f2 = 1e2
θ = vcat(θ, θl)
pp = [17.85, 3.65]
θ = vcat(θ, pp)

## Nonlinear transformation
hidden = 11
ann = FastChain(
    FastDense(3, hidden, tanh),
    FastDense(hidden, hidden, tanh),
    FastDense(hidden, 2),
)
θn = initial_params(ann)
scale_f = 1e3

loss_lt_nn_(θ_t) =
    ML_Hopf.loss_lt_nn(θ_t, ann_l, scale_f_l, scale_f2, AA, θ_l, nh, Vel, Vel2, coθ, siθ)
res_l = DiffEqFlux.sciml_train(loss_lt_nn_, θ, ADAM(0.001), maxiters = 1000)
U₀ = res_l.minimizer[end-1];
s_ = res_l.minimizer[end];
loss_lt_nn_2(θ_t) = ML_Hopf.loss_lt_nn2(
    θ_t,
    ann_l,
    scale_f_l,
    scale_f2,
    U₀,
    s_,
    AA,
    θ_l,
    nh,
    Vel,
    Vel2,
    coθ,
    siθ,
)
#res_l = DiffEqFlux.sciml_train(loss_lt_nn_2,res_l.minimizer,BFGS(initial_stepnorm = 1e-4),maxiters = 200,) # First, train the simple model
θ_ = res_l.minimizer
θn = vcat(θn, [U₀, s_])
loss_nt_(θ_t) = ML_Hopf.loss_nt2(
    θ_t,
    θ_,
    ann_l,
    scale_f,
    scale_f2,
    scale_f_l,
    ann,
    AA,
    nh,
    Vel,
    Vel2,
    θ_l,
    coθ,
    siθ,
)
res1 = DiffEqFlux.sciml_train(loss_nt_, θn, ADAM(0.01), maxiters = 1000) # Train more complicated model
res1 = DiffEqFlux.sciml_train(loss_nt_, res1.minimizer, BFGS(initial_stepnorm = 1e-4), maxiters = 1000) # Train more complicated model
res1 = DiffEqFlux.sciml_train(loss_nt_, res1.minimizer, BFGS(initial_stepnorm = 1e-4), maxiters = 2000) # Train more complicated model


U₀ = res1.minimizer[end-1];
s_ = res1.minimizer[end];
θ_n = res1.minimizer
## Define parameters of the model as global variable
np1 = θ_n[end-1];
np2 = θ_n[end];
nf = ML_Hopf.nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
vl = nf.v;
vl2 = nf.v2;
p1, p2, p3, p4, p5, p6 = θ_[1:6]
T = [p1 p3; p2 p4]
pn1 = θ_[7:end-2]
pn2 = θ_n[1:end-2]


## Plot bifurcation diagram
#bd = ML_Hopf.lt_b_dia2(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ)
bd = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ,0.0)
h = [maximum(t_series[i][1, :]) - minimum(t_series[i][1, :]) for i = 1:length(Vel)]
h2 = [maximum(t_series[i+4][1, :]) - minimum(t_series[i+4][1, :]) for i = 1:length(Vel2)]
sens=18
a = @pgf Axis(
    {
        xlabel = "Wind speed (m/sec)",
        ylabel = "Heave amplitude (m)",
        legend_pos = "north west",
        height = "7cm",
        width = "11cm",
        ymin = 0,
        ymax = 10,
        mark_options = {scale = 1.5},
    },
    Plot({color = "blue", only_marks}, Coordinates(Vel, h * sens)),
    Plot({color = "red", only_marks, mark = "triangle*"}, Coordinates(Vel2, h2 * sens)),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(reverse(bd.v), bd.v2), vcat(reverse(bd.u * sens), bd.s * sens)),
    ),
)
pgfsave("./Figures/exp/bd_exp.pdf",a)

plot(vcat(reverse(bd.v), bd.v2),vcat(reverse(bd.u * sens), bd.s * sens))
scatter!(Vel,h*sens)

## Train speed of the phase
function Inv_T_u(th0, vel)
    s_amp = sqrt(np2 / 2 + sqrt(np2^2 + 4 * (vel - np1)) / 2)
    ttl = Int(1e5)
    theta = range(0, 2π, length = ttl)
    uu = [transpose(s_amp * cos.(theta)); transpose(s_amp * sin.(theta))]
    dis = transpose([p5 * ones(ttl) p6 * ones(ttl)]) / scale_f_l
    Tu = ML_Hopf.Nt(uu, T, vel, dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
    t0 = [abs(atan(Tu[2, i], Tu[1, i]) - th0) for i = 1:length(theta)]
    er = minimum(t0)
    return theta[argmin(t0)]
end

function Inv_T_uu(th0, vel)
    u_amp = sqrt(np2 / 2 + sqrt(np2^2 - 4 * (vel - np1)) / 2)
    ttl = Int(1e4)
    theta = range(0, 2π, length = ttl)
    uu = [transpose(u_amp * cos.(theta)); transpose(u_amp * sin.(theta))]
    dis = transpose([p5 * ones(ttl) p6 * ones(ttl)]) / scale_f_l
    Tu = ML_Hopf.Nt(uu, T, vel, dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
    t0 = [abs(atan(Tu[2, i], Tu[1, i]) - th0) for i = 1:length(theta)]
    er = minimum(t0)
    return theta[argmin(t0)]
end

function dudt_ph(u, p, t) # speed of phase
    np1 = U₀
    np2 = s_
    θ = u[1]
    r = u[2]
    c = u[3]
    δ₀ = np1
    ν = (c - δ₀)
    ω₀ = p[1]
    uu = [r * cos(θ), r * sin(θ), ν]
    du₁ = ω₀ + ann3(uu, p[2:end])[1] / om_scale
    du₂ = 0
    du₃ = 0
    [du₁, du₂, du₃]
end

function predict_time_T(p) # Predict the time series of the model with computed initial conditions
    #vel_l=300
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, u_t0[i], (0, tl2), p),
                Tsit5(),
                u_t0[i],
                p,
                saveat = st,
                abstol = 1e-8,
                reltol = 1e-8,
                sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
            ),
        ) for i = 1:length(Vel)
    ]
    uu = [
        transpose(
            hcat(
                A1[i][2, :] .* cos.(A1[i][1, :]),
                A1[i][2, :] .* sin.(A1[i][1, :]),
                A1[i][3, :],
            ),
        ) for i = 1:length(Vel)
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i = 1:length(Vel)]
    vl = [uu[i][1:2, :] for i = 1:length(Vel)]
    vlT = [
        ML_Hopf.Nt(vl[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
        for i = 1:length(Vel)
    ]
    Pr = zeros(0, spl)
    for i = 1:length(Vel)
        theta = vlT[i][[1, 2], :]
        Pr = vcat(Pr, theta)
    end
    ##
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, uu_t0[i], (0, tl2), p),
                Tsit5(),
                uu_t0[i],
                p,
                saveat = st,
                abstol = 1e-8,
                reltol = 1e-8,
                sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP()),
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
        ML_Hopf.Nt(vl[i], T, Vel2[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for i = 1:length(Vel2)
    ]
    Pr2 = zeros(0, spl)
    for i = 1:length(Vel2)
        theta = vlT[i][[1, 2], :]
        Pr2 = vcat(Pr2, theta)
    end
    ##
    [Pr; Pr2]
end
##
tl2 = 1.0
st = 1e-3
spl = Int(tl2 / st + 1)
vl = 7
t_s = zeros(vl * 2, spl)
t_series2 = [t_series[j][:, 1:5:end] for j = 1:7]

for i = 1:vl
    t_s[[2 * (i - 1) + 1, 2 * (i - 1) + 2], :] = t_series2[i][:, 1:1001]
end
A3 = t_s
## Generate data and initial θ
np1 = U₀;
np2 = s_;
s_amp = [sqrt(np2 / 2 + sqrt(np2^2 + 4 * (Vel[i] - np1)) / 2) for i = 1:length(Vel)]
u_amp = [sqrt(np2 / 2 - sqrt(np2^2 + 4 * (Vel2[i] - np1)) / 2) for i = 1:length(Vel2)]
theta0 = [atan(t_series[i][2, 1], t_series[i][1, 1]) for i = 1:length(Vel)]
theta0u = [
    atan(t_series[i+length(Vel)][2, 1], t_series[i+length(Vel)][1, 1]) for
    i = 1:length(Vel2)
]
θ₀ = [Inv_T_u(theta0[i], Vel[i]) for i = 1:length(Vel)]
θ₀u = [Inv_T_uu(theta0u[i], Vel2[i]) for i = 1:length(Vel2)]
u_t0 = [[θ₀[i], s_amp[i], Vel[i]] for i = 1:length(Vel)]
uu_t0 = [[θ₀u[i], u_amp[i], Vel2[i]] for i = 1:length(Vel2)]

function loss_time_T(p) # Loss function for the time series
    pred = predict_time_T(p)
    sum(abs2, A3 .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

om_scale = 0.3
tl2 = 1.0
st = 1e-3
spl = Int(tl2 / st + 1)
dis = transpose([p5 * ones(spl) p6 * ones(spl)]) / scale_f_l

vl = length(Vel)
hidden = 21
ann3 = FastChain(FastDense(3, hidden, tanh), FastDense(hidden, 1, tanh))
np = initial_params(ann3)
omega = 16.3
p = vcat(omega, np)

loss_time_T(p)
res_t = DiffEqFlux.sciml_train(loss_time_T, p, ADAM(0.01), maxiters = 500)

res_t = DiffEqFlux.sciml_train(
    loss_time_T,
    res_t.minimizer,
    BFGS(initial_stepnorm = 1e-3),
    maxiters = 500,
)
p = res_t.minimizer
res_t = DiffEqFlux.sciml_train(
    loss_time_T,
    res_t.minimizer,
    BFGS(initial_stepnorm = 1e-3),
    maxiters = 100,
)
time_p=predict_time_T(p)


sens=18


#data_exp=(θ_n=θ_n,θ_=θ_,p=p)
#serialize("exp_flutter_pp_full2.jls",data_exp)

newdata = deserialize("exp_flutter_pp_full2.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;


Ap = ML_Hopf.lt_pp_n2(
    θ_n,
    θ_,
    ann_l,
    scale_f,
    scale_f2,
    scale_f_l,
    ann,
    Vel,
    Vel2,
    θ_l,
    coθ,
    siθ,
)

## Plot time-series
sens=18
ind=4
a = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(a)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf a["every axis title/.style"] = "below right,at={(0,1)}";


ind=3
b = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
      #  ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(b)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf b["every axis title/.style"] = "below right,at={(0,1)}";


ind=2
c = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
      #  ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(c)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf c["every axis title/.style"] = "below right,at={(0,1)}";


ind=5
d = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(d)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf d["every axis title/.style"] = "below right,at={(0,1)}";

ind=6
e = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
     #   ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(e)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf e["every axis title/.style"] = "below right,at={(0,1)}";

ind=7
f = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
      #  ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = -6,
        xmax = 6,
        ymax = 9e-2,
        ymin = -9e-2,
        ylabel_shift="-10pt",
        xlabel_shift="-5pt",
        title="(f)"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[ind][1, :] * sens, Ap[ind][2, :])),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series[ind][1, :] * sens, t_series[ind][2, :]),
    ),
)
@pgf f["every axis title/.style"] = "below right,at={(0,1)}";

gp1=@pgf GroupPlot(
    { group_style = { group_size="3 by 3" },
      no_markers,
      legend_pos="north west",
      xlabel=raw"$x$",
    },
    a, b, c, d,e,f)

gp1


tl2 = 1.0
st = 1e-3
spl = Int(tl2 / st + 1)
tv = range(0, tl2, length = spl) #Generate a time vector

ind=4
a = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(a)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf a["every axis title/.style"] = "below right,at={(0,1)}";

ind=3
b = @pgf Axis(
    {
        xlabel = "Time (sec)",
     #   ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(b)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf b["every axis title/.style"] = "below right,at={(0,1)}";

ind=2
c = @pgf Axis(
    {
        xlabel = "Time (sec)",
     #   ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(c)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf c["every axis title/.style"] = "below right,at={(0,1)}";

ind=5
d = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(d)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf d["every axis title/.style"] = "below right,at={(0,1)}";

ind=6
e = @pgf Axis(
    {
        xlabel = "Time (sec)",
     #   ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(e)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf e["every axis title/.style"] = "below right,at={(0,1)}";

ind=7
f = @pgf Axis(
    {
        xlabel = "Time (sec)",
     #   ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "6cm",
        width = "6cm",
        xmin = 0,
        xmax = 1,
        ymax = 5,
        ymin = -5,
        ylabel_shift="-10pt",
        title="(f)"
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * time_p[2*(ind-1)+1, :]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[ind][1, 1:1001])),
)
@pgf f["every axis title/.style"] = "below right,at={(0,1)}";

gp2=@pgf GroupPlot(
    { group_style = { group_size="3 by 3" },
      no_markers,
      legend_pos="north west",
      xlabel=raw"$x$",
    },
    a, b, c, d,e,f)

gp2

bdp = @pgf Axis(
    {
        xlabel = "Wind speed (m/sec)",
        ylabel = "Heave amplitude (cm)",
        legend_pos = "north west",
        height = "7cm",
        width = "11cm",
        ymin = 0,
        #ymax = 9e-2,
        mark_options = {scale = 1.5},
        title="(a)"
    },
    Plot({color = "blue", only_marks}, Coordinates(Vel, h * sens)),
    Plot({color = "red", only_marks, mark = "triangle*"}, Coordinates(Vel2, h2 * sens)),
    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(reverse(bd.v), bd.v), vcat(reverse(bd.u * sens), bd.s * sens)),
    ),
)
@pgf bdp["every axis title/.style"] = "below right,at={(0,1)}";

pgfsave("./Figure/physical/bd_exp.pdf", bdp)
pgfsave("./Figure/physical/pp_exp.pdf", gp1)
pgfsave("./Figure/physical/ts_exp.pdf", gp2)

## Validation
using Plots

vars = matread("./src/measured_data/CBC_unstable_v17_3.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series2 = [[transpose(uu1); transpose(uu2)]]

Vel = [14.9, 15.6,  16.5,17.3]
Ap2 = ML_Hopf.lt_pp_n2(
    θ_n,
    θ_,
    ann_l,
    scale_f,
    scale_f2,
    scale_f_l,
    ann,
    Vel,
    Vel2,
    θ_l,
    coθ,
    siθ,
)

sens=18/100
a = @pgf Axis(
    {
        xlabel = L"$h$ (m)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "6cm",
        width = "8cm",
        xmin = -2e-2,
        xmax = 2e-2,
        ymax = 3e-2,
        ymin = -3e-2,
        ylabel_shift="-10pt"
    },
    Plot({color = "red", no_marks}, Coordinates(Ap2[8][1, :] * sens, Ap2[8][2, :])),
 #   LegendEntry("Learnt model"),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(uu1 * sens, uu2),
    ),
#    LegendEntry("Measured data"),
)


## surface plot

function U_trans1(x,y) # plotting transformation
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    pn=θ_n
    uu=[x,y]
    z=[p5,p6]*norm(uu)^2/scale_f_l+(T+reshape(ann_l([norm(uu),vel-U₀],θ_[7:end-2]),2,2)/scale_f2)*uu+ann([uu;vel-U₀],pn)/scale_f
    z[1]
end

function U_trans2(x,y) # plotting transformation
    p1,p2,p3,p4,p5,p6=θ_[1:6]
    T=[p1 p3;p2 p4]
    pn=θ_n
    uu=[x,y]
    z=[p5,p6]*norm(uu)^2/scale_f_l+(T+reshape(ann_l([norm(uu),vel-U₀],θ_[7:end-2]),2,2)/scale_f2)*uu+ann([uu;vel-U₀],pn)/scale_f
    z[2]
end

x = range(-3; stop = 3, length = 50)
y = range(-3; stop = 3, length = 50)
t=range(0,2π,length=100)

##Transformation at wind speed 15

vel=15
s_amp=sqrt(s_/2+sqrt(s_^2+4*(vel-U₀))/2)
u_amp=sqrt(s_/2-sqrt(s_^2+4*(vel-U₀))/2)

axis1 = @pgf Axis(

    {xlabel=L"$u_1$ ",
                ylabel =L"$u_2$",
                zlabel=L"$U_1$",
                legend_pos  = "north east",
                height="6cm",
                width="9cm","no marks",
                zlabel_shift="-10pt",
                ylabel_shift="-10pt",
                xlabel_shift="-10pt",
                title="(a)"
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])
)
@pgf axis1["every axis title/.style"] = "below right,at={(0,1)}";


x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis1, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis1, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans1.(x, y'))
)
push!(axis1, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans1.(x1, y1))
)
push!(axis1, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans1.(x2, y2))
)
push!(axis1, b)

## U2 wind speed 15 m/sec
axis2 = @pgf Axis(

    {xlabel=L"$u_1$ ",
    ylabel =L"$u_2$",
    zlabel=L"$U_2$",
    legend_pos  = "north east",
    height="6cm",
    width="9cm","no marks",
    zlabel_shift="-20pt",
    ylabel_shift="-10pt",
    xlabel_shift="-10pt",
    title="(b)"
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)
@pgf axis2["every axis title/.style"] = "below right,at={(0,1)}";


x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis2, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis2, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans2.(x, y'))
)
push!(axis2, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans2.(x1, y1))
)
push!(axis2, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans2.(x2, y2))
)
push!(axis2, b)

##
vel=17.0
s_amp=sqrt(s_/2+sqrt(s_^2+4*(vel-U₀))/2)
u_amp=sqrt(s_/2-sqrt(s_^2+4*(vel-U₀))/2)
x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

axis3 = @pgf Axis(

    {xlabel=L"$u_1$ ",
    ylabel =L"$u_2$",
    zlabel=L"$U_1$",
    legend_pos  = "north east",
    height="6cm",
    width="9cm","no marks",
    zlabel_shift="-10pt",
    ylabel_shift="-10pt",
    xlabel_shift="-10pt",
    title="(c)"
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])

)
@pgf axis3["every axis title/.style"] = "below right,at={(0,1)}";


b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.4*ones(length(x2))+zeros(length(x2)))
)
push!(axis3, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.4*ones(length(x1))+zeros(length(x1)))
)
push!(axis3, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans1.(x, y'))
)
push!(axis3, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans1.(x1, y1))
)
push!(axis3, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans1.(x2, y2))
)
push!(axis3, b)


## U2 17m/sec
axis4 = @pgf Axis(

    {xlabel=L"$u_1$ ",
    ylabel =L"$u_2$",
    zlabel=L"$U_2$",
    legend_pos  = "north east",
    height="6cm",
    width="9cm","no marks",
    zlabel_shift="-20pt",
    ylabel_shift="-10pt",
    xlabel_shift="-10pt",
    title="(d)"
#                colorbar,
#        "colormap/jet"
        },
#        Legend(["Stable LCO", "Unstable LCO"])
)
@pgf axis4["every axis title/.style"] = "below right,at={(0,1)}";


x1=s_amp*cos.(t)
y1=s_amp*sin.(t)
x2=u_amp*cos.(t)
y2=u_amp*sin.(t)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, -0.3*ones(length(x2))+zeros(length(x2)))
)
push!(axis4, b)

b=@pgf Plot3(
    {   color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, -0.3*ones(length(x1))+zeros(length(x1)))
)
push!(axis4, b)

#add stable LCO
a=@pgf Plot3(
    {
        surf,
    },
    Coordinates(x, y, U_trans2.(x, y'))
)
push!(axis4, a)

b=@pgf Plot3(
    { color="red",
    style = {thick},
        no_marks,
    },
    Coordinates(x1, y1, U_trans2.(x1, y1))
)
push!(axis4, b)

b=@pgf Plot3(
    {color="blue",
    style ={dashed,thick},
        no_marks,
    },
    Coordinates(x2, y2, U_trans2.(x2, y2))
)
push!(axis4, b)

gpU=@pgf GroupPlot(
    { group_style = { group_size="2 by 2" },
      no_markers,
      legend_pos="north west",
      xlabel=raw"$x$",
    },
    axis1, axis2, axis3, axis4)

pgfsave("./Figures/exp/gpU.pdf",gpU)