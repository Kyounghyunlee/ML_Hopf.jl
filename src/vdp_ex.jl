# Numercal experiment of Van der Pol oscillator
# Training data sets are generated from the ODE solutions

using DifferentialEquations
using LinearAlgebra
using DiffEqFlux
using LaTeXStrings
using PGFPlotsX
using Statistics

function vdp(du, u, p, t)
    μ = p[1]
    du[1] = u[2]
    return du[2] = 2 * μ * u[2] - u[1]^2 * u[2] - u[1]
end

function get_sol2(sol, ind) # Convert solution of ODE solver to array solution
    lu = length(sol.u)
    u = Vector{Float64}(undef, lu)
    for i in 1:lu
        uv = sol.u[i]
        u[i] = uv[ind]
    end
    return u
end

function zero_measure(u, ind, t) # Measuring the time of zero crossing points from numerical continuation
    l = length(u)
    zero = Array{Float64}(undef, 0)
    T = Array{Float64}(undef, 0)
    low_p = Array{Float64}(undef, 0)
    high_p = Array{Float64}(undef, 0)
    Ti = Array{Int64}(undef, 0)
    for i in 2:(l - 1)
        sign_con2 = u[i][ind + 1] * u[i - 1][ind + 1]
        if sign_con2 < 0
            if (u[i][ind] + u[i - 1][ind]) / 2 < 0
                low_p = vcat(low_p, (u[i][ind] + u[i - 1][ind]) / 2)
            else
                high_p = vcat(high_p, (u[i][ind] + u[i - 1][ind]) / 2)
            end
        end
    end
    h₀ = mean(high_p) + mean(low_p)
    h₀ = h₀ / 2
    for i in 2:(l - 1)
        sign_con = (u[i][ind] - h₀) * (u[i + 1][ind] - h₀)
        if sign_con < 0
            if (u[i][ind + 1] + u[i + 1][ind + 1]) / 2 > 0
                zero = vcat(zero, (u[i][ind] + u[i - 1][ind]) / 2)
                Ti = vcat(Ti, i)
                T = vcat(T, t[i])
            end
        end
    end
    t_l = length(T)
    P = Array{Float64}(undef, t_l - 1)
    for j in 2:t_l
        P[j - 1] = T[j] - T[j - 1]
    end
    return (T=Ti, P=P, hp=high_p, lp=low_p, h₀=h₀)
end

function get_stable_LCO(p, u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st) # Get a stable LCO from numerical integration
    u = u0
    dim = length(u0)
    prob = ODEProblem(eq, u, (0, tl * rp), p)
    sol = DifferentialEquations.solve(prob, Tsit5(); reltol=stol, abstol=stol, saveat=st)
    vP = 1
    P = 0
    T = 0
    while vP > tol
        u = sol.u[end]
        prob = ODEProblem(eq, u, (0, tl), p)
        sol = DifferentialEquations.solve(
            prob, Tsit5(); reltol=stol, abstol=stol, saveat=st
        )
        z = zero_measure(sol.u, 1, sol.t)
        vP = Statistics.var(z.hp)
        P = z.P[1]
        T = z.T
    end
    tl = length(sol)
    uu = Array{Float64}(undef, tl, dim)
    for i in 1:dim
        uu[:, i] = get_sol2(sol, i)
    end
    t = Array{Float64}(undef, length(sol))
    r = Array{Float64}(undef, length(sol))
    u = uu[:, ind1]
    v = uu[:, ind2]
    for i in 1:length(u)
        t[i] = atan(v[i] - v₀, u[i] - u₀)
        r[i] = sqrt((u[i] - u₀)^2 + (v[i] - v₀)^2)
    end
    return (u=uu, t=t, r=r, P=P, T=T)
end

function LS_harmonics(r, t, ω, N) # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
    # Fourier coefficients are computed in least square sence
    c = Array{Float64}(undef, 2 * N + 1)
    M = Array{Float64}(undef, 1, 2 * N + 1)
    tM = Array{Float64}(undef, 0, 2 * N + 1)
    tl = length(t)
    rr = Array{Float64}(undef, tl)
    M[1] = 1
    for j in 1:tl
        for i in 1:N
            M[1 + i] = cos(ω * t[j] * i)
            M[1 + N + i] = sin(ω * t[j] * i)
        end
        tM = vcat(tM, M)
    end
    MM = transpose(tM) * tM
    rN = transpose(tM) * r
    c = inv(MM) * rN
    for j in 1:tl
        rr[j] = c[1]
        for i in 1:N
            rr[j] += c[i + 1] * cos(ω * t[j] * i)
            rr[j] += c[i + 1 + N] * sin(ω * t[j] * i)
        end
    end
    return (coeff=c, rr=rr)
end

function generate_data_vdp(vel_l, Vel, nh)
    #Generate training data
    u0 = Float32[1.0, 0]
    tol = 1e-7
    stol = 1e-8
    eq = vdp
    rp = 5
    ind1 = 1
    ind2 = 2
    AA = zeros(vel_l, Int(nh * 2 + 1))

    p_ = Vel[1]
    g = get_stable_LCO(p_, u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st)
    u₀ = mean(g.u[:, 1])
    v₀ = mean(g.u[:, 2])
    for i in 1:vel_l
        p_ = Vel[i]
        g = get_stable_LCO(p_, u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st)
        r = g.r
        t = g.t
        c = LS_harmonics(r, t, 1, nh).coeff
        AA[i, :] = c
    end
    AA = transpose(AA)
    θ = range(0; stop=2π, length=θ_l)
    coθ = cos.(θ)
    siθ = sin.(θ)
    t_series = [
        Transpose(
            get_stable_LCO(Vel[i], u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st2).u[
                :, [1, 2]
            ],
        ) for i in 1:vel_l
    ]
    θ_series = [
        get_stable_LCO(Vel[i], u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st2).t for
        i in 1:vel_l
    ]
    return (data=AA, ts=t_series, coθ=coθ, siθ=siθ, theta_s=θ_series)
end

function nf_dis_vdp(U₀, Vel)
    del = Vel - U₀ * ones(length(Vel))
    s_amp = sqrt.(del)
    vl = [s_amp[i] * [coθ'; siθ'] for i in 1:length(Vel)]
    return vl
end

function f_coeff(vlT, Vel, u₀, v₀)
    Pr = zeros(2 * nh + 1, 0)
    for k in 1:length(Vel)
        z1 = vlT[k][1, :] - u₀ * ones(θ_l)
        z2 = vlT[k][2, :] - v₀ * ones(θ_l)
        theta = atan.(z2, z1)
        r = sqrt.(z1 .^ 2 + z2 .^ 2)
        tM = Array{Float64}(undef, 0, 2 * nh + 1)
        rr = Array{Float64}(undef, θ_l)
        for j in 1:θ_l
            tM1 = Array{Float64}(undef, 0, nh + 1)
            tM2 = Array{Float64}(undef, 0, nh)
            tM1_ = [cos(theta[j] * i) for i in 1:nh]
            tM2_ = [sin(theta[j] * i) for i in 1:nh]
            tM1_ = vcat(1, tM1_)
            tM1 = vcat(tM1, Transpose(tM1_))
            tM2 = vcat(tM2, Transpose(tM2_))
            tM_ = hcat(tM1, tM2)
            tM = vcat(tM, tM_)
        end
        MM = Transpose(tM) * tM
        rN = Transpose(tM) * r
        c = inv(MM) * rN
        Pr = hcat(Pr, c)
        Pr
    end
    return Pr
end
# Define the experimental parameter

θ_l = 200;
θ = range(0; stop=2π, length=θ_l)
coθ = cos.(θ)
siθ = sin.(θ)
tl = 30.0;
st = 0.01;
st2 = 0.02;
vel_l = 3
Vel = range(0.2; stop=1.0, length=vel_l)
nh = 20

dat = generate_data_vdp(vel_l, Vel, nh)
AA = dat.data
tl2 = 10.0

tt = Int(tl2 / st2 + 1)
t_series = [dat.ts[i][:, 1:tt] for i in 1:length(Vel)]
θ_series = [dat.theta_s[i][1:tt] for i in 1:length(Vel)]

function predict_lt_vdp(θ_t) #predict the linear transformation
    nf = nf_dis_vdp(0.0, Vel)
    vl = nf
    p1, p2, p3, p4 = θ_t
    T = [p1 p3; p2 p4]
    vlT = [T * (vl[i]) for i in 1:length(Vel)]

    Pr = f_coeff(vlT, Vel, 0, 0)
    return Pr
end

function lt_pp_vdp(θ_t) # This function gives phase portrait of the transformed system from the normal form
    nf = nf_dis_vdp(0.0, Vel)
    vl = nf
    p1, p2, p3, p4 = θ_t
    T = [p1 p3; p2 p4]
    vlT = [T * (vl[i]) for i in 1:length(Vel)]
    return vlT
end

function Array_chain(gu, ann, p) # vectorized input-> vectorized neural net
    al = length(gu[1, :])
    AC = zeros(2, 0)
    for i in 1:al
        AC = hcat(AC, ann(gu[:, i], p))
    end
    return AC
end

function loss_lt_vdp(θ_t)
    pred = predict_lt_vdp(θ_t)
    return sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

θ = vec([3.0 0.0; 0.0 3.0])
loss_lt_vdp(θ)

res1 = DiffEqFlux.sciml_train(loss_lt_vdp, θ, ADAM(0.01); maxiters=200)
res1.minimum
θ_ = res1.minimizer

##
function predict_nt_vdp(θ_t)
    p1, p2, p3, p4 = θ_
    np1 = θ_t[1]
    pn = θ_t[2:end]
    nf = nf_dis_vdp(np1, Vel)
    vl = nf
    T = [p1 p3; p2 p4]
    vlT = [
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f_pp for
        i in 1:length(Vel)
    ]

    return Pr = f_coeff(vlT, Vel, 0, 0)
end

function lt_pp_n_vdp(θ_t) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1, p2, p3, p4 = θ_
    np1 = θ_t[1]
    pn = θ_t[2:end]
    nf = nf_dis_vdp(np1, Vel)
    vl = nf
    T = [p1 p3; p2 p4]
    vlT = [
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f_pp for
        i in 1:length(Vel)
    ]
    return vlT
end

function lt_pp_n_vdp(θ_t, Vel) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1, p2, p3, p4 = θ_
    np1 = θ_t[1]
    pn = θ_t[2:end]
    nf = nf_dis_vdp(np1, Vel)
    vl = nf
    T = [p1 p3; p2 p4]
    vlT = [
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f_pp for
        i in 1:length(Vel)
    ]
    return vlT
end

function loss_nt_vdp(θ_t)
    pred = predict_nt_vdp(θ_t)
    return sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

hidden = 31
ann = FastChain(
    FastDense(3, hidden, tanh), FastDense(hidden, hidden, tanh), FastDense(hidden, 2)
)
θn = initial_params(ann)
scale_f_pp = 1e1

pp = [0]
θn = vcat(pp, θn)
loss_nt_vdp(θn)

res1 = DiffEqFlux.sciml_train(loss_nt_vdp, θn, ADAM(0.01); maxiters=300)
res1 = DiffEqFlux.sciml_train(
    loss_nt_vdp, res1.minimizer, BFGS(; initial_stepnorm=1e-5); maxiters=10000
)
θ_2 = res1.minimizer

Ap = lt_pp_n_vdp(θ_2)

a = @pgf Axis(
    {
        xlabel = L"$z_1$",
        ylabel = L"$z_2$",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3,
        xmax = 3,
        ymin = -5.5,
        ymax = 5.5,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[1][1, :], Ap[1][2, :])),
    LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(t_series[1][1, :], t_series[1][2, :])),
    LegendEntry("Ground truth"),
)
#pgfsave("./Figures/num_vdp/vdp_1.pdf",a)

a = @pgf Axis(
    {
        xlabel = L"$z_1$",
        ylabel = L"$z_2$",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = -3,
        xmax = 3,
        ymin = -5.5,
        ymax = 5.5,
    },
    Plot({color = "red", no_marks}, Coordinates(Ap[3][1, :], Ap[3][2, :])),
    LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(t_series[3][1, :], t_series[3][2, :])),
    LegendEntry("Ground truth"),
)

#pgfsave("./Figures/num_vdp/vdp_2.pdf",a)

function Inv_T_u_vdp(th0, vel, tol) # This function gives initial conditions of the model
    p1, p2, p3, p4, = θ_[1:4]
    T = [p1 p3; p2 p4]
    np1 = θ_2[1]
    pn = θ_2[2:end]
    del = vel - np1
    s_amp = sqrt.(del)
    theta = range(-π; stop=π, length=400)

    uu = [[s_amp * cos(theta[i]), s_amp * sin(theta[i])] for i in 1:length(theta)]
    u = [T * uu[i] + ann([uu[i]; vel - np1], pn) / scale_f_pp for i in 1:length(theta)]
    t0 = [abs(atan(u[i][2], u[i][1]) - th0) for i in 1:length(theta)]
    er = minimum(t0)
    while er > tol
        #    global theta,t0
        theta = range(theta[argmin(t0) - 1], theta[argmin(t0) + 1]; length=400)
        uu = [[s_amp * cos(theta[i]), s_amp * sin(theta[i])] for i in 1:length(theta)]
        u = [T * uu[i] + ann([uu[i]; vel - np1], pn) / scale_f_pp for i in 1:length(theta)]
        t0 = [abs(atan(u[i][2], u[i][1]) - th0) for i in 1:length(theta)]
        er = minimum(t0)
    end
    return (th=theta[argmin(t0)], uu=u[argmin(t0)])
end

function dudt_ph2(u, p, t)
    θ = u[1]
    r = u[2]
    c = u[3]
    np1 = θ_2[1]
    δ₀ = np1
    ν = (c - δ₀)
    ω₀ = p[1]
    uu = [r * cos(θ), r * sin(θ), ν]
    du₁ = ω₀ + ann3(uu, p[2:end])[1] / om_scale
    du₂ = 0
    du₃ = 0
    return [du₁, du₂, du₃]
end

function atan2(u)
    θ = atan.(u[2, :], u[1, :])
    for i in 2:length(θ)
        if θ[i] - θ[i - 1] > π
            θ = θ + vcat(zeros(i - 1), -2 * π * ones(length(θ) - i + 1))
        elseif θ[i] - θ[i - 1] < -π
            θ = θ + vcat(zeros(i - 1), 2 * π * ones(length(θ) - i + 1))
        end
    end
    return θ
end

function dudt_nf_vdp(u, p, t)
    c = u[3]
    np1 = θ_2[1]
    a2 = -1
    δ₀ = np1
    ν = (c - δ₀)
    r2 = u[1]^2 + u[2]^2
    r = sqrt(r2)
    θ = atan(u[2], u[1])
    ω₀ = p[1]
    ω₁ = p[2]
    uu = [r * cos(θ), r * sin(θ), ν]
    hc = ann3(ν, p[3:end])
    phs_s = [hc[nh2 + i + 1] * sin(i * θ) for i in 1:nh2]
    phs_s = sum(phs_s)
    phs_c = [hc[i + 1] * cos(i * θ) for i in 1:nh2]
    phs_c = sum(phs_c)
    ph = ω₀ + ν * ω₁ + ann3(uu, p[3:end])[1] / scale_f
    du₁ = ν * u[1] - u[2] * ph + a2 * u[1] * r2
    du₂ = u[1] * ph + ν * u[2] + a2 * u[2] * r2
    du₃ = 0
    return [du₁, du₂, du₃]
end

function predict_time_T_vdp(p) #,uu_t0
    np1 = θ_2[1]
    pn = θ_2[2:end]
    p1, p2, p3, p4 = θ_[1:4]
    T = [p1 p3; p2 p4]
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_nf_vdp, u_t0[i], (0, tl2), p),
                Tsit5(),
                u_t0[i],
                p;
                saveat=st3,
                abstol=1e-8,
                reltol=1e-8,
                sensealg=InterpolatingAdjoint(; autojacvec=ReverseDiffVJP()),
            ),
        ) for i in 1:length(Vel)
    ]
    uu = [transpose(hcat(A1[i][1, :], A1[i][2, :], A1[i][3, :])) for i in 1:vl]
    delU = zeros(2, spl2)
    delU2 = -np1 * ones(1, spl2)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i in 1:vl]
    vlT = [T * uu[i][1:2, :] + Array_chain(uu[i], ann, pn) / scale_f_pp for i in 1:vl]
    Pr = zeros(0, spl2)
    for i in 1:vl
        theta = transpose(atan2(vlT[i][[1, 2], :]))
        Pr = vcat(Pr, theta)
    end
    return Pr
end

st3 = 0.05

t_s3 = [
    Array(
        concrete_solve(
            ODEProblem(vdp, t_series[i][:, 1], (0, tl2), Vel[i]),
            Tsit5(),
            t_series[i][:, 1],
            Vel[i];
            saveat=st3,
        ),
    ) for i in 1:length(Vel)
]
t_s2 = [atan2(t_s3[i]) for i in 1:length(Vel)]

vl = length(Vel)
spl2 = length(t_s2[1])
t_s = zeros(vl, spl2)

tol = 1e-8
np1 = θ_2[1]
pn = θ_2[2:end]
s_amp = [sqrt(Vel[i] - np1) for i in 1:length(Vel)]
theta0 = [t_s2[i][1] for i in 1:length(Vel)]
θ₀ = [Inv_T_u_vdp(theta0[i], Vel[i], tol).th for i in 1:length(Vel)]
u_t0 = [[s_amp[i] * cos.(θ₀[i]), s_amp[i] * sin.(θ₀[i]), Vel[i]] for i in 1:length(Vel)]

for i in 1:vl
    t_s[i, :] = t_s2[i]
end
A3 = t_s
spl2 = length(A3[1, :])

function loss_time_T_vdp(p)
    pred = predict_time_T_vdp(p)
    return sum(abs2, A3 .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

hidden = 31
nh2 = 10
ann3 = FastChain(FastDense(3, hidden, tanh), FastDense(hidden, nh2 * 2 + 1, tanh))
np = initial_params(ann3)
aa = [-1, 0.16]
p = vcat(aa, np)
scale_f = 1
##harmonic representation of phase
res1 = DiffEqFlux.sciml_train(loss_time_T_vdp, p, ADAM(0.01); maxiters=200)
res1 = DiffEqFlux.sciml_train(
    loss_time_T_vdp, p, NADAM(0.002, (0.89, 0.995)); maxiters=4000
)
res1 = DiffEqFlux.sciml_train(
    loss_time_T_vdp, res1.minimizer, BFGS(; initial_stepnorm=1e-5); maxiters=30000
)

res1.minimum
p = res1.minimizer

tv = range(0, tl2; length=length(A3[1, :]))

ind = 1
a = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$z_1$",
        legend_pos = "north west",
        height = "9cm",
        width = "9cm",
        xmin = 0,
        xmax = 10,
        ymax = 4,
        ymin = -4,
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, predict_time_T_vdp2(p)[2 * (ind - 1) + 1, :]),
    ),
    LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(tv, t_s3[ind][1, :])),
    LegendEntry("Ground truth"),
)
#pgfsave("./Figures/num_vdp/vdp_t_020.pdf",a)

ind = 3
a = @pgf Axis(
    {
        xlabel = "Time (sec)",
        ylabel = L"$z_1$",
        legend_pos = "south east",
        height = "9cm",
        width = "9cm",
        xmin = 0,
        xmax = 10,
        ymax = 4,
        ymin = -4,
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, predict_time_T_vdp2(p)[2 * (ind - 1) + 1, :]),
    ),
    LegendEntry("Learnt model"),
    Plot({color = "blue", no_marks}, Coordinates(tv, t_s3[ind][1, :])),
    LegendEntry("Ground truth"),
)

#pgfsave("./Figures/num_vdp/vdp_t_100.pdf",a)
