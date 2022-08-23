module ML_Hopf

# Add core packages to the enviroment
using DifferentialEquations
using LinearAlgebra
#using DiffEqFlux
using LaTeXStrings
using MAT
using NLsolve
using Statistics
using StaticArrays
using ForwardDiff
using Statistics

export generate_data, Array_chain

function flutter_eq_CBC(du, u, p, t) #Flutter equation of motion
    μ, Kp, Kd, Fa, Fω = p[1:5]
    c = p[6:end]
    ind1 = 1
    ind2 = 3
    theta = atan(u[ind1], u[ind2]) #theta is computed from two state variables
    r = c[1]
    h = c[2:end]
    nh = Int(length(h) / 2)
    for i in 1:nh
        r += h[i] * cos(i * theta) + h[i + nh] * sin(theta * i)
    end
    b = 0.15
    a = -0.5
    rho = 1.204
    x__alpha = 0.2340
    c__0 = 1
    c__1 = 0.1650
    c__2 = 0.0455
    c__3 = 0.335
    c__4 = 0.3
    c__alpha = 0.562766779889303
    c__h = 15.4430
    I__alpha = 0.1726
    k__alpha = 54.116182926744390
    k__h = 3.5294e+03
    m = 5.3
    m__T = 16.9
    U = μ

    MM = [
        b^2 * pi * rho+m__T -a * b^3 * pi * rho+b * m * x__alpha 0
        -a * b^3 * pi * rho+b * m * x__alpha I__alpha+pi * (0.1e1 / 0.8e1 + a^2) * rho * b^4 0
        0 0 1
    ]
    DD = [
        c__h+2 * pi * rho * b * U * (c__0 - c__1 - c__3) (1+(c__0 - c__1 - c__3) * (1 - 2 * a))*pi*rho*b^2*U 2*pi*rho*U^2*b*(c__1 * c__2+c__3 * c__4)
        -0.2e1*pi*(a+0.1e1 / 0.2e1)*rho*(b^2)*(c__0 - c__1-c__3)*U c__alpha+(0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b^3) * U -0.2e1*pi*rho*(U^2)*(b^2)*(a+0.1e1 / 0.2e1)*(c__1 * c__2+c__3 * c__4)
        -1/b a-0.1e1 / 0.2e1 (c__2 + c__4) * U/b
    ]
    KK = [
        k__h 2*pi*rho*b*U^2*(c__0 - c__1-c__3) 2*pi*rho*U^3*c__2*c__4*(c__1+c__3)
        0 k__alpha-0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b^2) * (U^2) -0.2e1*pi*rho*b*(U^3)*(a+0.1e1 / 0.2e1)*c__2*c__4*(c__1+c__3)
        0 -U/b c__2 * c__4 * U^2/b^2
    ]

    K1 = -inv(MM) * KK
    D1 = -inv(MM) * DD

    J1 = [0 1 0 0 0 0]
    J2 = [K1[1, 1] D1[1, 1] K1[1, 2] D1[1, 2] K1[1, 3] D1[1, 3]]
    J3 = [0 0 0 1 0 0]
    J4 = [K1[2, 1] D1[2, 1] K1[2, 2] D1[2, 2] K1[2, 3] D1[2, 3]]
    J5 = [0 0 0 0 0 1]
    J6 = [K1[3, 1] D1[3, 1] K1[3, 2] D1[3, 2] K1[3, 3] D1[3, 3]]

    J = [J1; J2; J3; J4; J5; J6]
    du_ = J * u
    #Control added on heave
    M2 = inv(MM)
    control = Kp * (r * cos(theta) - u[ind1]) + Kd * (r * sin(theta) - u[ind2])
    ka2 = 751.6
    ka3 = 5006
    nonlinear_h = -M2[1, 2] * (ka2 * u[3]^2 + ka3 * u[3]^3)
    nonlinear_theta = -M2[2, 2] * (ka2 * u[3]^2 + ka3 * u[3]^3)

    du[1] = du_[1]
    du[2] = du_[2] + nonlinear_h + M2[1, 1] * control + M2[1, 1] * Fa * sin(Fω * t)
    du[3] = du_[3]
    du[4] = du_[4] + nonlinear_theta + M2[2, 1] * control + M2[2, 1] * Fa * sin(Fω * t)
    du[5] = du_[5]
    return du[6] = du_[6]
end

function generate_data(vel_l, Vel, nh) # Training data
    #Generate training data
    u0 = Float32[2e-1, 0, 2e-1, 0, 0, 0]
    tol = 1e-7
    stol = 1e-8
    eq = flutter_eq_CBC
    rp = 5
    ind1 = 1
    ind2 = 3
    pp = [zeros(10) for i in 1:vel_l]
    AA = zeros(vel_l, Int(nh * 2 + 1))

    p_ = zeros(9)
    p_ = vcat(Vel[1], p_)
    tl = 10.0
    g = get_stable_LCO(p_, u0, tl, tol, eq, stol, rp, ind1, ind2, 0.0, 0.0, st)

    u₀ = mean(g.u[:, 1])
    v₀ = mean(g.u[:, 3])
    for i in 1:vel_l
        pp[i][1] = Vel[i]
        g = get_stable_LCO(pp[i], u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st)
        r = g.r
        t = g.t
        c = LS_harmonics(r, t, 1, nh).coeff
        AA[i, :] = c
    end
    cc = [
        LS_harmonics(
            get_sol(U[s_ind[i]], N, 1, 3).r, get_sol(U[s_ind[i]], N, 1, 3).t, 1, nh
        ).coeff for i in 1:length(s_ind)
    ]
    cc = hcat(cc)

    AA = transpose(AA)
    Al = AA
    AA = hcat(AA, cc)
    t_series = [
        Transpose(
            get_stable_LCO([Vel[i] transpose(zeros(9))], u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st).u[
                :, [1, 3]
            ],
        ) for i in 1:vel_l
    ]
    θ_series = [
        get_stable_LCO(
            [Vel[i] transpose(zeros(9))], u0, tl, tol, eq, stol, rp, ind1, ind2, u₀, v₀, st
        ).t for i in 1:vel_l
    ]
    θ = range(0; stop=2π, length=θ_l)
    coθ = cos.(θ)
    siθ = sin.(θ)
    return (data=AA, ts=t_series, d0=[u₀, v₀], data2=Al, coθ=coθ, siθ=siθ, theta_s=θ_series)
end

## Numerical model

function nf_dis(U₀, s, Vel, Vel2, coθ, siθ)
    del = Vel - U₀ * ones(length(Vel))
    del2 = Vel2 - U₀ * ones(length(Vel2))
    va2 = s * ones(length(Vel))
    va2_2 = s * ones(length(Vel2))
    s_amp = sqrt.(va2 / 2 + sqrt.(va2 .^ 2 + 4 * del) / 2)
    u_amp = sqrt.(va2_2 / 2 - sqrt.(va2_2 .^ 2 + 4 * del2) / 2)

    vl = [s_amp[i] * [coθ'; siθ'] for i in 1:length(Vel)]
    vl2 = [u_amp[i] * [coθ'; siθ'] for i in 1:length(Vel2)]
    return (v=vl, v2=vl2)
end

function f_coeff(vlT, Vel, u₀, v₀, nh, θ_l)
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

function predict_lt(θ_t, U₀, s, Vel, Vel2, uu0, scale_f_l, nh, θ_l, coθ, siθ) #predict the linear transformation
    nf = nf_dis(U₀, s, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_t[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    T = [p1 p3; p2 p4]
    vlT = [dis * norm(vl[i][:, 1])^2 + T * (vl[i]) for i in 1:length(Vel)]
    vlT2 = [dis * norm(vl2[i][:, 1])^2 + T * (vl2[i]) for i in 1:length(Vel2)]
    u₀ = uu0[1]
    v₀ = uu0[2]
    Pr = f_coeff(vlT, Vel, u₀, v₀, nh, θ_l)
    Pr2 = f_coeff(vlT2, Vel2, 0, 0, nh, θ_l)
    return hcat(Pr, Pr2)
end

function lt_pp(θ_t, U₀, s, Vel, Vel2, scale_f_l, θ_l, coθ, siθ) # This function gives phase portrait of the transformed system from the normal form
    nf = nf_dis(U₀, s, Vel, Vel2, coθ, siθ)
    vl = nf.v
    p1, p2, p3, p4, p5, p6 = θ_t[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    vlT = [dis * norm(vl[i][:, 1])^2 + T * (vl[i]) for i in 1:length(Vel)]
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

function loss_lt(θ_t, U₀, s, Vel, Vel2, uu0, AA, scale_f_l, nh, θ_l, coθ, siθ)
    pred = predict_lt(θ_t, U₀, s, Vel, Vel2, uu0, scale_f_l, nh, θ_l, coθ, siθ)
    return sum(abs2, AA .- pred) # + 1e-5*sum(sum.(abs, params(ann)))
end

function predict_nt(θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, nh, uu0, coθ, siθ)
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    np1, np2 = θ_t[1:2]

    pn = θ_t[3:end]
    T = [p1 p3; p2 p4]

    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2

    vlT = [
        dis * norm(vl[i][:, 1])^2 +
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel)
    ]
    vlT2 = [
        dis * norm(vl2[i][:, 1])^2 +
        T * (vl2[i]) +
        Array_chain([vl2[i]; (Vel2[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel2)
    ]

    u₀ = uu0[1]
    v₀ = uu0[2]

    Pr = f_coeff(vlT, Vel, u₀, v₀, nh, θ_l)
    Pr2 = f_coeff(vlT2, Vel2, 0, 0, nh, θ_l)
    return hcat(Pr, Pr2)
end

function lt_pp_n(θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    np1, np2 = θ_t[1:2]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v

    pn = θ_t[3:end]
    vlT = [
        dis * norm(vl[i][:, 1])^2 +
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel)
    ]
    return vlT
end

function lt_pp_n_u(θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, coθ, siθ) # This function gives phase portrait of the transformed system from the normal form (unstable LCO)
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    np1, np2 = θ_t[1:2]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl2 = nf.v2
    pn = θ_t[3:end]

    vlT = [
        dis * norm(vl2[i][:, 1])^2 +
        T * (vl2[i]) +
        Array_chain([vl2[i]; (Vel2[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel2)
    ]
    return vlT
end

function loss_nt(θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, AA, ann, nh, uu0, coθ, siθ)
    pred = predict_nt(θ_t, θ_, θ_l, scale_f_l, Vel, Vel2, scale_f, ann, nh, uu0, coθ, siθ)
    return sum(abs2, AA .- pred)
end

function lt_b_dia(θ_t, ind, vel_l, ann, θ_l, coθ, siθ, scale_f_l, scale_f)
    p1, p2, p3, p4, p5, p6 = θ_t[1:6]
    T = [p1 p3; p2 p4]
    np1, np2 = θ_t[7:8]
    Vel = range(np1 - np2^2 / 4 + 1e-7; stop=np1, length=vel_l)
    nf = nf_dis(np1, np2, Vel, Vel, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    pn = θ_t[9:end]

    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    vlT = [
        dis * norm(vl[i][:, 1])^2 +
        T * (vl[i]) +
        Array_chain([vl[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel)
    ]
    vlT2 = [
        dis * norm(vl[i][:, 1])^2 +
        T * (vl2[i]) +
        Array_chain([vl2[i]; (Vel[i] - np1) * ones(1, θ_l)], ann, pn) / scale_f for
        i in 1:length(Vel)
    ]
    vlTas = [maximum(vlT[i][ind, :]) - minimum(vlT[i][ind, :]) for i in 1:length(Vel)]
    vlTau = [maximum(vlT2[i][ind, :]) - minimum(vlT2[i][ind, :]) for i in 1:length(Vel)]
    return (s=vlTas, u=vlTau, v=Vel)
end

## Experimental model

function LT(vl, T, np1, dis, Vel, pn1, ann_l, scale_f2)
    TT =
        dis * norm(vl[:, 1])^2 +
        (T + reshape(ann_l([norm(vl[:, 1]), Vel - np1], pn1), 2, 2) / scale_f2) * (vl)
    return TT
end

function predict_lt_nn(θ_t, ann_l, scale_f_l, scale_f2, θ_l, nh, Vel, Vel2, coθ, siθ)
    np1 = θ_t[end - 1]
    np2 = θ_t[end]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_t[1:6]
    T = [p1 p3; p2 p4]

    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    pn1 = θ_t[7:(end - 2)]
    vlT = [LT(vl[i], T, np1, dis, Vel[i], pn1, ann_l, scale_f2) for i in 1:length(Vel)]
    vlT2 = [LT(vl2[i], T, np1, dis, Vel2[i], pn1, ann_l, scale_f2) for i in 1:length(Vel2)]

    Pr = f_coeff(vlT, Vel, 0, 0, nh, θ_l)
    Pr2 = f_coeff(vlT2, Vel2, 0, 0, nh, θ_l)
    PP = hcat(Pr, Pr2)
    return PP
end

function predict_lt_nn2(
    θ_t, ann_l, scale_f_l, scale_f2, U₀, s_, θ_l, nh, Vel, Vel2, coθ, siθ
) #predict the linear transformation
    np1 = U₀
    np2 = s_
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_t[1:6]
    T = [p1 p3; p2 p4]

    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    pn1 = θ_t[7:(end - 2)]
    vlT = [LT(vl[i], T, np1, dis, Vel[i], pn1, ann_l, scale_f2) for i in 1:length(Vel)]
    vlT2 = [LT(vl2[i], T, np1, dis, Vel2[i], pn1, ann_l, scale_f2) for i in 1:length(Vel2)]

    Pr = f_coeff(vlT, Vel, 0, 0, nh, θ_l)
    Pr2 = f_coeff(vlT2, Vel2, 0, 0, nh, θ_l)
    PP = hcat(Pr, Pr2)
    return PP
end

function loss_lt_nn(θ_t, ann_l, scale_f_l, scale_f2, AA, θ_l, nh, Vel, Vel2, coθ, siθ)
    pred = predict_lt_nn(θ_t, ann_l, scale_f_l, scale_f2, θ_l, nh, Vel, Vel2, coθ, siθ)
    return sum(abs2, AA .- pred)
end

function loss_lt_nn2(
    θ_t, ann_l, scale_f_l, scale_f2, U₀, s_, AA, θ_l, nh, Vel, Vel2, coθ, siθ
)
    pred = predict_lt_nn2(
        θ_t, ann_l, scale_f_l, scale_f2, U₀, s_, θ_l, nh, Vel, Vel2, coθ, siθ
    )
    return sum(abs2, AA .- pred)
end

function Nt(vl, T, Vel, dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
    vlT =
        LT(vl, T, np1, dis, Vel, pn1, ann_l, scale_f2) +
        Array_chain([vl; (Vel - np1) * ones(1, length(vl[1, :]))], ann, pn2) / scale_f
    return vlT
end

function predict_nt2(
    θ_t, θ_, ann_l, scale_f, scale_f2, scale_f_l, ann, nh, Vel, Vel2, θ_l, coθ, siθ
)
    np1 = θ_t[end - 1]
    np2 = θ_t[end]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    pn1 = θ_[7:(end - 2)]
    pn2 = θ_t[1:(end - 2)]

    vlT = [
        Nt(vl[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel)
    ]
    vlT2 = [
        Nt(vl2[i], T, Vel2[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel2)
    ]

    Pr = f_coeff(vlT, Vel, 0, 0, nh, θ_l)
    Pr2 = f_coeff(vlT2, Vel2, 0, 0, nh, θ_l)
    return hcat(Pr, Pr2)
end

function predict_nt3(θ_t, θ_, ann_l, scale_f, scale_f2, scale_f_l, ann, nh, Vel, Vel2, θ_l, coθ, siθ)
    np1 = θ_t[end - 1]
    np2 = θ_t[end]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    pn1 = θ_[7:(end - 2)]
    pn2 = θ_t[1:(end - 2)]

    if Vel2==[]
        vlT = [Nt(vl[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for i in 1:length(Vel)]
    elseif Vel==[]
        vlT = [Nt(vl2[i], T, Vel2[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for i in 1:length(Vel2)]
    end
    Pr = f_coeff(vlT, Vel, 0, 0, nh, θ_l)
    Pr2 = f_coeff(vlT, Vel2, 0, 0, nh, θ_l)
    return hcat(Pr, Pr2)
end

function lt_pp_n2(
    θ_t, θ_, ann_l, scale_f, scale_f2, scale_f_l, ann, Vel, Vel2, θ_l, coθ, siθ
) # This function gives phase portrait of the transformed system from the normal form (stable LCO)
    np1 = θ_t[end - 1]
    np2 = θ_t[end]
    nf = nf_dis(np1, np2, Vel, Vel2, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    dis = transpose([p5 * ones(θ_l) p6 * ones(θ_l)]) / scale_f_l
    pn1 = θ_[7:(end - 2)]
    pn2 = θ_t[1:(end - 1)]

    vlT = [
        Nt(vl[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel)
    ]
    vlT2 = [
        Nt(vl2[i], T, Vel2[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel2)
    ]

    return vcat(vlT, vlT2)
end

function loss_nt2(
    θ_t, θ_, ann_l, scale_f, scale_f2, scale_f_l, ann, AA, nh, Vel, Vel2, θ_l, coθ, siθ
) # Loss function
    pred = predict_nt2(
        θ_t, θ_, ann_l, scale_f, scale_f2, scale_f_l, ann, nh, Vel, Vel2, θ_l, coθ, siθ
    )
    return sum(abs2, AA .- pred)
end

function lt_b_dia2(ind, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ)
    vel_l = 300
    np1 = θ_n[end - 1]
    np2 = θ_n[end]
    Vel = range(np1 - np2^2 / 4 + 1e-7; stop=np1, length=vel_l)
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    pn1 = θ_[7:(end - 2)]
    pn2 = θ_n[1:(end - 2)]

    nf = nf_dis(np1, np2, Vel, Vel, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    dis = transpose([p5 * ones(length(vl)) p6 * ones(length(vl))]) / scale_f_l

    vlT = [
        Nt(vl[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel)
    ]
    vlT2 = [
        Nt(vl2[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel)
    ]

    vlTas = [maximum(vlT[i][ind, :]) - minimum(vlT[i][ind, :]) for i in 1:length(Vel)]
    vlTau = [maximum(vlT2[i][ind, :]) - minimum(vlT2[i][ind, :]) for i in 1:length(Vel)]
    return (s=vlTas, u=vlTau, v=Vel)
end

function lt_b_dia3(ind, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ,add)
    vel_l = 300
    np1 = θ_n[end - 1]
    np2 = θ_n[end]
    Vel = range(np1 - np2^2 / 4 + 1e-7; stop=np1, length=vel_l)
    Vel2 = range(np1 - np2^2 / 4 + 1e-7; stop=np1+add, length=vel_l)
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    pn1 = θ_[7:(end - 2)]
    pn2 = θ_n[1:(end - 2)]

    nf = nf_dis(np1, np2, Vel2, Vel, coθ, siθ)
    vl = nf.v
    vl2 = nf.v2
    dis = transpose([p5 * ones(length(vl)) p6 * ones(length(vl))]) / scale_f_l

    vlT = [
        Nt(vl[i], T, Vel2[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel2)
    ]
    vlT2 = [
        Nt(vl2[i], T, Vel[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann) for
        i in 1:length(Vel)
    ]

    vlTas = [maximum(vlT[i][ind, :]) - minimum(vlT[i][ind, :]) for i in 1:length(Vel2)]
    vlTau = [maximum(vlT2[i][ind, :]) - minimum(vlT2[i][ind, :]) for i in 1:length(Vel)]
    return (s=vlTas, u=vlTau, v=Vel,v2=Vel2)
end


function vdp(du, u, p, t)
    μ = p[1]
    du[1] = u[2]
    return du[2] = 2 * μ * u[2] - u[1]^2 * u[2] - u[1]
end

function generate_data_vdp(vel_l, Vel, nh,tl,st,st2,θ_l )
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


include("Numerical_Cont_Hopf_CBC.jl")

end
