using ML_Hopf
using LinearAlgebra
using Statistics
using MAT
using DiffEqFlux
using PGFPlotsX
using LaTeXStrings
using DifferentialEquations
using Serialization


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
θ = range(0, stop=2π, length=θ_l)
coθ = cos.(θ)
siθ = sin.(θ)
## Linear transformation
scale_f_l = 1e1 # optimization works for scale_f_l>=50 for small scale_f_l optimization does not work.

hidden = 12
ann_l = FastChain(FastDense(2, hidden, tanh),FastDense(hidden, hidden, tanh),FastDense(hidden, 4),)
scale_f2 = 1e2

## Nonlinear transformation
hidden = 11
ann = FastChain(FastDense(3, hidden, tanh),FastDense(hidden, hidden, tanh),FastDense(hidden, 2),)
scale_f = 1e3

hidden = 21
ann3 = FastChain(FastDense(3, hidden, tanh), FastDense(hidden, 1, tanh))
om_scale = 0.3

sens = 18


##
function dudt_ph(u, p, t) # speed of phase
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

function Inv_T_u(th0, vel)
    s_amp = sqrt(np2 / 2 + sqrt(np2^2 + 4 * (vel - np1)) / 2)
    ttl = Int(1e5)
    theta = range(0, 2π, length=ttl)
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
    theta = range(0, 2π, length=ttl)
    uu = [transpose(u_amp * cos.(theta)); transpose(u_amp * sin.(theta))]
    dis = transpose([p5 * ones(ttl) p6 * ones(ttl)]) / scale_f_l
    Tu = ML_Hopf.Nt(uu, T, vel, dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
    t0 = [abs(atan(Tu[2, i], Tu[1, i]) - th0) for i = 1:length(theta)]
    er = minimum(t0)
    return theta[argmin(t0)]
end

function pred_t(Vel3,b,θ_n,θ_,p,uu1,uu2)
    np1 = θ_n[end-1];
    np2 = θ_n[end];
    p1, p2, p3, p4, p5, p6 = θ_[1:6]
    T = [p1 p3; p2 p4]
    pn1 = θ_[7:end-2]
    pn2 = θ_n[1:end-2]
    spl=1001
    s_amp = [sqrt(np2 / 2 + b* sqrt(np2^2 + 4 * (Vel3[i] - np1)) / 2) for i = 1:length(Vel3)]
    t_series3 = transpose(hcat(uu1[1:5:end][1:spl], uu2[1:5:end][1:spl]))
    
    #t_series3=t_series[1][:,1:5:end][:,1:spl]
    
    theta02 = [atan(t_series3[2, 1], t_series3[1, 1]) for i = 1:length(Vel3)]
    if b==1
        θ₀2 = [Inv_T_u(theta02[i], Vel3[i]) for i = 1:length(Vel3)]
    elseif b==-1
        θ₀2 = [Inv_T_uu(theta02[i], Vel3[i]) for i = 1:length(Vel3)]
    end
    
    u_t02 = [[θ₀2[i], s_amp[i], Vel3[i]] for i = 1:length(Vel3)]
    
    tl2 = 1.0
    st = 1e-3
    dis = transpose([p5 * ones(spl) p6 * ones(spl)]) / scale_f_l    
    A1 = [
        Array(
            concrete_solve(
                ODEProblem(dudt_ph, u_t02[i], (0, tl2), p),
                Tsit5(),
                u_t02[i],
                p,
                saveat=st,
                abstol=1e-8,
                reltol=1e-8,
                sensealg=InterpolatingAdjoint(autojacvec=ReverseDiffVJP()),
            ),
        ) for i = 1:length(Vel3)
    ]
    uu = [
        transpose(
            hcat(
                A1[i][2, :] .* cos.(A1[i][1, :]),
                A1[i][2, :] .* sin.(A1[i][1, :]),
                A1[i][3, :],
            ),
        ) for i = 1:length(Vel3)
    ]
    delU = zeros(2, spl)
    delU2 = -np1 * ones(1, spl)
    delU = vcat(delU, delU2)
    uu = [uu[i] + delU for i = 1:length(Vel3)]
    vl = [uu[i][1:2, :] for i = 1:length(Vel3)]
    vlT = [
        ML_Hopf.Nt(vl[i], T, Vel3[i], dis, pn1, np1, pn2, ann_l, scale_f, scale_f2, ann)
        for i = 1:length(Vel3)
    ]
    return vlT
end

##
tl2=1.0
spl=1001
tv = range(0, tl2, length=spl)

newdata = deserialize("exp_flutter_pp_val1.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;

bd1 = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ, 0.0)
h = [maximum(t_series[i][1, :]) - minimum(t_series[i][1, :]) for i = 1:length(Vel)]
h2 = [maximum(t_series[i+4][1, :]) - minimum(t_series[i+4][1, :]) for i = 1:length(Vel2)]

Vel3 = [14.9]
Vel0=[]
Ap1 = ML_Hopf.lt_pp_n2(θ_n,θ_,ann_l,scale_f,scale_f2,scale_f_l,ann,Vel3,Vel0,θ_l,coθ,siθ,)

vars = matread("./src/measured_data/CBC_stable_v14_9.mat")

uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series1 = transpose(hcat(uu1[1:5:end][1:spl], uu2[1:5:end][1:spl]))

np1 = θ_n[end-1];
np2 = θ_n[end];
p1, p2, p3, p4, p5, p6 = θ_[1:6]
T = [p1 p3; p2 p4]
pn1 = θ_[7:end-2]
pn2 = θ_n[1:end-2]

vlT1=pred_t(Vel3,1,θ_n,θ_,p,uu1,uu2)

newdata = deserialize("exp_flutter_pp_val2.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;

bd2 = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ, 0.0)
Vel3 = [15.6]
Vel0=[]
Ap2 = ML_Hopf.lt_pp_n2(θ_n,θ_,ann_l,scale_f,scale_f2,scale_f_l,ann,Vel3,Vel0,θ_l,coθ,siθ,)

vars = matread("./src/measured_data/CBC_stable_v15_6.mat")

uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series2 = transpose(hcat(uu1[1:5:end][1:spl], uu2[1:5:end][1:spl]))

np1 = θ_n[end-1];
np2 = θ_n[end];
p1, p2, p3, p4, p5, p6 = θ_[1:6]
T = [p1 p3; p2 p4]
pn1 = θ_[7:end-2]
pn2 = θ_n[1:end-2]

vlT2=pred_t(Vel3,1,θ_n,θ_,p,uu1,uu2)


newdata = deserialize("exp_flutter_pp_val3.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;

bd3 = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ, 0.0)
Vel3 = [14.9]
Vel0=[]
Ap3 = ML_Hopf.lt_pp_n2(θ_n,θ_,ann_l,scale_f,scale_f2,scale_f_l,ann,Vel0,Vel3,θ_l,coθ,siθ,)

vars = matread("./src/measured_data/CBC_unstable_v14_9.mat")

uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series3 = transpose(hcat(uu1[1:5:end][1:spl], uu2[1:5:end][1:spl]))

np1 = θ_n[end-1];
np2 = θ_n[end];
p1, p2, p3, p4, p5, p6 = θ_[1:6]
T = [p1 p3; p2 p4]
pn1 = θ_[7:end-2]
pn2 = θ_n[1:end-2]

vlT3=pred_t(Vel3,-1,θ_n,θ_,p,uu1,uu2)

newdata = deserialize("exp_flutter_pp_val4.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;

bd4 = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ, 0.0)
Vel3 = [15.6]
Vel0=[]
Ap4 = ML_Hopf.lt_pp_n2(θ_n,θ_,ann_l,scale_f,scale_f2,scale_f_l,ann,Vel0,Vel3,θ_l,coθ,siθ,)

vars = matread("./src/measured_data/CBC_unstable_v15_6.mat")
uu = get(vars, "data", 1)
uu1 = uu[ind1, 1:l]
uu2 = uu[ind2, 1:l]
uu1 = uu[ind1, 1:l] - mu1 * ones(l)
uu2 = uu[ind2, 1:l] - mu2 * ones(l)
t_series4 = transpose(hcat(uu1[1:5:end][1:spl], uu2[1:5:end][1:spl]))

np1 = θ_n[end-1];
np2 = θ_n[end];
p1, p2, p3, p4, p5, p6 = θ_[1:6]
T = [p1 p3; p2 p4]
pn1 = θ_[7:end-2]
pn2 = θ_n[1:end-2]

vlT4=pred_t(Vel3,-1,θ_n,θ_,p,uu1,uu2)

newdata = deserialize("exp_flutter_pp_full2.jls")
θ_n = newdata.θ_n;
θ_ = newdata.θ_;
p = newdata.p;
bd5 = ML_Hopf.lt_b_dia3(1, θ_n, θ_, ann_l, ann, scale_f_l, scale_f, scale_f2, coθ, siθ, 0.0)


bdp = @pgf Axis(
    {
        xlabel = "Wind speed (m/sec)",
        ylabel = "Heave amplitude (cm)",
        legend_pos = "north west",
        height = "7cm",
        width = "11cm",
        ymin = 0,
        ymax = 10,
        title="(a)"
    },
    Plot({color = "black", only_marks,style={mark_size=3.0}}, Coordinates(Vel, h * sens)),
    Plot({color = "black", only_marks, mark = "triangle*",style={mark_size=3.0}}, Coordinates(Vel2, h2 * sens)),
    
    Plot({color = "blue", only_marks,style={mark_size=3.0}}, Coordinates([Vel[1]], [h[1]] * sens)),
    Plot({color = "green", only_marks,style={mark_size=3.0}}, Coordinates([Vel[2]], [h[2]] * sens)),
    Plot({color = "red", only_marks,style={mark_size=3.0},mark = "triangle*"}, Coordinates([Vel2[1]], [h2[1]] * sens)),
    Plot({color = "brown", only_marks,style={mark_size=3.0},mark = "triangle*"}, Coordinates([Vel2[2]], [h2[2]] * sens)),

    Plot(
        {color = "blue", no_marks},
        Coordinates(vcat(reverse(bd1.v), bd1.v2), vcat(reverse(bd1.u * sens), bd1.s * sens)),
    ),
    Plot(
        {color = "green", no_marks},
        Coordinates(vcat(reverse(bd2.v), bd2.v2), vcat(reverse(bd2.u * sens), bd2.s * sens)),
    ),
    Plot(
        {color = "red", no_marks},
        Coordinates(vcat(reverse(bd3.v), bd3.v2), vcat(reverse(bd3.u * sens), bd3.s * sens)),
    ),
    Plot(
        {color = "brown", no_marks},
        Coordinates(vcat(reverse(bd4.v), bd4.v2), vcat(reverse(bd4.u * sens), bd4.s * sens)),
    ),
    
    Plot(
        {color = "black", style={line_width=1.2},dashed},
        Coordinates(vcat(reverse(bd5.v), bd5.v2), vcat(reverse(bd5.u * sens), bd5.s * sens)),
    ),
)
@pgf bdp["every axis title/.style"] = "below right,at={(0,1)}";
pgfsave("./Figure/physical/bd_val.pdf", bdp)


b_1 = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
        #ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = -4.,
        xmax = 4.,
        ymax = 6e-2,
        ymin = -6e-2,
        ylabel_shift = "-10pt",
        title = "(c-1)",
        xlabel_shift = "-5pt",
        
    },
    Plot({color = "red", no_marks}, Coordinates(Ap1[1][1, :] * sens, Ap1[1][2, :])),
    #   LegendEntry("Learnt model"),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series1[1, 1:1001] * sens, t_series1[2, 1:1001]),
    ),
    #    LegendEntry("Measured data"),
)
@pgf b_1["every axis title/.style"] = "below right,at={(0,1)}";

b_2 = @pgf Axis(
    {
        xlabel = "Time (sec)",
        #ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = 0,
        xmax = 1,
        ymax = 4,
        ymin = -4,
        ylabel_shift = "-10pt",
        title = "(c-2)",
        xlabel_shift = "-5pt",
        xtick=[0,0.5,1.]
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * vlT1[1][1,:]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series1[1, 1:1001])),
)
@pgf b_2["every axis title/.style"] = "below right,at={(0,1)}";

a_1 = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
        ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = -4.,
        xmax = 4.,
        ymax = 6e-2,
        ymin = -6e-2,
        ylabel_shift = "-15pt",
        title = "(b-1)",
        xlabel_shift = "-5pt",
        
    },
    Plot({color = "red", no_marks}, Coordinates(Ap2[1][1, :] * sens, Ap2[1][2, :])),
    #   LegendEntry("Learnt model"),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series2[1, 1:1001] * sens, t_series2[2, 1:1001]),
    ),
    #    LegendEntry("Measured data"),
)
@pgf a_1["every axis title/.style"] = "below right,at={(0,1)}";
a_2 = @pgf Axis(
    {
        xlabel = "Time (sec)",
           ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = 0,
        xmax = 1,
        ymax = 4,
        ymin = -4,
        ylabel_shift = "-15pt",
        title = "(b-2)",
        xlabel_shift = "-5pt",
        xtick=[0,0.5,1.]
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * vlT2[1][1,:]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series2[1, 1:1001])),
)
@pgf a_2["every axis title/.style"] = "below right,at={(0,1)}";

c_1 = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
       # ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = -4.,
        xmax = 4.,
        ymax = 6e-2,
        ymin = -6e-2,
        ylabel_shift = "-10pt",
        title = "(d-1)",
        xlabel_shift = "-5pt",
        
    },
    Plot({color = "red", no_marks}, Coordinates(Ap3[1][1, :] * sens, Ap3[1][2, :])),
    #   LegendEntry("Learnt model"),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series3[1, 1:1001] * sens, t_series3[2, 1:1001]),
    ),
    #    LegendEntry("Measured data"),
)
@pgf c_1["every axis title/.style"] = "below right,at={(0,1)}";

c_2 = @pgf Axis(
    {
        xlabel = "Time (sec)",
        #ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = 0,
        xmax = 1,
        ymax = 4,
        ymin = -4,
        ylabel_shift = "-10pt",
        title = "(d-2)",
        xlabel_shift = "-5pt",
        xtick=[0,0.5,1.]
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * vlT3[1][1,:]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series3[1, 1:1001])),
)
@pgf c_2["every axis title/.style"] = "below right,at={(0,1)}";

d_1 = @pgf Axis(
    {
        xlabel = L"$h$ (cm)",
        #ylabel = L"$\alpha$ (rad)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = -4.,
        xmax = 4.,
        ymax = 6e-2,
        ymin = -6e-2,
        ylabel_shift = "-10pt",
        title = "(e-1)",
        xlabel_shift = "-5pt",
        
    },
    Plot({color = "red", no_marks}, Coordinates(Ap4[1][1, :] * sens, Ap4[1][2, :])),
    #   LegendEntry("Learnt model"),
    Plot(
        {color = "blue", mark = "o", mark_size = "0.5pt"},
        Coordinates(t_series4[1, 1:1001] * sens, t_series4[2, 1:1001]),
    ),
    #    LegendEntry("Measured data"),
)
@pgf d_1["every axis title/.style"] = "below right,at={(0,1)}";

d_2 = @pgf Axis(
    {
        xlabel = "Time (sec)",
        #   ylabel = L"$h$ (cm)",
        legend_pos = "north west",
        height = "4.5cm",
        width = "4.5cm",
        xmin = 0,
        xmax = 1,
        ymax = 4,
        ymin = -4,
        ylabel_shift = "-10pt",
        title = "(e-2)",
        xlabel_shift = "-5pt",
        xtick=[0,0.5,1.]
    },
    Plot(
        {color = "red", no_marks},
        Coordinates(tv, sens * vlT4[1][1,:]),
    ),
    Plot({color = "blue", no_marks}, Coordinates(tv, sens * t_series4[1, 1:1001])),
)
@pgf d_2["every axis title/.style"] = "below right,at={(0,1)}";

gp = @pgf GroupPlot(
    {group_style = {group_size = "4 by 4",       
     horizontal_sep="20pt"},

    },
    a_1, b_1,c_1,d_1,a_2,b_2,c_2,d_2)


pgfsave("./Figure/physical/bd_val.pdf", bdp)
pgfsave("./Figure/physical/pp_ts_val.pdf", gp)
