function eq(u, p) #Flutter equation of motion
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

    du=zeros(6)
    du[1] = du_[1]
    du[2] = du_[2] + nonlinear_h 
    du[3] = du_[3]
    du[4] = du_[4] + nonlinear_theta 
    du[5] = du_[5]
    du[6] = du_[6]
    return du
end

eq2=ML_Hopf.flutter_eq_CBC
ML_Hopf.Hopf_point(eq2,[18.0])

#fold point

zz=zeros(799)

for ll=2:800
    zz[ll-1]=P[ll]-P[ll-1]
end

zz2=zz.^2
findmin(zz2)
P[295]
sqrt(18.274-14.936)*2
