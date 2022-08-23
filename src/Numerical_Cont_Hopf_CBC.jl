#=
Numerical_Cont.jl is able to do numerical continuation of system with Hopf bifurcation starting from the stable LCO solutions.
Equation of motion should be expressed in nonmutating form as flutter_eq_CBC_nonmu(u, p, t)
and the dimension of the equation should be defined as: dimension(::typeof(flutter_eq_CBC_nonmu)) = 6
=#

function fourier_diff(N::Integer)
    # For a MATLAB equivalent see http://appliedmaths.sun.ac.za/~weideman/research/differ.html
    h = 2π/N
    D = zeros(N, N)
    # First column
    n = ceil(Int, (N-1)/2)
    if (N % 2) == 0
        for i in 1:n
            D[i+1, 1] = -((i % 2) - 0.5)*cot(i*h/2)
        end
    else
        for i in 1:n
            D[i+1, 1] = -((i % 2) - 0.5)*csc(i*h/2)
        end
    end
    for i in n+2:N
        D[i, 1] = -D[N-i+2, 1]
    end
    # Other columns (circulant matrix)
    for j in 2:N
        D[1, j] = D[N, j-1]
        for i in 2:N
            D[i, j] = D[i-1, j-1]
        end
    end
    return D
end

function periodic_zero_problem(rhs, u, p, ee)
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
    n = dimension(rhs)
    N = (length(u)-1)÷n
    D = fourier_diff(N)
    T = u[end]/(2π)  # the differentiation matrix is defined on [0, 2π]
    # Evaluate the right-hand side of the ODE
    res=Vector{Float64}(undef, n*N+1)
    for i in 1:n:n*N
        res[i:i+n-1] = T*rhs(@view(u[i:i+n-1]), p, 0)  # use a view for speed
    end
    # Evaluate the derivative; for-loop for speed equivalent to u*Dᵀ
    for (i, ii) in pairs(1:n:n*N)
        # Evaluate the derivative at the i-th grid point
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                res[ii+k-1] -= D[i, j]*u[jj+k-1]
            end
        end
    end
    res[end] = (u[1] - ee)  # phase condition - assumes that the limit cycle passes through the Poincare section u₁=1e-4; using a non-zero value prevents convergence to the trivial solution
    return res
end

function periodic_zero_problem2(rhs, u, p, ee, dv, pu, ds) # Periodic zero problem including Pseudo arc-length equation
    # Assume that u has dimensions nN + 1 where n is the dimension of the ODE
    n = dimension(rhs)
    N = (length(u)-1)÷n
    res=Vector{Float64}(undef, n*N+2)
    res1=periodic_zero_problem(rhs, u, p, ee)
    p1=p[1];
    uu=[u;p1]
    du=uu-pu

    res[1:end-1]=res1
    arclength=norm(transpose(dv)*du)
    res[end] = arclength-ds  #Pseudo-arclength equation
    return res
end

function periodic_zero_J(jeq, em, u, p, ee) # Jacobian of periodic zero problem
    n = dimension(em)
    N = (length(u)-1)÷n
    T = u[end]/(2π)
    J = zeros(n*N+1,n*N+1)
    D = fourier_diff(N)
    for i in 1:n:n*N
        J[i:i+n-1,i:i+n-1] = T*jeq(@view(u[i:i+n-1]), p, 0)
        J[i:i+n-1,n*N+1] = em(@view(u[i:i+n-1]), p, 0)
    end
    for (i, ii) in pairs(1:n:n*N)
        for (j, jj) in pairs(1:n:n*N)
            for k = 1:n
                J[ii+k-1,jj+k-1] -= D[i, j]
            end
        end
    end
    J[n*N+1,1:n*N+1] = zeros(1,n*N+1)
    J[n*N+1,1] = 1
    return J
end

function periodic_zero_J2(jeq, jeq2, em, u, p, ee, dv)  # Jacobian of periodic zero problem 2
    n=dimension(em)
    N = (length(u)-1)÷n
    J = zeros(n*N+2,n*N+2)
    T = u[end]/(2π)
    J[1:n*N+1,1:n*N+1]=periodic_zero_J(jeq, em, u, p, ee)
    for i in 1:n:n*N
        J[i:i+n-1,n*N+2] = T*jeq2(@view(u[i:i+n-1]), p, 0)
    end
    J[n*N+2,:]=dv
    return J
end

function LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol) # Solve the periodic zero problem with Newton methods
    u0=vcat(u1,ini_T)
    err=[]
    for i in 1:40
        res=periodic_zero_problem(eq, u0, p, ee)
        u1=u0-periodic_zero_J(jeq, eq, u0, p, ee)\res
        u0=u1
        er=transpose(res)*res
        err=vcat(err,er)
        if er<z_tol
            break
        end
    end
    return (u=u0, err=err)
end

function LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,z_tol,dv,pu,ds) # Solve the periodic zero problem 2 with Newton methods
    p0=p[1]
    u0=[u1;ini_T]
    uu0=[u0;p0]
    err=[]
    np=p
    for i in 1:40
        res=periodic_zero_problem2(eq, u0, np, ee, dv, pu, ds)
        uu1=uu0-periodic_zero_J2(jeq, jeq2, eq, u0, np, ee, dv)\res
        uu0=uu1
        u0=uu0[1:end-1]
        p0=uu0[end]
        np=vcat(p0,p[2:end])
        er=norm(transpose(res)*res)
        err=vcat(err,er)
        if er<z_tol
            break
        end
    end
    return (u=uu0, err=err)
end

function get_sol(u0,N,ind1,ind2) # convert solution of collocation -> Array form
    # returns phase angle, amplitude
    dim=Int((length(u0)-1)/N)
    u=u0[1:end-1]
    T=u0[end]
    uu=Array{Float64}(undef,dim,N)
    theta=Array{Float64}(undef,N)
    r=Array{Float64}(undef,N)
    for i in 1:dim
        uu[i,:]=u[i:dim:end]
    end
    for i in 1:N
        theta[i]=atan(uu[ind2,i],uu[ind1,i])
        r[i]=sqrt(uu[ind1,i]^2+uu[ind2,i]^2)
    end
    return (u=uu,T=T,t=theta,r=r)
end

function get_sol(u0,dim,N,ind1,ind2)  # Convert solution of periodic zero-problem -> array solution and amplitude, phase angle of measured coordinate (z_1,z_2)
    u=u0[1:end-1]
    T=u0[end]
    uu=Array{Float64}(undef,N,dim)
    theta=Array{Float64}(undef,N)
    r=Array{Float64}(undef,N)
    for i in 1:dim
        uu[:,i]=u[i:dim:end]
    end
    for i in 1:N
        theta[i]=atan(uu[i,ind2],uu[i,ind1])
        r[i]=sqrt(uu[i,ind1]^2+uu[i,ind2]^2)
    end
    return (u=uu,T=T,t=theta,r=r)
end

function get_sol2(sol,ind) # Convert solution of ODE solver to array solution
    lu=length(sol.u)
    u=Vector{Float64}(undef, lu)
    for i in 1:lu
        uv=sol.u[i]
        u[i]=uv[ind]
    end
    return u
end

function get_stable_LCO(p,u0,tl,tol,eq,stol,rp) # Get a stable LCO from the numerical integration
    u=u0
    dim=length(u0)
    prob = ODEProblem(eq,u,(0,tl*rp),p)
    sol = solve(prob,Tsit5(),reltol=stol,abstol=stol)
    vP=1
    P=0
    T=0
    while vP>tol
        u=sol.u[end]
        prob = ODEProblem(eq,u,(0,tl),p)
        sol = solve(prob,Tsit5(),reltol=stol,abstol=stol)
        z=zero_measure(sol.u,1,sol.t)
        vP=var(z.hp)
        P=z.P[1]
        T=z.T
    end
    tl=length(sol)
    uu=Array{Float64}(undef,tl,dim)
    for i in 1:dim
        uu[:,i]=get_sol2(sol,i)
    end
    t=Array{Float64}(undef,length(sol))
    r=Array{Float64}(undef,length(sol))
    ind1=1;ind2=3;u=uu[:,ind1];v=uu[:,ind2]
    for i in 1:length(u)
        t[i]=atan(v[i],u[i])
        r[i]=sqrt(u[i]^2+v[i]^2)
    end
    return (u=uu,t=t,r=r,P=P,T=T)
end

function get_stable_LCO(p,u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st) # Get a stable LCO from numerical integration
    u=u0
    dim=length(u0)
    prob = ODEProblem(eq,u,(0,tl*rp),p)
    sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol,saveat=st)
    vP=1
    P=0
    T=0
    while vP>tol
        u=sol.u[end]
        prob = ODEProblem(eq,u,(0,tl),p)
        sol = DifferentialEquations.solve(prob,Tsit5(),reltol=stol,abstol=stol,saveat=st)
        z=zero_measure(sol.u,1,sol.t)
        vP=Statistics.var(z.hp)
        P=z.P[1]
        T=z.T
    end
    tl=length(sol)
    uu=Array{Float64}(undef,tl,dim)
    for i in 1:dim
        uu[:,i]=get_sol2(sol,i)
    end
    t=Array{Float64}(undef,length(sol))
    r=Array{Float64}(undef,length(sol))
    u=uu[:,ind1];v=uu[:,ind2]
    for i in 1:length(u)
        t[i]=atan(v[i]-v₀,u[i]-u₀)
        r[i]=sqrt((u[i]-u₀)^2+(v[i]-v₀)^2)
    end
    return (u=uu,t=t,r=r,P=P,T=T)
end

function Newton_CBC(u1,ini_T,eq,jeq,p,ee,z_tol,c_tol,ind1,ind2) # Newton method to solve zero-problem of CBC
    p0=p[1:3]
    c0=p[4:end]
    nh=Int((length(c0)-1)/2)
    dm=dimension(eq)
    N=Int(length(u1)/dm)
    s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,6,N,ind1,ind2)
    r=LS_harmonics(g.r,g.t,1,nh)
    c=r.coeff
    del_c0=c-c0
    err=transpose(del_c0)*del_c0
    er=err
    for i in 1:10
        if err<c_tol
            break
        end
        J=Jac_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol)
        del_c=inv(J)*del_c0
        c1=c0-del_c
        p=vcat(p0,c1)
        s=LCO_solve(u1,ini_T,eq,jeq,p,ee,z_tol)
        g=get_sol(s.u,6,N,ind1,ind2)
        r=LS_harmonics(g.r,g.t,1,nh)
        del_c1=r.coeff-c1
        del_c0=del_c1
        c0=c1
        err=transpose(del_c1)*del_c1
        er=vcat(er,err)
    end
    return (s=s,p=p,e=er,u1=s.u[1:end-1],T1=s.u[end])
end

function Newton_CBC(u1,ini_T,eq,jeq,p,ee,z_tol,c_tol,ind1,ind2,dv,pu,ds)  # Newton method to solve zero-problem of CBC including Pseudo arclength equation
    p0=p[1:3]
    c0=p[4:end]
    nh=Int((length(c0)-1)/2)
    dm=dimension(eq)
    N=Int(length(u1)/dm)
    rr=res_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds)
    res=rr.res
    u1=rr.u1;ini_T=rr.T1;
    err=transpose(res)*res
    er=err
    for i in 1:30
        if err<c_tol
            break
        end
        J=Jac_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv)
        del_u=inv(J)*res
        c1=c0-del_u[1:end-1]
        p0[1]=p0[1]-del_u[end]
        p=vcat(p0,c1)
        rr=res_CBC(p,u1,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds)
        res=rr.res
        u1=rr.u1;ini_T=rr.T1;
        c0=c1
        err=transpose(res)*res
        er=vcat(er,err)
    end
    return (u1=u1,T1=ini_T,p=p,e=er)
end

function Jac_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol) # Jacobian of CBC zero-problem
    pp=p[1:3]
    c0=p[4:end]
    dm=dimension(eq)
    nh=Int((length(c0)-1)/2)
    N=Int(length(u0)/dm)
    s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    c=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c0=c-c0
    J=Array{Float64}(undef,2*nh+1,2*nh+1)
    eye=SMatrix{2*nh+1,2*nh+1}(1I)
    delta=1e-11
    for i in 1:2*nh+1
        c=c0+eye[i,:]*delta
        np=vcat(pp,c)
        s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
        g=get_sol(s.u,dm,N,ind1,ind2)
        dc=LS_harmonics(g.r,g.t,1,nh).coeff
        del_c1=dc-c
        dp=(del_c1-del_c0)/delta
        J[:,i]=dp
    end
    return J
end

function Jac_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv) #Jacobian of CBC zero problem with Pseudo arclength
    pp=p[1:3]
    c0=p[4:end]
    dm=dimension(eq)
    nh=Int((length(c0)-1)/2)
    N=Int(length(u0)/dm)
    s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    c=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c0=c-c0
    J=Array{Float64}(undef,2*nh+2,2*nh+2)
    eye=SMatrix{2*nh+1,2*nh+1}(1I)
    delta=1e-11
    for i in 1:2*nh+1
        c=c0+eye[i,:]*delta
        np=vcat(pp,c)
        s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
        g=get_sol(s.u,dm,N,ind1,ind2)
        dc=LS_harmonics(g.r,g.t,1,nh).coeff
        del_c1=dc-c
        dp=(del_c1-del_c0)/delta
        J[1:end-1,i]=dp
    end
    npp=pp
    npp[1]=npp[1]+delta
    np=vcat(npp,c0)
    s=LCO_solve(u0,ini_T,eq,jeq,np,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    dc=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c1=dc-c0
    dp=(del_c1-del_c0)/delta
    J[1:end-1,end]=dp
    J[end,:]=dv
    return J
end

function res_CBC(p,u0,ini_T,eq,jeq,ee,ind1,ind2,z_tol,dv,pu,ds) # Residue of zero problem of CBC
    pp=p[1:3]
    c0=p[4:end]
    dm=dimension(eq)
    nh=Int((length(c0)-1)/2)
    N=Int(length(u0)/dm)
    #res=Array{Float64}(undef,2*nh+2,1)
    res=Vector{Float64}(undef, 2*nh+2)
    s=LCO_solve(u0,ini_T,eq,jeq,p,ee,z_tol)
    g=get_sol(s.u,dm,N,ind1,ind2)
    c=LS_harmonics(g.r,g.t,1,nh).coeff
    del_c0=c-c0
    uu=[c0;pp[1]]
    du=uu-pu
    res[1:end-1]=del_c0
    arclength=transpose(du)*dv
    res[end] = arclength-ds
    return (res=res,u1=s.u[1:end-1],T1=s.u[end])
end

function zero_measure(u,ind,t) # Measuring the time of zero crossing points from numerical continuation
    l=length(u)
    zero=Array{Float64}(undef, 0)
    T=Array{Float64}(undef, 0)
    low_p=Array{Float64}(undef, 0)
    high_p=Array{Float64}(undef, 0)
    Ti=Array{Int64}(undef, 0)
    for i in 2:l-1
        sign_con2=u[i][ind+1]*u[i-1][ind+1]
        if sign_con2 < 0
            if (u[i][ind]+u[i-1][ind])/2 < 0
                low_p=vcat(low_p,(u[i][ind]+u[i-1][ind])/2)
            else
                high_p=vcat(high_p,(u[i][ind]+u[i-1][ind])/2)
            end
        end
    end
    h₀=mean(high_p)+mean(low_p)
    h₀=h₀/2
    for i in 2:l-1
        sign_con=(u[i][ind]-h₀)*(u[i+1][ind]-h₀)
        if sign_con < 0
            if (u[i][ind+1]+u[i+1][ind+1])/2 > 0
                zero=vcat(zero,(u[i][ind]+u[i-1][ind])/2)
                Ti=vcat(Ti,i)
                T=vcat(T,t[i])
            end
        end
    end
    t_l=length(T)
    P=Array{Float64}(undef, t_l-1)
    for j in 2:t_l
        P[j-1]=T[j]-T[j-1]
    end
    return (T=Ti, P=P, hp=high_p, lp=low_p, h₀=h₀)
end

function MonoD(eq,dim,u,p,t) # Variational equation for monodromy matrix computation (ForwardDiff is used for jacobian computation)
    jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
    n=dim
    v=u[1:Int(n*n)]
    uu=u[Int(n*n+1):end]
    M=reshape(v, n, n)
    J=jeq(uu,p,t)
    dM=J*M
    dv=vec(dM)
    duu=eq(uu,p,t)
    duu=vec(duu)
    du=[dv;duu]
    return du
end

function Monodromy_compute(eq,u,p,N,ind) # Monodromy matrix computation using numerical integration
    n=dimension(eq)
    eye=1.0*Matrix(I, n, n)
    v0=vec(eye)
    g=get_sol(u,n,N,ind[1],ind[2])
    T=g.T
    uu=g.u
    M=eye
    tl2=T/N
    for i in 1:N
        uu0=uu[i,:]
        u0=[v0;uu0]
        prob = ODEProblem((u,p,t) -> MonoD(eq,n,u,p,t),u0,(0,tl2),p)
        sol = solve(prob,Tsit5(),reltol=1e-14,abstol=1e-14)
        w=sol.u[end]
        m1=w[1:Int(n*n)]
        M1=reshape(m1, n, n)
        M=M1*M
    end
    Eig1=eigen(M)
    μ=Eig1.values
    return μ
end

function LS_harmonics(r,t,ω,N) # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
    # Fourier coefficients are computed in least square sence
    c=Array{Float64}(undef,2*N+1)
    M=Array{Float64}(undef,1,2*N+1)
    tM=Array{Float64}(undef,0,2*N+1)
    tl=length(t)
    rr=Array{Float64}(undef,tl)
    M[1]=1
    for j in 1:tl
        for i in 1:N
            M[1+i]=cos(ω*t[j]*i)
            M[1+N+i]=sin(ω*t[j]*i)
        end
        tM=vcat(tM,M)
    end
    MM=transpose(tM)*tM
    rN=transpose(tM)*r
    c=inv(MM)*rN
    for j in 1:tl
        rr[j]=c[1]
        for i in 1:N
            rr[j]+=c[i+1]*cos(ω*t[j]*i)
            rr[j]+=c[i+1+N]*sin(ω*t[j]*i)
        end
    end
    return (coeff=c,rr=rr)
end

function LS_h_recon(tl,c,N) # Computing Fourier coefficients of the amplitude in the measued state-variable coordinates
    # Fourier coefficients are computed in least square sence
    t=range(0.0,2π,length=tl+1)
    t=t[1:end-1]
    rr=zeros(length(t))
    for j in 1:tl
        rr[j]=c[1]
        for i in 1:N
            rr[j]+=c[i+1]*cos(1*t[j]*i)
            rr[j]+=c[i+1+N]*sin(1*t[j]*i)
        end
    end
    return (r=rr,t=t)
end

function ini_val(p,tl,tol,N,eq,u0,stol,rp) #Get stable LCO from the ODE numerical solutions- initial collocation points
    dim=length(u0)
    s0=get_stable_LCO(p,u0,tl,tol,eq,stol,rp)
    pp=s0.P[1]
    ini_T=pp
    mu=s0.u[:,1]
    u=mu-1e-3*ones(length(mu))
    u=broadcast(abs, u)
    ps=findall(isequal(minimum(u)), u)
    ps=ps[1]
    u0= s0.u[ps,:]
    #Get a time series of periodic solution
    tl2=pp/N
    #ini_u=Array{Float64}(undef,Int(dim*N))
    ini_u=Vector{Float64}(undef, Int(dim*N))
    ini_u[1:dim]=u0
    #uu=pu+ds*dv;
    for i in 1:N-1
        prob = ODEProblem(eq,u0,(0,tl2),p)
        sol = solve(prob,Tsit5(),reltol=1e-11,abstol=1e-11)
        u0=sol.u[end]
        ini_u[Int(dim*i+1):Int(dim*i+dim)]=u0
    end
    return (u=ini_u,T=ini_T)
end

function get_CBC_u(p)
    c0=p[4:end]
    nu=p[1]
    u=vcat(c0,nu)
    return u
end

function get_CBC_p(nu,Kp,Kd,N)
    c=nu[1:end-1]
    p=vcat([nu[end],Kp,Kd],c)
    return p
end

function get_dv_u(u,T1,pu,ds)
    uu=vcat(u,T1)
    dv=uu-pu;dv=dv/norm(dv)
    nu=uu+dv*ds
    u1=nu[1:end-1]
    T1=nu[end]
    return (u=u1,T=T1)
end

function amp_LCO(U,ind,N) #Extract amplitude from collocation points
    l1=length(U)
    amp=Vector{Float64}(undef, l1)
    for i in 1:l1
        g=get_sol(U[i],6,N,1,3)
        uu=g.u[:,ind]
        amp[i]=maximum(uu)-minimum(uu)
    end
    return amp
end

function Hopf_point(eq,p) # Compute the Hopf point
t=0.0;u=zeros(dimension(eq))
jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
s=nlsolve(p0->maximum(real(eigen(jeq(u,p0,t)).values))-1e-8,p)
    return s.zero[1]
end
## Numerical continuation of Flutter model and CBC zero problem (Examples)

function flutter_eq_CBC(u, p, t) #  Equation of motion of Flutter model with CBC
    μ=p[1]
    ind1=1;ind2=3;
    theta=atan(u[ind2],u[ind1])
    if length(p)>=2
        Kp=p[2]
        Kd=p[3]
        c=p[4:end]
        r=c[1]
        h=c[2:end]
        nh=Int(length(h)/2)
        for i=1:nh
            r+=h[i]*cos(i*theta)+h[i+nh]*sin(theta*i)
        end
    else
        Kp=0
        Kd=0
        r=0
        theta=0
    end
#theta is computed from two state variables

    b=0.15; a=-0.5; rho=1.204; x__alpha=0.2340
    c__0=1; c__1=0.1650; c__2=0.0455; c__3=0.335; c__4=0.3; c__alpha=0.562766779889303; c__h=15.4430
    I__alpha=0.1726; k__alpha=54.116182926744390; k__h=3.5294e+03; m=5.3; m__T=16.9
    U=μ

    MM = [b ^ 2 * pi * rho + m__T -a * b ^ 3 * pi * rho + b * m * x__alpha 0; -a * b ^ 3 * pi * rho + b * m * x__alpha I__alpha + pi * (0.1e1 / 0.8e1 + a ^ 2) * rho * b ^ 4 0; 0 0 1;]
    DD = [c__h + 2 * pi * rho * b * U * (c__0 - c__1 - c__3) (1 + (c__0 - c__1 - c__3) * (1 - 2 * a)) * pi * rho * b ^ 2 * U 2 * pi * rho * U ^ 2 * b * (c__1 * c__2 + c__3 * c__4); -0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (b ^ 2) * (c__0 - c__1 - c__3) * U c__alpha + (0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b ^ 3) * U -0.2e1 * pi * rho * (U ^ 2) * (b ^ 2) * (a + 0.1e1 / 0.2e1) * (c__1 * c__2 + c__3 * c__4); -1 / b a - 0.1e1 / 0.2e1 (c__2 + c__4) * U / b;]
    KK = [k__h 2 * pi * rho * b * U ^ 2 * (c__0 - c__1 - c__3) 2 * pi * rho * U ^ 3 * c__2 * c__4 * (c__1 + c__3); 0 k__alpha - 0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b ^ 2) * (U ^ 2) -0.2e1 * pi * rho * b * (U ^ 3) * (a + 0.1e1 / 0.2e1) * c__2 * c__4 * (c__1 + c__3); 0 -U / b c__2 * c__4 * U ^ 2 / b ^ 2;]

    K1=-inv(MM)*KK;
    D1=-inv(MM)*DD;

    J1=[0 1 0 0 0 0]
    J2=[K1[1,1] D1[1,1] K1[1,2] D1[1,2] K1[1,3] D1[1,3]]
    J3=[0 0 0 1 0 0]
    J4=[K1[2,1] D1[2,1] K1[2,2] D1[2,2] K1[2,3] D1[2,3]]
    J5=[0 0 0 0 0 1]
    J6=[K1[3,1] D1[3,1] K1[3,2] D1[3,2] K1[3,3] D1[3,3]]

    J=[J1;J2;J3;J4;J5;J6]
    du=J*u
    #Control added on heave
    M2=inv(MM)
    control=Kp*(r*cos(theta)-u[ind1])+Kd*(r*sin(theta)-u[ind2])
    ka2=751.6; ka3=5006;
    nonlinear_h=-M2[1,2]*(ka2*u[3]^2+ka3*u[3]^3)
    nonlinear_theta=-M2[2,2]*(ka2*u[3]^2+ka3*u[3]^3)

    du[2]+=nonlinear_h+M2[1,1]*control
    du[4]+=nonlinear_theta+M2[2,1]*control
    return du
end

function flutter_eq_2(u, p, t) #  Equation of motion of Flutter model with CBC
    μ=p[1]
    ind1=1;ind2=3;
    theta=atan(u[ind2],u[ind1])
    if length(p)>=2
        Kp=p[2]
        Kd=p[3]
        c=p[4:end]
        r=c[1]
        h=c[2:end]
        nh=Int(length(h)/2)
        for i=1:nh
            r+=h[i]*cos(i*theta)+h[i+nh]*sin(theta*i)
        end
    else
        Kp=0
        Kd=0
        r=0
        theta=0
    end
#theta is computed from two state variables

    b=0.15; a=-0.5; rho=1.204; x__alpha=0.24
    c__0=1; c__1=0.1650; c__2=0.0455; c__3=0.335; c__4=0.3; c__alpha=0.562766779889303; c__h=14.5756
    I__alpha=0.1726; k__alpha=54.11; k__h=3.5294e+03; m=5.3; m__T=16.9
    U=μ

    MM = [b ^ 2 * pi * rho + m__T -a * b ^ 3 * pi * rho + b * m * x__alpha 0; -a * b ^ 3 * pi * rho + b * m * x__alpha I__alpha + pi * (0.1e1 / 0.8e1 + a ^ 2) * rho * b ^ 4 0; 0 0 1;]
    DD = [c__h + 2 * pi * rho * b * U * (c__0 - c__1 - c__3) (1 + (c__0 - c__1 - c__3) * (1 - 2 * a)) * pi * rho * b ^ 2 * U 2 * pi * rho * U ^ 2 * b * (c__1 * c__2 + c__3 * c__4); -0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (b ^ 2) * (c__0 - c__1 - c__3) * U c__alpha + (0.1e1 / 0.2e1 - a) * (1 - (c__0 - c__1 - c__3) * (1 + 2 * a)) * pi * rho * (b ^ 3) * U -0.2e1 * pi * rho * (U ^ 2) * (b ^ 2) * (a + 0.1e1 / 0.2e1) * (c__1 * c__2 + c__3 * c__4); -1 / b a - 0.1e1 / 0.2e1 (c__2 + c__4) * U / b;]
    KK = [k__h 2 * pi * rho * b * U ^ 2 * (c__0 - c__1 - c__3) 2 * pi * rho * U ^ 3 * c__2 * c__4 * (c__1 + c__3); 0 k__alpha - 0.2e1 * pi * (a + 0.1e1 / 0.2e1) * rho * (c__0 - c__1 - c__3) * (b ^ 2) * (U ^ 2) -0.2e1 * pi * rho * b * (U ^ 3) * (a + 0.1e1 / 0.2e1) * c__2 * c__4 * (c__1 + c__3); 0 -U / b c__2 * c__4 * U ^ 2 / b ^ 2;]

    K1=-inv(MM)*KK;
    D1=-inv(MM)*DD;

    J1=[0 1 0 0 0 0]
    J2=[K1[1,1] D1[1,1] K1[1,2] D1[1,2] K1[1,3] D1[1,3]]
    J3=[0 0 0 1 0 0]
    J4=[K1[2,1] D1[2,1] K1[2,2] D1[2,2] K1[2,3] D1[2,3]]
    J5=[0 0 0 0 0 1]
    J6=[K1[3,1] D1[3,1] K1[3,2] D1[3,2] K1[3,3] D1[3,3]]

    J=[J1;J2;J3;J4;J5;J6]
    du=J*u
    #Control added on heave
    M2=inv(MM)
    control=Kp*(r*cos(theta)-u[ind1])+Kd*(r*sin(theta)-u[ind2])
    ka2=751.6; ka3=5006.7;
    nonlinear_h=-M2[1,2]*(ka2*u[3]^2+ka3*u[3]^3)
    nonlinear_theta=-M2[2,2]*(ka2*u[3]^2+ka3*u[3]^3)

    du[2]+=nonlinear_h+M2[1,1]*control
    du[4]+=nonlinear_theta+M2[2,1]*control
    return du
end

dimension(::typeof(flutter_eq_CBC)) = 6 # dimension should be defined

function continuation_flutter(N,sp,sU,ds) #Get stable LCO from the ODE numerical solutions- initial collocation points
    μ=sU+0.02;p=[μ,0,0,0,0,0];ee=1e-3;
    eq=flutter_eq_CBC;
    tol=1e-7;
    jeq=(u,p,t) -> ForwardDiff.jacobian(u -> eq(u,p,t), u)
    jeq2=(u,p,t) -> ForwardDiff.jacobian(p -> eq(u,p,t), p)[:,1]
    stol=1e-10;rp=5;u0=[0.1;0.1;0.1;0;0;0];tl=10.0
    s1=ini_val(p,tl,tol,N,eq,u0,stol,rp)
    s=LCO_solve(s1.u,s1.T,eq,jeq,p,s1.u[1],tol)
    μ=sU;p=[μ,0,0,0,0,0];
    s2=LCO_solve(s1.u,s1.T,eq,jeq,p,s1.u[1],tol)

    du=s2.u-s.u
    du=[du;-0.02]
    dv=du/norm(du)
    dim=6;ll=dim*N+1;
    V = [zeros(ll) for _ in 1:sp]
    P = zeros(sp)
    pu=s2.u;pu=[pu;sU]
    V[1]=vec(s2.u);P[1]=sU

    uu=pu+ds*dv
    μ=uu[end];p=[μ,0,0,0,0,0];u=uu[1:end-1];
    u1=u[1:end-1];ini_T=u[end];z_tol=tol*0.1;
    for i in 2:sp
        s2=LCO_solve2(u1,ini_T,eq,jeq,jeq2,p,ee,z_tol,dv,pu,ds)
        s2.err
        du=s2.u-pu;dv=du/norm(du)
        V[i]=s2.u[1:end-1];P[i]=s2.u[end]
        pu=s2.u;
        uu=pu+ds*dv;
        p=[uu[end],0,0,0,0,0];u=uu[1:end-1];
        u1=u[1:end-1];ini_T=u[end];z_tol=tol*0.1;
    end
    return (V=V,P=P)
end


function hhcat(amp) # vcat vector of vectors
    aamp=amp[1]
    for ii=2:length(amp)
        aamp=hcat(aamp,amp[ii])
    end
    return aamp
end

function generate_data(vel_l,Vel,nh,st,s_ind,U,N, θ_l) # Training data
    #Generate training data
    u0=Float32[2e-1,0,2e-1,0,0,0];
    tol=1e-7;stol=1e-8
    eq=flutter_eq_CBC;
    rp=5;ind1=1;ind2=3;
    pp=[zeros(10) for i in 1:vel_l]
    AA=zeros(vel_l,Int(nh*2+1))

    p_=zeros(9)
    p_=vcat(Vel[1],p_)
    tl=10.0
    g=get_stable_LCO(p_,u0,tl,tol,eq,stol,rp,ind1,ind2,0.0,0.0,st)

    u₀=mean(g.u[:,1]);v₀=mean(g.u[:,3])
    for i in 1:vel_l
        pp[i][1]=Vel[i]
        g=get_stable_LCO(pp[i],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st)
        r=g.r;t=g.t;
        c=LS_harmonics(r,t,1,nh).coeff
        AA[i,:]=c
    end
    cc=[LS_harmonics(get_sol(U[s_ind[i]],N,1,3).r,get_sol(U[s_ind[i]],N,1,3).t,1,nh).coeff for i in 1:length(s_ind)]
    cc=hhcat(cc)

    AA=transpose(AA)
    Al=AA
    AA=hcat(AA,cc)
    t_series=[Transpose(get_stable_LCO([Vel[i] transpose(zeros(9))],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st).u[:,[1,3]]) for i in 1:vel_l]
    θ_series=[get_stable_LCO([Vel[i] transpose(zeros(9))],u0,tl,tol,eq,stol,rp,ind1,ind2,u₀,v₀,st).t for i in 1:vel_l]
    θ=range(0, stop = 2π, length = θ_l)
    coθ=cos.(θ)
    siθ=sin.(θ)
    return (data=AA,ts=t_series,d0=[u₀,v₀],data2=Al,coθ=coθ,siθ=siθ,theta_s=θ_series)
end


export continuation_flutter