% This file reproduces fig 1 in the paper, to compare knn hatrho and kde hatp

% the code (including the for loops for all values of k) runs in about 90 secends.


clear all; rng(2020);

%% data and p

%p = 1-(1-\delta_p)*sin(2pi*t)

delta_p = 0.05;

p0_func = @(t, w, ome) 1+ w*sin( ome * 2*pi *t);
F0_func = @(t, w, ome) t + w*(1- cos(ome * 2*pi * t))/(ome*2*pi);
dp0_func = @(t,w,ome) w*cos( ome * 2*pi *t)*(ome * 2*pi);

p_func = @(t) p0_func(t, -1+delta_p, 1 );
F_func = @(t) F0_func(t, -1+delta_p, 1);
dp_func = @(t) dp0_func(t, -1+delta_p, 1 );

%%

dM =1;

rho_func = @(t) (p_func(t)).^(-1/dM);
drho_func = @(t) ((-1/dM)*rho_func(t)).*(dp_func(t)./p_func(t));

map_to_RD_func = @(t) [1/(2*pi)*cos(2*pi * t), 1/(2*pi)*sin(2*pi * t)];

% grid used to sample ~ pdt
dtgrid = 1e-5;
tgrid = (0: dtgrid : 1)';
pgrid = p_func(tgrid);
Fgrid = F_func(tgrid);

% visualize density, sample by approx inverse
n = 1e4;

%tic,
tt= sample_by_cdf_1d( tgrid, Fgrid, n);
%toc 

figure(11),clf; hold on;
[nout,xout] = hist( tt, 100);
plot( xout, (nout/n)/(xout(2)-xout(1)),'x-')
plot( xout, p_func( xout), '.-')
grid on; axis([0, 1, 0, max(pgrid)+.5]);
legend('sampled distribution', 'p')

%%

Ny = 5000;
ky_list = floor(2.^(3:0.5:8));


nrow = numel(ky_list);

% sample X as a grid on M
Nx = 100;
tX = (1/Nx : 1/Nx :1)';
dataX = map_to_RD_func(tX);
% true rho(x)
barrho= rho_func(tX);
barp = p_func(tX);

m0hknn=2;
barR = (1./(m0hknn*barp)* ky_list/Ny).^(1/dM); % barR = (1/(m0*p)*k/N)^1/d
epsx_list = min(barR.^2); %match the min of variable bandwidth

%%
ah =4/pi;
h_func = @(xi) exp(-xi/ah);
dh_func = @(xi) -1/ah*exp(-xi/ah);
m0_h = 2;


nrun = 500;

if_plot = 1;% 0;

errrho = zeros( Nx, nrow, nrun);
errp = zeros( Nx, nrow, nrun);

%%
for irow =1:nrow
    ky = ky_list(irow);
    
    fprintf('ky= %d\n', ky);
    for irun = 1:nrun
        
        fprintf('-%d-',irun);
        
        %% sample Y
        tY = sample_by_cdf_1d( tgrid, Fgrid, Ny);
        dataY = map_to_RD_func(tY);
        
        k_knn = ky*20; %Ny;
        %tic,
        [~, disXY] = knnsearch( dataY, dataX, 'k', k_knn);
        %toc
        
        %% estimate hatrho by knn
        m0_knn = 2;
        hatR_knn = disXY(:,ky);
        hatrho_knn = ((1/m0_knn*(ky/Ny))^(-1/dM))*hatR_knn;
        
        hatp2 = hatrho_knn.^(-dM);
        
        errrho(:,irow,irun) = abs(barrho-hatrho_knn)./barrho;
        
        %% hatp by kde
        epsx = epsx_list(irow);
        hatp = (epsx^(-dM/2)/m0_h)* (1/Ny*sum(h_func( disXY.^2/epsx),2));
        
        errp(:, irow, irun) = abs( hatp- barp)./barp;
        
        if if_plot && irun ==1
            
            figure(1),clf;
            subplot(121), hold on;
            plot( tX, barrho, '-b');
            plot(tX, hatrho_knn, 'xr');
            grid on; 
            title(sprintf('Ny=%d,ky=%d',Ny,ky), 'Interpreter','latex');
            legend('barrho','hatrho-knn');
            set(gca,'FontSize',20);
            subplot(122), hold on;
            plot( tX,  errrho(:,irow,irun), 'x-b');
            grid on; set(gca,'FontSize',20);
            title('relative error','Interpreter','latex');
            
            %
            figure(3),clf;
            subplot(121), hold on;
            plot( tX, barp, '-b');
            plot(tX, hatp , 'xr');
            plot(tX, hatp2 , 'xg');
            grid on; 
            title(sprintf('sqrt(epsx)=%6.4f',sqrt(epsx)), 'Interpreter','latex');
            legend('p','hatp')
            set(gca,'FontSize',20);
            subplot(122), hold on;
            plot( tX, errp(:, irow, irun), 'x-b');
            grid on; 
            set(gca,'FontSize',20);
            title('relative error','Interpreter','latex');
            drawnow();
        end
        
    end
    fprintf('\n');
end
 
%% generate plots

merr1 = mean( errrho, 3);
merr2 = mean( errp, 3);

figure(5),clf; %hold on;
%imagesc( tX, ky_list, merr1');
surf( tX, ky_list, merr1');
title('relative error $\hat{\rho}$ Knn','Interpreter','latex')
xlabel('$t_X$','Interpreter','latex');
ylabel('$k_y$','Interpreter','latex');
%axis([min(tX),max(tX),min(ky_list),max(ky_list)])
view(40,40); grid on;
%colorbar(); 
set(gca,'FontSize',20);
axis([0,1,0,300,0,1.5])


figure(6),clf;
%imagesc( tX, sqrt(epsx_list), merr2');
surf( tX, sqrt(epsx_list), merr2');
title('relative error $\hat{p}$ fixed $\epsilon$','Interpreter','latex')
xlabel('$t_X$','Interpreter','latex');
ylabel('$\sqrt{\epsilon}$','Interpreter','latex');
view(40,40);
%colorbar();
set(gca,'FontSize',20);
axis([0,1,0,0.015,0,1.5])

%% 
irow = 5;

ky = ky_list(irow);

fprintf('ky= %d\n', ky);

% sample Y
tY = sample_by_cdf_1d( tgrid, Fgrid, Ny);
dataY = map_to_RD_func(tY);

k_knn = ky*20; %Ny;
%tic,
[~, disXY] = knnsearch( dataY, dataX, 'k', k_knn);
%toc

% estimate hatrho by knn
m0_knn = 2;
hatR_knn = disXY(:,ky);
hatrho_knn = ((1/m0_knn*(ky/Ny))^(-1/dM))*hatR_knn;
hatp2 = hatrho_knn.^(-dM);

% hatp by kde
epsx = epsx_list(irow);
hatp = (epsx^(-dM/2)/m0_h)* (1/Ny*sum(h_func( disXY.^2/epsx),2));

%% plots at a value of k and eps
figure(1),clf;
hold on;
plot( tX, barrho, '-b');
plot(tX, hatrho_knn, 'xr');
grid on; 
title(sprintf('$N_y=%d$, $k_y=%d$',Ny,ky),'Interpreter','latex');
legend({'$\bar{\rho}$','$\hat{\rho}_{knn}$'}, 'Interpreter','latex');
set(gca,'FontSize',20);
xlabel('$t_X$','Interpreter','latex')

figure(2),clf;
hold on;
plot( tX,  abs(hatrho_knn-barrho)./barrho, 'x-b');
    %to see flat error when variance error dominates, i.e. when ky small
grid on; set(gca,'FontSize',20);
title('$|\hat{\rho}(x)-\bar{\rho}(x)|/\bar{\rho}(x)$', 'Interpreter', 'latex')
xlabel('$t_X$','Interpreter','latex')

%
figure(3),clf; hold on;
plot( tX, barp, '-b');
plot(tX, hatp , 'xr');
grid on; 
title(sprintf('$\\sqrt{\\epsilon}=%6.4f$',sqrt(epsx)),'Interpreter','latex');
legend( {'$p$','$\hat{p}$'}, 'Interpreter','latex','Location','northwest')
set(gca,'FontSize',20);
xlabel('$t_X$','Interpreter','latex')

figure(4),clf, hold on;
plot( tX, abs(hatp-barp)./barp, 'x-b');
grid on; 
set(gca,'FontSize',20);
title('$|\hat{p}(x)-p(x)|/p(x)$', 'Interpreter', 'latex')
drawnow();
xlabel('$t_X$','Interpreter','latex')

