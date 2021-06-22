% eigen embedding of mnist, over a range of sigma0

clear all; rng(2020);

%% load mnist 5000
load MNIST_X_te_all_32.mat

nperclassX=200;
nperclassY=100;

Nclass= 5;

dataX=zeros(nperclassX*Nclass,28^2);
labelsX=zeros(nperclassX*Nclass,1);

dataY=zeros(nperclassY*Nclass,28^2);

for iclass=1:Nclass
    id=(iclass-1)*nperclassX+1:iclass*nperclassX;
    
    xi = X_te{iclass};
    ni = size(xi,2);
    
    xi = xi(:, randperm( ni)); %shuttle
    
    x=xi(:,1:nperclassX)';
    x= reshape(x,nperclassX,32,32);
    x=x(:,3:30,3:30);
    dataX(id,:)=reshape(x,[nperclassX,28^2])/255;
    labelsX(id)=iclass;
    
    
    id=(iclass-1)*nperclassY+1:iclass*nperclassY;
    x=xi(:,nperclassX+1:nperclassX+nperclassY)';
    x= reshape(x,nperclassY,32,32);
    x=x(:,3:30,3:30);
    dataY(id,:) = reshape(x,[nperclassY,28^2])/255;
    
end

[Nx, dim] = size(dataX);
[Ny, ~] = size(dataY);

%% kde on X
kx = 7;

[nnInds, nnDist] = knnsearch( dataX, dataX, 'k', kx*20);

epsx_kde = median( nnDist(:,kx))^2; 

ah =4/pi;
h_func = @(xi) exp(-xi/ah);
m0_h = 2;

hatp_X =  mean(h_func( nnDist.^2/epsx_kde),2);
    
%% vis data by pca
maxk = 5;

x = dataX;
mx = mean(x,1);
x = bsxfun(@minus, x, mx);

[u,s,~] = svd(x);
x_pca = u(:,1:maxk);
s_pca = diag(s(1:maxk,1:maxk));


%%
figure(10),clf;
s=scatter(x_pca(:,1), x_pca(:,2),80, labelsX, 'o', 'filled')
alpha(s,0.25)
colormap(jet); colorbar();
grid on; axis equal; 
set(gca,'FontSize',20);
title('pca coloered by labels','Interpreter','latex')

figure(11),clf;
s=scatter(x_pca(:,1), x_pca(:,2),80, hatp_X, 'o', 'filled')
alpha(s,0.25)
colormap(jet); colorbar();
grid on; axis equal; 
set(gca,'FontSize',20);
title('pca localed by KDE density','Interpreter','latex')

%% diffusion map embedding
m2k = 1; m0k = 1;
k0_func = @(r) exp(-r/2);


%
sig0_list = 1/3*[1:0.1:1.5]';
nrow = numel(sig0_list);

%%
sig0DM_list = sig0_list+1/3*0.2;

psi5s = zeros( Nx, 3, nrow);


for irow =1:nrow
    sig0 = sig0DM_list(irow);
    
    sigma = median( nnDist(:,kx))*sig0;
    
    
    k_nn = size(nnDist, 2);
    rowInds = kron((1:Nx)', ones(k_nn,1));
    colInds = reshape(nnInds', k_nn*Nx, 1);
    vals    = reshape( exp(- nnDist.^2/(2*sigma^2))', k_nn*Nx, 1);
    K = sparse(rowInds, colInds, vals, Nx, Nx);
    K = (K+K')/2;
    
    dK = sum(K,2);
    
    % normalize by D, "alpha=1" in diffusion map
    K5 = K./((dK)*(dK)');
    dK5= sum(K5,2);
    
    
    %
    disp('eig: diffusion map')
    
    [v,d] = eigs(-K5+diag(dK5), diag(dK5), maxk, 'sr', 'SubspaceDimension', 50,...
        'MaxIterations', 300, 'Display', 1);
    
    [lam5, tmp]=sort(diag(d),'ascend');
    v5 = v(:,tmp);
    
    mu5= lam5(1:maxk);
    
    tmp =v5(:,1:maxk);
    tmp = bsxfun(@rdivide, tmp, sqrt(sum(tmp.^2,1))); %normalize to be 2norm 1
    
    psi5 = tmp*sqrt(Nx);

    psi5s(:,:,irow)=psi5(:,[2,3,4]);
    
    
end

%% self-tune embedding
psi1s = zeros( Nx, 3, nrow);

for irow= 1:nrow
    sig0 = sig0_list(irow);
    
    sigmaKvec = nnDist(:,kx)*sig0;
    
    
    rowInds = kron((1:Nx)', ones(k_nn,1));
    colInds = reshape(nnInds', k_nn*Nx, 1);
    vals    = reshape(nnDist', k_nn*Nx, 1);
    autotuneVals = sigmaKvec(rowInds).*sigmaKvec(colInds); %hatrhoi*hatrhoj
    
  

    Kvals = exp( -vals.^2 ./ (2*autotuneVals)  )./(autotuneVals);
    
    K = sparse(rowInds, colInds, Kvals, Nx, Nx);
    K = (K+K')/2;
    
    
    K1= K;
    dK1 = sum(K1,2);
    
    
    % L_rw
    B1 = diag(dK1.*sigmaKvec.^2);
    
    [v,d] = eigs(-K1+diag(dK1), B1, maxk, 'sr', 'SubspaceDimension', 100,...
        'MaxIterations', 300, 'Display', 1);
    
    
    [lam1, tmp]=sort(diag(d),'ascend');
    v1 = v(:,tmp);
    
    mu1= lam1(1:maxk);
    
    tmp =v1(:,1:maxk);
    tmp = bsxfun(@rdivide, tmp, sqrt(sum(tmp.^2,1))); %normalize to be 2norm 1
    psi1 = tmp*sqrt(Nx);
    
    psi1s(:,:,irow) = psi1(:,[2,3,4]);
   
end

%% 
psi2s = zeros( Nx, 3, nrow);


for irow= 1:nrow
    sig0 = sig0_list(irow);
    
    sigmaKvec = nnDist(:,kx)*sig0;
    
    
    rowInds = kron((1:Nx)', ones(k_nn,1));
    colInds = reshape(nnInds', k_nn*Nx, 1);
    vals    = reshape(nnDist', k_nn*Nx, 1);
    autotuneVals = sigmaKvec(rowInds).*sigmaKvec(colInds); %hatrhoi*hatrhoj
    
    hatpVals = hatp_X(rowInds).*hatp_X(colInds); 
        % by using W', normalizing by density, low-density area more spreaded
        % out in the embeddding
    
    Kvals = exp( -vals.^2 ./ (2*autotuneVals)  )./(autotuneVals.*sqrt(hatpVals));
  
    K = sparse(rowInds, colInds, Kvals, Nx, Nx);
    K = (K+K')/2;
    
    K2= K;
    dK2 = sum(K2,2);
    
    % L_rw
    B2 = diag(dK2.*sigmaKvec.^2);
    
    [v,d] = eigs(-K2+diag(dK2), B2, maxk, 'sr', 'SubspaceDimension', 100,...
        'MaxIterations', 300, 'Display', 1);
    
    
    [lam2, tmp]=sort(diag(d),'ascend');
    v2 = v(:,tmp);
    
    
    mu2= lam2(1:maxk);
     
    tmp =v2(:,1:maxk);
    tmp = bsxfun(@rdivide, tmp, sqrt(sum(tmp.^2,1))); %normalize to be 2norm 1
    psi2 = tmp*sqrt(Nx);
       
    psi2s(:,:,irow) = psi2(:,[2,3,4]);
     
end

%% plot the embeddings


%% diffusion map (DM) embedding 

vs = psi5s;
for irow = 2:nrow
    v_ref = vs(:,:,irow-1); %[Nx, 3]
    v= vs(:,:,irow);
    
    tmp = ( v\v_ref);
    vs(:,:,irow)=v* diag( sign( diag(tmp)));
        % align the embeddings before plotting
end

figure(20),clf;
for irow =1:nrow
    subplot(1,nrow,irow);
    
  
    x_DM = vs(:,:,irow);
    
    s=scatter3(x_DM(:,1), x_DM(:,2), x_DM(:,3), 80, labelsX, 'o', 'filled');
    alpha(s,0.5);
    colormap(jet); 
    grid on; 
    view(170,20);
    
    set(gca,'FontSize',20); axis equal
    title(sprintf('DM, $\\sigma_0=%6.4f$',sig0DM_list(irow)), ...
        'Interpreter','latex')
end

% investigate outlier points in the embedding
hatR = nnDist(:,kx);

irow = 3;
% vis embedding
x_DM = vs(:,:,irow);
j_out = find( sum(abs(x_DM),2) >5)

figure(21),clf; hold on;
s=scatter3(x_DM(:,1), x_DM(:,2), x_DM(:,3), 80, labelsX, 'o', 'filled');
alpha(s,0.5);
scatter3(x_DM(j_out,1), x_DM(j_out,2), x_DM(j_out,3), 200, 'or');
grid on; view(-40,20); colormap(jet);
title(sprintf('DM, $\\sigma_0=%6.4f$',sig0DM_list(irow)), ...
        'Interpreter','latex')
%colorbar(); 
set(gca,'FontSize',20); axis equal


 figure(23), clf; hold on;
s=scatter(1:Nx, hatR, 40, labelsX, 'o', 'filled');
alpha(s,0.5);
colormap(jet);
 plot( j_out, hatR(j_out),'or','MarkerSize', 15)
set(gca,'FontSize',20); grid on;
%set(gca,'XTickLabel','')
xlabel('index of sample','Interpreter','latex')
title(sprintf('distance to %d-th nearest neighbor',kx), 'Interpreter','latex');
 
%% self-tune embedding W^(1)

vs = psi1s;


for irow = nrow-1:-1:1
    v_ref = vs(:,:,irow+1); %[Nx, 3]
    v= vs(:,:,irow);
    tmp = ( v\v_ref);
    vs(:,:,irow)=v* diag( sign( diag(tmp)));
end

figure(30),clf;
for irow =1:nrow
    subplot(1,nrow,irow);
    
    % vis embedding
    x_ST = vs(:,:,irow);
    
    s=scatter3(x_ST(:,1), x_ST(:,2), x_ST(:,3), 80, labelsX, 'o', 'filled');
    alpha(s,0.5);
    colormap(jet); %colorbar();
    grid on; axis equal; view(170,20);
    set(gca,'FontSize',20); 
    title(sprintf('$W^{(1)}$, $\\sigma_0=%6.4f$',sig0_list(irow)), ...
        'Interpreter','latex')
end


%% embedding by W'

vs = psi2s;

v_ref = psi1s(:,:,nrow); %[Nx, 3]
v= vs(:,:,nrow);
tmp = ( v\v_ref);
vs(:,:,nrow)=v* diag( sign( diag(tmp)));

for irow = nrow-1:-1:1
    v_ref = vs(:,:,irow+1); %[Nx, 3]
    v= vs(:,:,irow);
    tmp = ( v\v_ref);
    vs(:,:,irow)=v* diag( sign( diag(tmp)));
end

figure(40),clf;
for irow =1:nrow
    subplot(1,nrow,irow);
    
    % vis embedding
    x_ST = vs(:,:,irow);
    
    s=scatter3(x_ST(:,1), x_ST(:,2), x_ST(:,3), 80, labelsX, 'o', 'filled');
    alpha(s,0.5);
    colormap(jet); %colorbar();
    grid on; axis equal;
    set(gca,'FontSize',20); view(170,20);
    title(sprintf('$W^{\\prime}$, $\\sigma_0=%6.4f$',sig0_list(irow)), ...
        'Interpreter','latex')
end

%%

return;
