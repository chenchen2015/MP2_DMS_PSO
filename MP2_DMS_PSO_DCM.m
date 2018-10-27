function [outIm,PSNR] = MP2_DMS_PSO_DCM(~)
% IMAGE SPARSE DECOMPOSITIONER
% Version : 2.10
% function: Use MP Algorithm Based on DMS-PSO Altorithm - Bird-oid Model Demo
% by Chen Chen in Jan 13, 2013
% Last Update : Jan 16, 2013
%               Mar 23, 2013   - Inertia Boundary Sterategy's invited
%               Mar 24, 2013   - Discrete Parameter Mutation operator's invited
%                              - Several bugs has been fixed
%               Mar 27, 2013   - Add a Strategy Select to choose DMS-PSO or Std-PSO 


% Paper: C. Chen, J.J. Liang, B.Y. Qu, and B. Niu, "Using Dynamic Multi-Swarm 
%        Particle Swarm Optimizer to Improve the Image Sparse Decomposition 
%        Based on Matching Pursuit," ICIC 2013, LNAI 7996, pp. 587-595, 2013.
% 
% 
% Copy Right - Chen Chen
% 6/5/2015
% Email: chenchen.bme@gmail.com
% LinkedIn: https://www.linkedin.com/in/chen90


%% Workspace Initialization
home
% close all
SHOW_FIGURE = 1;

%% Image Acquisition
filename = uigetfile(...
    {'*.jpg;*.bmp;*.png;*.jpeg;*.tif;*.gif',...
    'All Supported Image File(*.jpg,*.bmp,*.png,*.jpeg,*.tif,*.gif)'},...
    'Pick a Image file to open ...');
% filename = 'lena64.bmp';
try
    Im = imread(filename);
catch msg
    switch msg.identifier
        case 'MATLAB:imagesci:imread:badImageSourceDatatype'
            errordlg('You MUST Pick a file to open !!!', 'Error');
        case 'MATLAB:imagesci:imread:fileDoesNotExist'
            errordlg('Image not EXIST or not in the Current Workspace !!!', 'Error');
        otherwise
            disp(['Error : Unknown Error Message Identifier - ' msg.identifier]);
    end
    disp('Loading default image: Lena64 as a demo...');
    filename = 'lena64.bmp';
    Im = imread(filename);
    return
end
%% Start Global Time Counting
tstart = tic;
%% Determine the Decomposition Parameters
% Matching Pursuit Processing iterative number
iterative_number = 200;
% Convert the 2-D Image[H,L] to 1-D[1,H*L] 
if length(size(Im)) > 2
    Im = rgb2gray(Im(:, :, 1:3));
end
Im = double(Im);
[H,L] = size(Im);
N = H * L;
Im1D = reshape(Im,1,N);
% signal_reconstruct: the Signal reconstructed by the sparse parameters
% signal_reconstruct = zeros(1,N);
% signal_r: the residual signal, that is, R(i)f
% signal_r = Im1D;
signal = Im1D;
PSNR = zeros(1,iterative_number);
%% Gabor Atom Parameters
%%%%%%%%%%%%% CONST %%%%%%%%%%%%%%%
NN = 5;
theta_min = 1;
stx_min = 0;
sty_min = 0;
%%%%%%%%%%% VARIATIONS %%%%%%%%%%%%% 
theta_max = max(H,L);
stx_max = NN * log2(H) - NN;
sty_max = NN * log2(L) - NN;
%% Wipe off the DC Vector
signal_reconstruct = (1/N) * sum(signal);
signal_r = signal - signal_reconstruct;

%% Initialize plots
if SHOW_FIGURE
    figure(1); clf;
    subplot(221);
        imshow(Im, []);
        title('Input Image');
    subplot(222);
        h_gabor = imshow(Im, []);
        title('Current Gabor atom used');
    ax_perf = subplot(223);
        h_perf = plot(0, 'LineWidth', 4);
        title('Performance');
        xlabel('Number of Gabor atoms used');
        ylabel('Reconstruction accuracy in PSNR [dB]');
        grid on;
    subplot(224);
        h_im = imshow(Im, []);
        title('Reconstructed Signal');
end

%% The MP Process
for n = 1:iterative_number
    tic
    % The following program uses one subroutine to select the best atom
    [proj,g] = PSO_Select(signal_r,H,L,theta_min,theta_max,stx_min,...
        stx_max,sty_min,sty_max,n,iterative_number);
    % Reconstrcut the best atom from the parameters gotted by the above
    % subroutine
    % Reducing the best atom part from the residual signal and adding it to
    % the reconstructed signal
    % g is the best atom
    atom = proj * g;
    signal_reconstruct = signal_reconstruct + atom;
    signal_r = signal_r - atom;
    PSNR(n) = psnr(signal, signal_reconstruct);
    % at each step of MP, we display the figures of the orignal signal, the
    % best atom selected, the residual signal and the reconstructed signal
    % at window 1,2,3,4, respectively
    if SHOW_FIGURE
        plot_g = atom - min(atom);
        h_gabor.CData = reshape(plot_g, H, L) * 2;
        h_perf.YData = PSNR(1:n);
        ax_perf.XLim = [1 n+2];
        ax_perf.YLim = [min(PSNR(1:n))-1, max(PSNR(1:n))+3];
        h_im.CData = reshape(signal_reconstruct, H, L);
        drawnow;   
    end
    % n is the MP process number, or the number of atoms selected
    fprintf('Atoms : %3d , PSNR : %f dB\n',n,PSNR(n));
%     disp(['Number of Atoms Selected : ' num2str(n)]);
%     disp(['PSNR : ' num2str(PSNR(n)) ' dB ']);
    toc
end
%% End Global Time Counting
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
telapsed = toc(tstart);
disp(['Total Time Cost : ' num2str(telapsed) ' seconds.']);
outIm = reshape(signal_reconstruct,H,L);
% imwrite(uint8(reshape(signal_reconstruct,H,L)),'lena1.bmp')
end
%% Asymmetrical Atom Geneator 1-D
% function g = Asym_Atom_1D(H,L,theta,stx,sty,u,v)
% [hy,hx] = meshgrid(0:H-1,0:L-1);
% g = Asym_Single(hx,hy,H,L,theta,stx,sty,u,v);
% % g = reshape(g',1,H*L);
% end
%% Asymmetrical Atom Geneator 2-D
function gr = Asym_Atom_GA(H,L,Birds,Num_Birds)
[hy,hx] = meshgrid(0:H-1,0:L-1);
gr = zeros(Num_Birds,H*L);
theta = Birds(:,1);
stx = Birds(:,2);
sty = Birds(:,3);
u = Birds(:,4);
v = Birds(:,5);
for i = 1:Num_Birds
    g = Asym_Single(hx,hy,H,L,theta(i),stx(i),sty(i),u(i),v(i));
    gr(i,:) = reshape(g',1,H*L);
end
end
%% Single Asymmetrical Atom Geneator
function g = Asym_Single(x0,y0,H,L,theta,stx,sty,u,v)
% Generate one atom using the following parameters 
%    theta : Rotation angle
%    sx   : scale in x direction
%    sy   : scale in y direction
% the center of atom is (H/2,L/2)
% Normalization process is included in this program
% Translation
hx = x0 - u;
hy = y0 - v;
% Rotation
theta_real = theta / max(H,L) * 2 .* pi;
rotate_sin = sin(theta_real);
rotate_cos = cos(theta_real);
% Scale
NN = 5;
sx = 2^(stx/NN);
sy = 2^(sty/NN);
% the all operation
x = (rotate_cos * hx + rotate_sin * hy) / sx;
y = (rotate_cos * hy - rotate_sin * hx) / sy;
% Generate the Atom
g = (4 .* x .* x - 2) .* exp(-x .* x - y .* y);
g_energy = sum(sum(g .* g));
% Normalizationd
g = g / sqrt(g_energy);
end
%% PSNR Calculator
function PSNR = psnr(f1, f2)
% Calculate PSNR of the input image
k = 8;
% k denote for the bits of the image
fmax = 2.^k - 1;
a = fmax .^ 2;
e = double(f1) - double(f2);
[m, n] = size(e);
b = sum(sum(e.^2));
PSNR = 10 * log10(m * n * a / b);
end
%% PSO Atom Selector
function [proj,gr] = PSO_Select(signal_r,H,L,theta_min,theta_max,stx_min,stx_max,sty_min,sty_max,CurAtom,MaxAtom)
% PSO Altorithm - Bird-oid Model Demo
% Designed and Modified By Cliff Chen in Jan 8, 2013
% Update in Jan 11, 2013 : Replace the Fixed Weight by Linearly Decreasing Weight, LDW
% Update in Jan 13, 2013 : Use Dynamic Multi-Swarm Algorithm to further improve the PSO's performance 
% Update in Jan 16, 2013 : A Fatal Error and several bugs have been fixed  

%% Strategy Select
DMS = 1; % DMS Strategy Switch
         % |0 - OFF - Using Std-PSO
         % |1 - ON  - Using DMS-PSO
DCM = 1; % Discrete Coefficient Mutatuion Strategy Switch
         % |0 - OFF - Using Std-PSO
         % |1 - ON  - Using DCM
LDDR = 1;% Linear Decreasing DMS Ratio Strategy Switch
         % |0 - OFF - Using Std-PSO
         % |1 - ON  - Using LDDR
ScaleFactorDPM = 1;
MaxRatioDMS = 0.6;
MinRatioDMS = 0.1;
FixRatioDMS = 0.6;
%% PSO Parameters
Dim = 5;
% 测试不同Group_Num影响
Group_Num = 7;
Group_Bird_Num = 5;
Regroup_Era = 4;
Num_Birds = Group_Num * Group_Bird_Num;
Num_Birds = Num_Birds * DMS + (1 - DMS) * 30;
% Strategy Select
Group_Num = DMS * Group_Num + (1 - DMS);
Group_Bird_Num = DMS * Group_Bird_Num + (1 - DMS) * Num_Birds;
% 测试不同MAX-FES影响
MAX_FES = 6001;
GroupID = zeros(Group_Num,Group_Bird_Num);
Posi_Group = zeros(1,Num_Birds);
Para_Scale = [theta_min stx_min sty_min 0 0;...
              theta_max stx_max sty_max H L];
Scale = Para_Scale(2,:) - Para_Scale(1,:);
Scale_H = Para_Scale(2,:);
Scale_L = Para_Scale(1,:);
Bound_Scale = 0.1 * Scale;
% Accelerator = [1.49445 1.49445];
% Weight = 0.729;
Accelerator = [2 2];
Weight_Ini = 0.90;
Weight_End = 0.20;
V_MAX = 0.5 * Scale;
Birds = round( rand(Num_Birds,Dim) .* repmat(Scale,Num_Birds,1) + repmat(Scale_L,Num_Birds,1) );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
V_Birds = -repmat(V_MAX,Num_Birds,1) + 2 * rand(Num_Birds,Dim) .* repmat(V_MAX,Num_Birds,1);
gBestPosi = zeros(Group_Num,5);
gBest_res = zeros(Group_Num,1);
gBest_proj = zeros(Group_Num,1);
%% Initialize the Bird-oid
g = Asym_Atom_GA(H,L,Birds,Num_Birds);
Proj_Temp = signal_r * g';
Res_Temp  = abs( Proj_Temp );
pBest_res = Res_Temp;
pBest_proj = Proj_Temp;
pBestPosi = Birds;
for i = 1:Group_Num
    GroupID(i,:) = ( (i-1) * Group_Bird_Num + 1 ):(i * Group_Bird_Num);
    Posi_Group( GroupID(i,:) ) = i; 
    [gBest_res(i),gBestID] = max( pBest_res( GroupID(i,:) ) );
    gBest_proj = pBest_proj( GroupID(i,gBestID) );
    gBestPosi(i,:) = pBestPosi(GroupID(i,gBestID),:);%initialize the gbest and the gbest's fitness value
end
%% Loop for Dynamic Multi-Swarm
Era = 0;
FitCount = 0;
DMS_FES = DMS * ( LDDR * round( (MaxRatioDMS - CurAtom / MaxAtom * (MaxRatioDMS - MinRatioDMS)) * MAX_FES ) + (1 - LDDR) * round(FixRatioDMS * MAX_FES) );
while FitCount < DMS_FES
    Era = Era + 1;
    Weight = (Weight_Ini - Weight_End) * (MAX_FES - FitCount) / MAX_FES + Weight_End;
    for j = 1:Num_Birds
        V_Birds(j,:) = Weight * V_Birds(j,:) + Accelerator(1) * rand(1,Dim) .* (pBestPosi(j,:) - Birds(j,:)) + Accelerator(2) * rand(1,Dim) .* (gBestPosi(Posi_Group(j),:) - Birds(j,:));
%         V_Birds(j,:) = (V_Birds(j,:) == 0) * ( (rand(1,Dim) - 0.5) .* Bound_Scale ) + (V_Birds(j,:) ~= 0) * V_Birds(j,:);
        V_Birds(j,:) = (V_Birds(j,:) > V_MAX) .* V_MAX + (V_Birds(j,:) <= V_MAX) .* V_Birds(j,:);
        V_Birds(j,:) = (V_Birds(j,:) < (-V_MAX)) .* (-V_MAX) + (V_Birds(j,:) >= (-V_MAX)) .* V_Birds(j,:);     
        % Discrete Parameter Mutation Operator
        Temp_Birds = Birds(j,:);
        Birds(j,:) =  Birds(j,:) + V_Birds(j,:);
        Disturb_Flag = DCM .* sum( (round(Birds(j,:)) - round(Temp_Birds) ~= 0),2) == 0;
        Birds(j,:) = Disturb_Flag * ( (rand(1,Dim) - 0.5) .* Bound_Scale * ScaleFactorDPM + Birds(j,:) ) + ( 1 - Disturb_Flag ) * Birds(j,:);
        % 超出边界 : 2.在边界附近邻域内随机初始化
        tmp_Scale_H = Scale_H - round( rand(1,Dim) .* Bound_Scale);
        tmp_Scale_L = Scale_L + round( rand(1,Dim) .* Bound_Scale);
        % Strategy #3 Apply inertial boundary only for theta
%         Birds(j,1) = (Birds(j,1) > theta_max) * (Birds(j,1) - theta_max) + (Birds(j,1) <= theta_max) * Birds(j,1);
%         Birds(j,1) = (Birds(j,1) < theta_min) * (Birds(j,1) + theta_max) + (Birds(j,1) >= theta_min) * Birds(j,1);
%         Birds(j,1) = (Birds(j,1) > theta_max) * theta_max + (Birds(j,1) <= theta_max) * Birds(j,1);
%         Birds(j,1) = (Birds(j,1) < theta_min) * theta_min + (Birds(j,1) >= theta_min) * Birds(j,1);        
%         Birds(j,2:end) = (Birds(j,2:end) > Scale_H(2:end)) .* tmp_Scale_H(2:end) + (Birds(j,2:end) <= Scale_H(2:end)) .* Birds(j,2:end);
%         Birds(j,2:end) = (Birds(j,2:end) < Scale_L(2:end)) .* tmp_Scale_L(2:end) + (Birds(j,2:end) >= Scale_L(2:end)) .* Birds(j,2:end);
        Birds(j,:) = (Birds(j,:) > Scale_H) .* tmp_Scale_H + (Birds(j,:) <= Scale_H) .* Birds(j,:);
        Birds(j,:) = (Birds(j,:) < Scale_L) .* tmp_Scale_L + (Birds(j,:) >= Scale_L) .* Birds(j,:);
        % Cross Operator
        Cross = rand(1,Dim) < 0.35;
%         Birds(j,:) = round( Cross .* pBestPosi(j,:) + (1 - Cross) .* Birds(j,:) );
        Birds(j,:) = ceil( Cross .* pBestPosi(j,:) + (1 - Cross) .* Birds(j,:) );
        FitCount = FitCount + 1;
        g = Asym_Atom_GA(H,L,Birds(j,:),1);
        Proj_Temp = signal_r * g';
        Res_Temp  = abs( Proj_Temp );
        if pBest_res(j) < Res_Temp
            % update pBest
            pBest_res(j) = Res_Temp;
            pBest_proj(j) = Proj_Temp;
            pBestPosi(j,:) = Birds(j,:);
        end
        if gBest_res(Posi_Group(j)) < pBest_res(j)
            % update gBest
            gBest_res(Posi_Group(j)) = pBest_res(j);
            gBest_proj(Posi_Group(j)) = pBest_proj(j);
            gBestPosi(Posi_Group(j),:) = pBestPosi(j,:);
        end
    end
    if mod(Era,Regroup_Era) == 0
        RegroupID = randperm(Num_Birds);
        for k = 1:Group_Num
            GroupID(k,:) = RegroupID( ((k-1) * Group_Bird_Num + 1) : k*Group_Bird_Num );
            Posi_Group( GroupID(k,:) ) = k; 
            [gBest_res(k),gBestID] = max( pBest_res( GroupID(k,:) ) );
            gBest_proj = pBest_proj( GroupID(k,gBestID) );
            % Update the gbest and the gbest's fitness value
            gBestPosi(k,:) = pBestPosi(GroupID(k,gBestID),:);
        end
    end
end
%% Loop for Standard Single Swarm
[gBest_res,tempID] = sort(gBest_res);
gBest_res = gBest_res(end);
gBestPosi = gBestPosi(tempID(end),:);
while FitCount < MAX_FES
    Era = Era + 1;
    Weight = (Weight_Ini - Weight_End) * (MAX_FES - FitCount) / MAX_FES + Weight_End;
    V_Birds = Weight * V_Birds + Accelerator(1) * rand(Num_Birds,Dim) .* (pBestPosi - Birds) + Accelerator(2) * rand(Num_Birds,Dim) .* (repmat(gBestPosi,Num_Birds,1) - Birds);
%     V_Birds = (V_Birds == 0) * ( (rand(Num_Birds,Dim) - 0.5) .* Bound_Scale(ones(Num_Birds,1),:) ) + (V_Birds ~= 0) * V_Birds;
    V_Birds = (V_Birds > V_MAX(ones(Num_Birds,1),:)) .* repmat(V_MAX,Num_Birds,1) + (V_Birds <= V_MAX(ones(Num_Birds,1),:)) .* V_Birds;
    V_Birds = (V_Birds < (-V_MAX(ones(Num_Birds,1),:))) .* repmat(-V_MAX,Num_Birds,1) + (V_Birds >= (-V_MAX(ones(Num_Birds,1),:))) .* V_Birds;
    % Discrete Parameter Mutation Operator
    Temp_Birds = Birds;
%     Birds = round(Birds + V_Birds);
    Birds = ceil(Birds + V_Birds);
    Disturb_Flag = DCM .* sum( (Birds - round(Temp_Birds) ~= 0),2) == 0;
    Birds = repmat(Disturb_Flag,1,Dim) .* round( repmat(Disturb_Flag,1,Dim) .* (rand(Num_Birds,Dim) - 0.5) .* Bound_Scale(ones(Num_Birds,1),:) * ScaleFactorDPM + Birds ) + ( 1 - repmat(Disturb_Flag,1,Dim) ) .* Birds;
    % 超出边界 : 2.在边界附近邻域内随机初始化
    tmp_Scale_H = repmat(Scale_H,Num_Birds,1) - round( rand(Num_Birds,Dim) .* repmat(Bound_Scale,Num_Birds,1) );
    tmp_Scale_L = repmat(Scale_L,Num_Birds,1) + round( rand(Num_Birds,Dim) .* repmat(Bound_Scale,Num_Birds,1) );
    % Strategy #3 Apply inertial boundary only for theta
%     Birds(:,1) = ( Birds(:,1) > theta_max(ones(Num_Birds,1)) ) .* ( Birds(:,1) - theta_max(ones(Num_Birds,1)) ) + ( Birds(:,1) <= theta_max(ones(Num_Birds,1)) ) .* Birds(:,1);
%     Birds(:,1) = ( Birds(:,1) < theta_min(ones(Num_Birds,1)) ) .* ( Birds(:,1) + theta_max(ones(Num_Birds,1)) ) + ( Birds(:,1) >= theta_min(ones(Num_Birds,1)) ) .* Birds(:,1);
%     Birds(:,1) = ( Birds(:,1) > theta_max(ones(Num_Birds,1)) ) .* theta_max(ones(Num_Birds,1)) + ( Birds(:,1) <= theta_max(ones(Num_Birds,1)) ) .* Birds(:,1);
%     Birds(:,1) = ( Birds(:,1) < theta_min(ones(Num_Birds,1)) ) .* theta_min(ones(Num_Birds,1)) + ( Birds(:,1) >= theta_min(ones(Num_Birds,1)) ) .* Birds(:,1);
%     Birds(:,2:end) = ( Birds(:,2:end) > Scale_H(ones(Num_Birds,1),2:end) ) .* tmp_Scale_H(:,2:end) + ( Birds(:,2:end) <= Scale_H(ones(Num_Birds,1),2:end) ) .* Birds(:,2:end);
%     Birds(:,2:end) = ( Birds(:,2:end) < Scale_L(ones(Num_Birds,1),2:end) ) .* tmp_Scale_L(:,2:end) + ( Birds(:,2:end) >= Scale_L(ones(Num_Birds,1),2:end) ) .* Birds(:,2:end);
    Birds = ( Birds > Scale_H(ones(Num_Birds,1),:) ) .* tmp_Scale_H + ( Birds <= Scale_H(ones(Num_Birds,1),:) ) .* Birds;
    Birds = ( Birds < Scale_L(ones(Num_Birds,1),:) ) .* tmp_Scale_L + ( Birds >= Scale_L(ones(Num_Birds,1),:) ) .* Birds;
    FitCount = FitCount + Num_Birds;
    g = Asym_Atom_GA(H,L,Birds,Num_Birds);
    Proj_Temp = signal_r * g';
    Res_Temp  = abs( Proj_Temp );
    temp = find(pBest_res < Res_Temp);
    pBest_res(temp) = Res_Temp(temp);
    pBest_proj(temp) = Proj_Temp(temp);
    pBestPosi(temp,:) = Birds(temp,:);
    [gBest_res,Sub] = max(pBest_res);
    gBestPosi = pBestPosi(Sub,:);
    gBest_proj = pBest_proj(Sub);
%     fprintf('PSO - Era : %3d ; gBest %f\n',Era,gBest_res);
end
proj = gBest_proj;
gr = Asym_Atom_GA(H,L,gBestPosi,1);
fprintf('Era : %3d ; gBest %f\n',Era,gBest_res);
end
