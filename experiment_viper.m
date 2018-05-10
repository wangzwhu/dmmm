clear all;
clc;
addpath('.\data');
addpath('.\helper');

%%%%% load data and PCA to low dimensions
load('VIPeR_feature_FTCNN.mat');
meanX = mean(feature_all, 1); 
features = (feature_all - repmat(meanX, size(feature_all,1), 1)); % Mean removal
for dnum = 1:size(feature_all, 1)
    features(dnum,:) = features(dnum, :)./norm(features(dnum, :), 2); % L2 norm normalization
end
clear feature_all;
clear dnum;
clear meanX;
[W, S, latent] = PCA(features, 70);
features = features*W;
features = features';
features = double(features);
clear latent;
clear S;
clear W;

%%%%% prepare for the training, testing and reference data, including
%%%%% Feature Vector and Discrepancy Matrix
TIMES = 3;
TOTAL_NUM = 632;
TRAIN_NUM = 200;
TEST_NUM  = 200;
REF_NUM   = 100;
result_total = zeros(3, TEST_NUM);
for roops = 1:TIMES
    result = zeros(3, TEST_NUM);

    perm = randperm(TOTAL_NUM);                         % training data
    idx_a = perm(1:TRAIN_NUM);                   
    idx_b = idx_a + TOTAL_NUM;
    xa   = features(:,idx_a);
    xb   = features(:,idx_b);   
    
    idx_a_test = perm(TRAIN_NUM+1:TRAIN_NUM+TEST_NUM);  % testing data
    idx_b_test = idx_a_test + TOTAL_NUM; 
    xa_test = features(:,idx_a_test);
    xb_test = features(:,idx_b_test);
    
    idx_a_reference = perm(TRAIN_NUM+TEST_NUM+1:TRAIN_NUM+TEST_NUM+REF_NUM);    % reference data
    idx_b_reference = idx_a_reference + TOTAL_NUM; 
    XR_A = features(:,idx_a_reference);
    XR_B = features(:,idx_b_reference);
 
    % construct Discrepancy Matrix, XB1 and XB2 stand for coresponding postive and negative samples 
    for i=1:TRAIN_NUM
        a = xa(:,i);
        b = xb(:,i);
        A = kron(a,ones(1,REF_NUM));
        XA(:,:,i) = A - XR_A;
        B = kron(b,ones(1,REF_NUM));
        XB(:,:,i) = B - XR_B;
    end
    perm = randperm(TRAIN_NUM);
    XB2 = XB(:, :, perm);
    for i=1:TEST_NUM
        a = xa_test(:,i);
        b = xb_test(:,i);
        A = kron(a,ones(1,REF_NUM));
        XA_test(:,:,i) = A - XR_A;
        B = kron(b,ones(1,REF_NUM));
        XB_test(:,:,i) = B - XR_B;
    end
    
    %%%% Test
    % L2 distance for feature vector
    dist_v = sqdist(xa_test, xb_test);
    result(1, :) = cmc(dist_v, TEST_NUM);
    
    % F2 distance for discrepancy matrix
    for i=1:TEST_NUM
        for j=1:TEST_NUM
            dist_n(i,j) = norm(XA_test(:,:,i)-XB_test(:,:,j), 'fro');
        end
    end
    result(2, :) = cmc(dist_n, TEST_NUM);
    
    % DMMM for discrepancy matrix
    [L1, L2, error, iter] = Learn_L1L2(XA, XB, XB2, 2, 0.5, 0.5);
    for i=1:TEST_NUM
        for j=1:TEST_NUM
            P =  L1*(XA_test(:,:,i)-XB_test(:,:,j))*L2;
            dist_l1_l2(i,j) = norm(P, 'fro'); 
        end
    end
    result(3, :) = cmc(dist_l1_l2, TEST_NUM);
    
    result_total = result_total + result;
end
result_total = result_total/TIMES;



%% Plot Cumulative Matching Characteristic (CMC) Curves
hold on; 

plot(result_total(1,:)/TEST_NUM,'LineWidth',2, ...
       'Color','g');
plot(result_total(2,:)/TEST_NUM,'LineWidth',2, ...
       'Color','b');
plot(result_total(3,:)/TEST_NUM,'LineWidth',2, ...
        'Color','r');

title('Cumulative Matching Characteristic (CMC) Curves - VIPeR dataset');
box('on');
set(gca,'XTick',[0 10 20 30 40 50 60 70 80 90 100]);
ylabel('Matches');
xlabel('Rank');
ylim([0 1]);
xlim([0 100]);
hold off;
grid on; 

legend('Feature Vector','Discrepancy Matrix','Discrepancy Matrix+Matrix Metric',3); 