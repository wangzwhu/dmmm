function [L1, L2, error, iter] = Learn_L1L2(XA, XB, XB2, beta, mu1, mu2)

    sparse_weight = 15;

    Nf = size(XA, 1);
    Nr = size(XA, 2);
    L1 = eye(Nf, Nf); 
    L2 = eye(Nr, Nr); 
    L1_last = L1;
    L2_last = L2;
    L1_new = L1_last;
    L2_new = L2_last;    
    
    tol = 0.000000001;
    lamda1 =  0.1;
    lamda2 =  0.1;
    MaxIter = 2000;
    
    d = ones(Nr,1);
    D = spdiags(d,0,Nr,Nr);
    
    Error_last = SamplesLoss(XA,XB,XB2,L1,L2,beta,mu1,mu2,sparse_weight);  
    for iter = 1:MaxIter

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % fix L2, calculate L1     
        gradient_L1 = mu1*gradient_l1_ec(XA,XB,L1_last,L2) + mu2*gradient_l1_ed(XA,XB,XB2,L1_last,L2,beta);            
        findLamda = -1;
        for inner_iter= 1:20
            L1_new = L1_last - lamda1*gradient_L1;
            Error_new = SamplesLoss(XA,XB,XB2,L1_new,L2,beta,mu1,mu2,sparse_weight);    
            fprintf('find lambda1, iter:%d,  lambda1:%f,  error_New:%f,   error_Last:%f\n',inner_iter,lamda1,Error_new,Error_last);
            suff = (Error_new<Error_last);
            if inner_iter ==1
                decr_lamda1 = ~suff; L1_pre = L1_last; Error_pre = Error_last;
            end
            if decr_lamda1
                if suff
                   findLamda = 1;break; 
                else
                    lamda1 = lamda1/2;
                end
            else
                findLamda = 1;
                if ~suff
                    Error_new = Error_pre;
                    L1_new = L1_pre;
                    break;
                else
                    lamda1 = lamda1 * 1.1;
                    L1_pre = L1_new;
                    Error_pre = Error_new;
                end
            end
        end
        if findLamda >0 
            L1_last = L1_new;
        end

        L1 = L1_last;
        L2 = L2_last;
        error = Error_last;
        if (Error_new>=Error_last) 
            disp(sprintf('No better error, error:%f', Error_new));
            break;
        elseif((Error_last -Error_new)<tol)
            disp(sprintf('error no change. last_error:%f, error_tmp:%f',Error_last,Error_new));
            break;
        else
             Error_last = Error_new;
        end
    
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % fix L1, calculate L2             
        gradient_L2 = mu1*gradient_l2_ec(XA,XB,L1,L2_last) + mu2*gradient_l2_ed(XA,XB,XB2,L1,L2_last,beta);
        [D, g_sparse_l2] = sparse_L2(L2_last, D, Nr);
        gradient_L2 = gradient_L2 + sparse_weight*g_sparse_l2;
        findLamda = -1;
        for inner_iter= 1:20
            L2_new = L2_last - lamda2*gradient_L2;                
            Error_new = SamplesLoss(XA,XB,XB2,L1,L2_new,beta,mu1,mu2,sparse_weight); 
            fprintf('find lambda2, iter:%d,  lambda2:%f,  error_New:%f,   error_Last:%f\n',inner_iter,lamda2,Error_new,Error_last);
            suff = (Error_new<Error_last);
            if inner_iter ==1
                decr_lamda2 = ~suff; L2_pre = L2_last; Error_pre = Error_last;
            end
            if decr_lamda2
                if suff
                   findLamda = 1;break;
                else
                    lamda2 = lamda2/2;
                end
            else
                findLamda = 1;
                if ~suff
                    Error_new = Error_pre;
                    L2_new = L2_pre;
                    break;
                else
                    lamda2 = lamda2 * 1.1;
                    L2_pre = L2_new;
                    Error_pre = Error_new;
                end
            end
        end
        if findLamda >0 
            L2_last = L2_new;
        end

        L1 = L1_last;
        L2 = L2_last;
        error = Error_last;
        if (Error_new>=Error_last) 
            disp(sprintf('No better error, error:%f', Error_new));
            break;
        elseif((Error_last -Error_new)<tol)
            disp(sprintf('error no change. last_error:%f, error_tmp:%f',Error_last,Error_new));
            break;
        else
             Error_last = Error_new;
        end
        
 
    end
    
end

function g_l1_ec = gradient_l1_ec(XA,XB,L1,L2)
    tic;
	num = size(XA, 3);
    g_l1_ec = 0;
	for i=1:num
        Zi = XA(:,:,i) - XB(:,:,i);
        g_i = L1*Zi*L2*(L2')*(Zi');
        g_l1_ec = g_l1_ec + g_i;
    end
    g_l1_ec = 2*g_l1_ec/num;    
    disp(sprintf('gradient_l1_ec:%f', toc));
end

function g_l2_ec = gradient_l2_ec(XA,XB,L1,L2)
    tic;
	num = size(XA, 3);
    g_l2_ec = 0;
	for i=1:num
        Zi = XA(:,:,i) - XB(:,:,i);
        g_i = (Zi')*(L1')*L1*Zi*L2;
        g_l2_ec = g_l2_ec + g_i;
    end
    g_l2_ec = 2*g_l2_ec/num;       
    disp(sprintf('gradient_l2_ec:%f', toc));
end

function g_l1_ed = gradient_l1_ed(XA,XB,XB2,L1,L2,beta)
    tic;
    num = size(XB2, 3);
    g_l1_ed = 0;
    for i=1:num
        Ui = XA(:,:,i) - XB(:,:,i);
        Vi = XA(:,:,i) - XB2(:,:,i);
        e  = l1l2_distance(XA(:,:,i),XB(:,:,i),L1,L2) - l1l2_distance(XA(:,:,i),XB2(:,:,i),L1,L2);
        weigh_i = gradient_logistic_fun(e,beta);
        g_i = L1*Ui*L2*(L2')*(Ui')-L1*Vi*L2*(L2')*(Vi');
        g_l1_ed = g_l1_ed + weigh_i*g_i;
    end
    g_l1_ed = 2*g_l1_ed/num;
    disp(sprintf('gradient_l1_ed:%f', toc));
end

function g_l2_ed = gradient_l2_ed(XA,XB,XB2,L1,L2,beta)
    tic;
    num = size(XB2, 3);
    g_l2_ed = 0;
    for i=1:num   
        Ui = XA(:,:,i) - XB(:,:,i);
        Vi = XA(:,:,i) - XB2(:,:,i);
        e  = l1l2_distance(XA(:,:,i),XB(:,:,i),L1,L2) - l1l2_distance(XA(:,:,i),XB2(:,:,i),L1,L2);
        weigh_i = gradient_logistic_fun(e,beta);
        g_i = (Ui')*(L1')*L1*Ui*L2-(Vi')*(L1')*L1*Vi*L2;
        g_l2_ed = g_l2_ed + weigh_i*g_i;
    end
    g_l2_ed = 2*g_l2_ed/num;
    
    disp(sprintf('gradient_l2_ed:%f', toc));
end

function Error = SamplesLoss(XA,XB,XB2,L1,L2,beta,mu1,mu2,sparse_weight)
    tic;
    num = size(XB2, 3);
    Error = 0.0;
    for i=1:num
        XAi = XA(:,:,i);
        XBi = XB(:,:,i);
        XBj = XB2(:,:,i);
        ec_i = l1l2_distance(XAi,XBi,L1,L2);
        ed_i = l1l2_distance(XAi,XBi,L1,L2) - l1l2_distance(XAi,XBj,L1,L2); 
        ed_i = log(1+exp(beta*ed_i))/beta;  
        e_i = mu1*ec_i + mu2*ed_i;  
        Error = Error + e_i;
    end
    d = sqrt(sum(L2.*L2,2));
    d = sum(d);
    Error = Error/num + sparse_weight*d;
    
    disp(sprintf('SamplesLoss:%f', toc));
end

function d = l1l2_distance(XAi,XBj,L1,L2)
    P =  L1*(XAi-XBj)*L2;
    d = norm(P, 'fro'); 
end

function weight = gradient_logistic_fun(errorIn,beta)
    weight = 1./(1+exp((-beta).*errorIn));
end


function [D, g_sparse_l2] = sparse_L2(L2_last, D_last, Nr)
    g_sparse_l2 = 2*D_last*L2_last;
    d = sqrt(sum(L2_last.*L2_last,2));
    D = spdiags(d,0,Nr,Nr);
end






