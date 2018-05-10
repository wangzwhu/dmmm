       
        function s = LearnKISSME(X,idxa,idxb,matches)
           p = struct();     
           p.lambda =  1;
           p.pmetric = 1;
           obj.p  = p;

            idxa = double(idxa);
            idxb = double(idxb);
            
            %--------------------------------------------------------------
            %   KISS Metric Learning CORE ALGORITHM
            %
            tic;
            % Eqn. (12) - sum of outer products of pairwise differences (similar pairs)
            % normalized by the number of similar pairs.
            covMatches    = SOPD(X,idxa(matches),idxb(matches)) / sum(matches);
            % Eqn. (13) - sum of outer products of pairwise differences (dissimilar pairs)
            % normalized by the number of dissimilar pairs.
            covNonMatches = SOPD(X,idxa(~matches),idxb(~matches)) / sum(~matches);
            t = toc;
            
            tic;
            % Eqn. (15-16)
            s.M = inv(covMatches) - obj.p.lambda * inv(covNonMatches);   
            if obj.p.pmetric
                % to induce a valid pseudo metric we enforce that  M is p.s.d.
                % by clipping the spectrum
                s.M = validateCovMatrix(s.M);
            end
            s.t = toc + t;   
            
            %
            %   END KISS Metric Learning CORE ALGORITHM
            %--------------------------------------------------------------
            
            s.learnAlgo = obj;
        end
        

        
        function d = dist(obj, s, X, idxa,idxb) 
            d = cdistM(s.M,X,idxa,idxb); 
        end


