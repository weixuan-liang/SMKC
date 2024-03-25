function [G_star,obj] = fusion_p_Kernel(GH,numclass)

    numker = size(GH,3);
    num = size(GH,1);
    anchor = size(GH,2);
    GH_low_rank = zeros(num, anchor, numker);

    %% Initialization
    for ker = 1:numker
        [u,d,v] = svds(GH(:,:,ker), numclass);
        GH_low_rank(:,:,ker) = u * d * v';
    end
    G_star = sum(GH_low_rank, 3) / numker;

   %% Iterations
    G_star_old = zeros(num, anchor);
    iter = 1;
    while norm(G_star_old - G_star, 'fro')/sqrt(num*anchor) > 1e-4
        for ker = 1:numker
           G_temp = (G_star + GH(:,:,ker)) / 2;
           [u,d,v] = svds(G_temp, numclass);
           GH_low_rank(:,:,ker) = u * d * v';
       end
       G_star_old = G_star;
       [u_star,d_star,v_star] = svds(sum(GH_low_rank, 3) / numker, numclass);
       G_star = u_star*d_star*v_star';
       obj_temp = 0;
       for ker = 1:numker
           obj_temp = obj_temp + 0.5 * norm(G_star - GH_low_rank(:,:,ker),'fro')^2 + 0.5 * norm(GH(:,:,ker) - GH_low_rank(:,:,ker),'fro')^2;
       end
       obj(iter) = obj_temp;
       iter = iter + 1;
    end



end