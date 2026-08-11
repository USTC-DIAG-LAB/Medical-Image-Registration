function main(in_path, out_path, mode, varargin)
    % Fast modified version:
    % 1) Compress A and B with KCenter+.
    % 2) Run one coreset registration pass only.
    % 3) Skip repeated candidate selection and trimmed refinement for speed.
    %
    % Usage:
    %   Auto mode:
    %       main(in_path, out_path, 'auto')
    %       Auto uses tau=0.05, ratio=0.05.
    %
    %   Manual mode:
    %       main(in_path, out_path, 'manual', tau, ratio)
    %
    %   Backward-compatible manual mode:
    %       main(in_path, out_path, 'manual', ratio)
    %       This uses tau=0.05.

    S = load(in_path);
    A = double(S.A);
    B = double(S.B);

    [dimA, NA] = size(A);
    [dimB, NB] = size(B);
    if dimA ~= 3 || dimB ~= 3
        error('A and B must be 3xN point clouds. Got A=%s, B=%s.', mat2str(size(A)), mat2str(size(B)));
    end

    if nargin < 3 || isempty(mode)
        mode = 'auto';
    end

    default_tau = 0.05;
    default_ratio = 0.05;

    if strcmpi(mode, 'auto')
        tau = default_tau;
        ratio = default_ratio;
        fprintf('Mode: Auto | tau=%.4f | ratio=%.4f\n', tau, ratio);
    else
        if numel(varargin) >= 2
            tau = double(varargin{1});
            ratio = double(varargin{2});
        elseif numel(varargin) == 1
            tau = default_tau;
            ratio = double(varargin{1});
        else
            error('Manual mode requires either (tau, ratio) or the old single ratio argument.');
        end
        fprintf('Mode: Manual | tau=%.4f | ratio=%.4f\n', tau, ratio);
    end

    if ~(isfinite(tau) && tau > 0)
        error('tau must be a positive finite scalar.');
    end
    if ~(isfinite(ratio) && ratio > 0 && ratio <= 1)
        error('ratio must be in (0, 1].');
    end

    k_min = 50;
    n_repeats = 1;
    refine_iter = 0;
    trim_keep_ratio = NaN;
    score_sample_n = NA;
    base_seed = randi(2^31 - 1);
    seeds = base_seed;

    WA = double(ones(1, NA) / NA);
    WB = double(ones(1, NB) / NB);
    kA = max(k_min, round(NA * ratio));
    kB = max(k_min, round(NB * ratio));

    total_tic = tic;
    MdlB = build_nn_model(B);
    coreset_method = 'KCenterPlus';

    fprintf('   Fast Fusion Coreset (%s): kA=%d, kB=%d, tau=%.4f | single pass, no refine\n', ...
        coreset_method, kA, kB, tau);

    rng(base_seed, 'twister');
    run_tic = tic;
    [R, t, EMD, FM, CSA, CSB] = run_one_coreset(A, B, WA, WB, kA, kB, tau);
    single_run_elapsed = toc(run_tic);

    best_score_before_refine = fast_nn_rmse(A, MdlB, R, t);
    final_score_after_refine = best_score_before_refine;
    best_seed = base_seed;
    best_run = 1;
    refine_rmse = NaN;
    run_scores = best_score_before_refine;
    run_elapsed = single_run_elapsed;
    run_EMD = EMD;
    score_idx = [];
    elapsed_time = toc(total_tic);

    TM = eye(4);
    TM(1:3, 1:3) = R;
    TM(1:3, 4) = t;

    selection_metric = 'single_pass_full_point_nearest_neighbor_rmse';
    refine_metric = 'not_used';

    save(out_path, ...
        'TM', 'R', 't', 'CSA', 'CSB', 'FM', 'elapsed_time', 'ratio', 'EMD', 'tau', ...
        'n_repeats', 'seeds', 'best_seed', 'best_run', ...
        'best_score_before_refine', 'final_score_after_refine', ...
        'run_scores', 'run_elapsed', 'run_EMD', ...
        'refine_iter', 'refine_rmse', 'selection_metric', 'refine_metric', ...
        'score_sample_n', 'score_idx', 'trim_keep_ratio', 'coreset_method');

    fprintf('   Done! single_pass_score=%.6f | total_time=%.4fs\n', ...
        best_score_before_refine, elapsed_time);
end


function seeds = generate_spread_seeds(base_seed, n_repeats)
    max_seed = 2^31 - 1;
    seeds = zeros(1, n_repeats);
    c1 = 1664525;
    c2 = 1013904223;
    c3 = 2654435761;

    x = double(base_seed);
    for ii = 1:n_repeats
        x = mod(x * c1 + c2 + ii * c3, max_seed);
        seed_val = floor(x) + 1;
        if seed_val < 1 || seed_val > max_seed
            seed_val = mod(seed_val, max_seed) + 1;
        end
        seeds(ii) = seed_val;
    end

    while numel(unique(seeds)) < n_repeats
        used = unique(seeds, 'stable');
        n_missing = n_repeats - numel(used);
        extra = randi(max_seed, 1, n_missing);
        seeds = unique([used, extra], 'stable');
    end

    seeds = seeds(1:n_repeats);
end


function [R, t, EMD, FM, CSA, CSB] = run_one_coreset(A, B, WA, WB, kA, kB, tau)
    [CSA, CSWA] = KCenter(A, WA, kA);
    [CSB, CSWB] = KCenter(B, WB, kB);

    mean_CSA = CSA * CSWA';
    mean_CSB = CSB * CSWB';
    CSA_centered = CSA - mean_CSA;
    CSB_centered = CSB - mean_CSB;

    R0 = eye(3);
    [EMD, FM, R] = SinkhornInit(CSA_centered, CSB_centered, CSWA, CSWB, R0, tau);

    R = project_to_rotation(R);

    row_sum = sum(FM, 2) + eps;
    B_match = CSB * FM';
    B_match = B_match ./ row_sum';

    A_rot = R * CSA;
    w = row_sum / sum(row_sum);
    t = sum((B_match - A_rot) .* w', 2);
end


function [R, t, rmse] = refine_rigid_trimmed_points(A, MdlB, R, t, refine_iter, trim_keep_ratio)
    rmse = inf;
    trim_keep_ratio = min(1, max(0.05, trim_keep_ratio));

    for ii = 1:refine_iter
        A_trans = R * A + t;
        idx = nearest_neighbor_indices(MdlB, A_trans);
        B_corr = MdlB.B(:, idx);

        residual_before = sum((B_corr - A_trans).^2, 1);
        n_keep = max(3, round(trim_keep_ratio * numel(residual_before)));
        [~, ord] = sort(residual_before, 'ascend');
        keep = ord(1:n_keep);

        [dR, dt] = kabsch_columns(A_trans(:, keep), B_corr(:, keep));
        R = dR * R;
        t = dR * t + dt;

        A_refined = R * A + t;
        idx_after = nearest_neighbor_indices(MdlB, A_refined);
        B_after = MdlB.B(:, idx_after);
        residual_after = B_after - A_refined;
        rmse = sqrt(mean(sum(residual_after.^2, 1)));
    end
end


function score = fast_nn_rmse(A, MdlB, R, t)
    A_trans = R * A + t;
    idx = nearest_neighbor_indices(MdlB, A_trans);
    B_corr = MdlB.B(:, idx);
    residual = B_corr - A_trans;
    score = sqrt(mean(sum(residual.^2, 1)));
end


function MdlB = build_nn_model(B)
    MdlB = struct();
    MdlB.B = B;
    MdlB.use_kdtree = false;
    MdlB.kdtree = [];

    if exist('KDTreeSearcher', 'class') == 8 || exist('KDTreeSearcher', 'file') == 2
        try
            MdlB.kdtree = KDTreeSearcher(B');
            MdlB.use_kdtree = true;
            fprintf('   NN backend: cached KDTreeSearcher for B.\n');
            return;
        catch
            MdlB.use_kdtree = false;
        end
    end

    if exist('knnsearch', 'file') == 2
        fprintf('   NN backend: knnsearch without explicit KDTreeSearcher.\n');
    else
        fprintf('   NN backend: block pdist2 fallback.\n');
    end
end


function idx = nearest_neighbor_indices(MdlB, Q)
    if isfield(MdlB, 'use_kdtree') && MdlB.use_kdtree
        idx = knnsearch(MdlB.kdtree, Q');
        idx = idx(:)';
    elseif exist('knnsearch', 'file') == 2
        idx = knnsearch(MdlB.B', Q');
        idx = idx(:)';
    else
        Nq = size(Q, 2);
        idx = zeros(1, Nq);
        block = 1000;
        Bt = MdlB.B';
        for s = 1:block:Nq
            e = min(Nq, s + block - 1);
            D = pdist2(Q(:, s:e)', Bt, 'squaredeuclidean');
            [~, local_idx] = min(D, [], 2);
            idx(s:e) = local_idx';
        end
    end
end


function [R, t] = kabsch_columns(X, Y)
    mx = mean(X, 2);
    my = mean(Y, 2);
    Xc = X - mx;
    Yc = Y - my;
    H = Xc * Yc';
    [U, ~, V] = svd(H);
    R = V * U';
    if det(R) < 0
        V(:, end) = -V(:, end);
        R = V * U';
    end
    t = my - R * mx;
end


function R = project_to_rotation(R)
    [U, ~, V] = svd(R);
    R = U * V';
    if det(R) < 0
        V(:, end) = -V(:, end);
        R = U * V';
    end
end
