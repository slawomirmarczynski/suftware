
class SurvivalAnalyzer:
    """Performs survival analysis using SAFTy (both SAFTy-KM and SAFTy-Cox)

    parameters
    ----------

    data_df: (pd.DataFrame)
        Pandas dataframe containing event times, event flags (optional), and
        covariates (optional).

    time_col: (str)
        The name of the event time column in data_df.

    flag_col: (str, None)
        The name of the column in data_df containing event flags. True
        indicates an observed event, False indicates a right-censored event.
        If None, all events will assume to be observed.

    covariate_cols: (list of str, None)
        The name of the columns in data_df containing covariates, which must be
        castable as floats. If None, SAFTy-KM will be run. Otherwise, if
        specified, SAFTy-Cox will be run.

    grid: (1D np.array)
        An array of evenly spaced time grid points on which survival curves
        will be estimated. Default value is ``None``, in which case the grid is
        set automatically.

    grid_spacing: (float > 0)
        The time between adjacent grid points. Default value
        is ``None``, in which case this spacing is set automatically.

    num_grid_points: (int)
        The number of grid points to draw within the time domain. Restricted
        to ``2*alpha <= num_grid_points <= 1000``. Default value is ``None``, in
        which case the number of grid points is chosen automatically.

    bounding_box: ([float, float])
        The boundaries of the time domain, within which survival curves
        will be estimated. Default value is ``None``, in which case the
        bounding box is set automatically to encompass all of the data.

    alpha: (int)
        The order of time derivative constrained in the definition of
        smoothness. Restricted to ``1 <= alpha <= 4``. Default value is 3.

    num_posterior_samples: (int >= 0)
        Number of samples to draw from the Bayesian posterior. Restricted to
        0 <= num_posterior_samples <= MAX_NUM_POSTERIOR_SAMPLES.

    max_u_step: (float > 0)
        Upper bound on the amount by which the parameter ``u``
        in the SAFTy algorithm is incremented when tracing the MAP curve.
        Default value is 1.0.

    tollerance: (float > 0)
        Sets the convergence criterion for the corrector algorithm used in
        tracing the MAP curve.

    resolution: (float > 0)
        The maximum geodesic distance allowed for neighboring points
        on the MAP curve.

    sample_only_at_tau_star: (boolean)
        Specifies whether to let tau vary when sampling from the Bayesian
        posterior.

    max_log_evidence_ratio_drop: (float > 0)
        If set, MAP curve tracing will terminate prematurely when
        max_log_evidence - current_log_evidence >  max_log_evidence_ratio_drop.

    evaluation_method_for_Z: (string)
        Method of evaluation of partition function Z. Possible values:
        'Lap'      : Laplace approximation (default).
        'Lap+Imp'  : Laplace approximation + importance sampling.
        'Lap+Fey'  : Laplace approximation + Feynman diagrams.

    num_samples_for_Z: (int >= 0)
        Number of posterior samples to use when evaluating the partition
        function Z. Only has an affect when
        ``evaluation_method_for_Z = 'Lap+Imp'``.

    seed: (int)
        Seed provided to the random number generator before density estimation
        commences. For development purposes only.

    print_u: (bool)
        Whether to print the values of ``u`` while tracing the MAP curve.
        For development purposes only.

    attributes
    ----------

    grid:
        The time grid points at which survival curves are estimated.
        (1D np.array)

    grid_spacing:
        The distance between neighboring time grid points.
        (float > 0)

    num_grid_points:
        The number of time grid points used.
        (int)

    bounding_box:
        The boundaries of the time domain within which survival curves were
        estimated. ([float, float])

    kaplan_meier_curve:
        Values and uncertainties of the Kaplan-Meier survival curve.
        (pd.DataFrame)

    cox_baseline_curve:
        Values and uncertainties for the baseline survival curve inferred using
        Cox regression. None if no covariates are specified. (pd.DataFrame)

    cox_effects:
        Values and uncertainties for the effects inferred by cox regression.
        None if no covariates are specified. (pd.DataFrame)

    values:
        The values of the optimal survival curve at each time grid point.
        (1D np.array)

    sample_values:
        The values of the posterior sampled survival curves at each time grid
        point. The first index specifies time grid points, the second posterior
        samples. (2D np.array)

    sample_weights:
        The importance weights corresponding to each posterior sample.
        (1D np.array)

    taus:
        The smoothness length scales at which the MAP curve was computed.
        (np.array)

    log_Es:
        The log evidence ratio values (Kinney, 2015, Eq. 27) at each length
        scale along the MAP curve. (np.array)

    max_log_E:
        The log evidence ratio at the optimal length scale. (float)

    runtime:
        The amount of time (in seconds) taken to execute.

    """