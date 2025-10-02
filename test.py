import control
import dkpy

uncertainty_model = {"iA", "iI", "iM"}

dkpy.compute_uncertainty_residual_response(
    control.FrequencyResponseData([], []),
    control.FrequencyResponseList([]),
    uncertainty_model,
)
