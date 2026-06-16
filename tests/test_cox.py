"""Cox proportional-hazards model + concordance index (roadmap Phase 9)."""

import numpy as np

from focuslens.intervention.cox import CoxPH, concordance_index
from focuslens.intervention.synthetic import make_cox_dataset


def test_concordance_index_hand_cases():
    # Earlier event with higher risk -> perfectly concordant.
    assert concordance_index([2.0, 1.0], [5.0, 10.0], [1, 1]) == 1.0
    # Earlier event with lower risk -> fully discordant.
    assert concordance_index([1.0, 2.0], [5.0, 10.0], [1, 1]) == 0.0
    # Tie in risk counts as 0.5.
    assert concordance_index([1.0, 1.0], [5.0, 10.0], [1, 1]) == 0.5


def test_cox_fit_reaches_c_index_target():
    x, dur, ev, beta_true = make_cox_dataset(n=800, seed=0)
    split = int(0.8 * len(x))
    model = CoxPH().fit(x[:split], dur[:split], ev[:split], epochs=300)
    c = concordance_index(model.predict_risk(x[split:]), dur[split:], ev[split:])
    assert c > 0.70  # roadmap target

    # The strong positive-effect covariates are recovered with the right sign.
    strong = beta_true > 0.3
    assert np.all(np.sign(model.beta[strong]) == 1.0)


def test_predict_risk_shapes_and_state_roundtrip():
    x, dur, ev, _ = make_cox_dataset(n=120, seed=1)
    model = CoxPH().fit(x, dur, ev, epochs=50)
    assert isinstance(model.predict_risk(x[0]), float)  # single vector
    assert model.predict_risk(x[:10]).shape == (10,)  # batch

    restored = CoxPH.from_state_dict(model.state_dict())
    assert np.allclose(restored.predict_risk(x[:5]), model.predict_risk(x[:5]), atol=1e-5)
