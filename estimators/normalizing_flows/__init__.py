from estimators.normalizing_flows.InvertedPlanarFlow import InvertedPlanarFlow
from estimators.normalizing_flows.InvertedRadialFlow import InvertedRadialFlow
from estimators.normalizing_flows.AffineFlow import AffineFlow

FLOWS = {"planar": InvertedPlanarFlow, "radial": InvertedRadialFlow, "affine": AffineFlow}
