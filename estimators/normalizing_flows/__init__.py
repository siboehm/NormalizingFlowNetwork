from estimators.normalizing_flows.PlanarFlow import PlanarFlow
from estimators.normalizing_flows.RadialFlow import RadialFlow
from estimators.normalizing_flows.AffineFlow import AffineFlow

FLOWS = {"planar": PlanarFlow, "radial": RadialFlow, "affine": AffineFlow}
